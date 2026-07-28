"""Reproducible SASRec training on the author-processed Yelp split."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from caged_ltr.data.sequential import (
    SASRecEvaluationDataset,
    SASRecTrainingDataset,
    YelpSequenceData,
    load_yelp_author_sequences,
)
from caged_ltr.models import (
    DualViewSASRec,
    FrozenRawSemanticSASRec,
    FrozenSemanticLateFusion,
    FrozenSemanticOnly,
    SASRec,
    SASRecConfig,
)
from caged_ltr.reproducibility import seed_everything, sha256_file, write_environment


@dataclass(frozen=True, slots=True)
class YelpSASRecRunConfig:
    processed_dir: Path
    report_path: Path
    output_dir: Path
    model: str = "sasrec"
    semantic_path: Path | None = None
    raw_semantic_path: Path | None = None
    seed: int = 42
    evaluation_seed: int = 20240722
    max_users: int | None = None
    max_eval_users: int | None = None
    max_length: int = 200
    hidden_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 1
    dropout: float = 0.5
    semantic_weight: float = 1.0
    batch_size: int = 128
    evaluation_batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    max_epochs: int = 200
    patience: int = 20
    evaluation_negatives: int = 100
    top_k: int = 10
    device: str = "cpu"
    test_after_selection: bool = True

    def __post_init__(self) -> None:
        supported_models = {
            "sasrec",
            "llm_init",
            "semantic_only",
            "late_fusion",
            "dual_view",
            "dual_view_no_ca",
            "dual_view_unshared",
            "dual_view_capacity",
            "raw_semantic_only",
        }
        if self.model not in supported_models:
            raise ValueError(
                f"model must be one of {sorted(supported_models)}"
            )
        if self.model not in {"sasrec", "raw_semantic_only"} and self.semantic_path is None:
            raise ValueError("semantic_path is required by semantic model variants")
        if (
            self.model.startswith("dual_view") or self.model == "raw_semantic_only"
        ) and self.raw_semantic_path is None:
            raise ValueError("raw_semantic_path is required by raw-semantic variants")
        positive = (
            self.max_length,
            self.hidden_dim,
            self.num_blocks,
            self.num_heads,
            self.batch_size,
            self.evaluation_batch_size,
            self.max_epochs,
            self.patience,
            self.evaluation_negatives,
            self.top_k,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("model, training, and evaluation sizes must be positive")
        if min(self.seed, self.evaluation_seed) < 0:
            raise ValueError("training and evaluation seeds must be non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("seed and optimizer settings are invalid")
        if self.max_users is not None and self.max_users <= 0:
            raise ValueError("max_users must be positive")
        if self.max_eval_users is not None and self.max_eval_users <= 0:
            raise ValueError("max_eval_users must be positive")
        if self.top_k > self.evaluation_negatives + 1:
            raise ValueError("top_k cannot exceed the sampled candidate count")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")

    @classmethod
    def from_yaml(cls, path: Path, **overrides: Any) -> YelpSASRecRunConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        selected_overrides = {
            key: value for key, value in overrides.items() if value is not None
        }
        values = {**payload, **selected_overrides}
        for key in (
            "processed_dir",
            "report_path",
            "output_dir",
            "semantic_path",
            "raw_semantic_path",
        ):
            if values.get(key) is not None:
                values[key] = Path(values[key])
        return cls(**values)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _load_semantics(config: YelpSASRecRunConfig, num_items: int) -> np.ndarray | None:
    if config.model in {"sasrec", "raw_semantic_only"} or config.semantic_path is None:
        return None
    array = np.load(config.semantic_path, allow_pickle=False)
    if array.ndim != 2 or array.shape[0] != num_items or not np.isfinite(array).all():
        raise ValueError("semantic NPY must be finite and contain one row per item")
    return np.asarray(array, dtype=np.float32)


def _load_raw_semantics(
    config: YelpSASRecRunConfig, num_items: int
) -> np.ndarray | None:
    if not (
        config.model.startswith("dual_view") or config.model == "raw_semantic_only"
    ):
        return None
    if config.raw_semantic_path is None:
        raise ValueError("raw_semantic_path was not configured")
    array = np.load(config.raw_semantic_path, allow_pickle=False, mmap_mode="r")
    if array.ndim != 2 or array.shape[0] != num_items or not np.isfinite(array).all():
        raise ValueError("raw semantic NPY must be finite with one row per item")
    return np.asarray(array, dtype=np.float32)


def _build_model(
    config: YelpSASRecRunConfig,
    data: YelpSequenceData,
    semantic_items: np.ndarray | None,
    raw_semantic_items: np.ndarray | None = None,
) -> SASRec | FrozenSemanticOnly | DualViewSASRec | FrozenRawSemanticSASRec:
    model_config = SASRecConfig(
        num_items=data.num_items,
        max_length=config.max_length,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout,
        semantic_weight=config.semantic_weight,
    )
    if config.model == "sasrec":
        return SASRec(model_config)
    if config.model == "raw_semantic_only":
        if raw_semantic_items is None:
            raise ValueError("raw semantic item array was not loaded")
        return FrozenRawSemanticSASRec(model_config, raw_semantic_items)
    if semantic_items is None:
        raise ValueError("semantic item array was not loaded")
    if config.model == "llm_init":
        return SASRec(model_config, item_initialization=semantic_items)
    if config.model == "semantic_only":
        return FrozenSemanticOnly(model_config, semantic_items)
    if config.model == "late_fusion":
        return FrozenSemanticLateFusion(model_config, semantic_items)
    if raw_semantic_items is None:
        raise ValueError("raw semantic item array was not loaded")
    return DualViewSASRec(
        model_config,
        raw_semantic_items,
        semantic_items,
        use_cross_attention=config.model in {"dual_view", "dual_view_unshared"},
        share_encoder=config.model != "dual_view_unshared",
        capacity_control=config.model == "dual_view_capacity",
    )


def _bucket_metrics(
    ranks: np.ndarray,
    buckets: list[str],
    *,
    top_k: int,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for bucket in ["overall", *sorted(set(buckets))]:
        mask = (
            np.ones(ranks.shape, dtype=bool)
            if bucket == "overall"
            else np.asarray(buckets) == bucket
        )
        selected = ranks[mask]
        hits = selected < top_k
        result[bucket] = {
            "count": int(selected.size),
            f"Hit@{top_k}": float(hits.mean()) if selected.size else 0.0,
            f"NDCG@{top_k}": (
                float(np.where(hits, 1.0 / np.log2(selected + 2.0), 0.0).mean())
                if selected.size
                else 0.0
            ),
        }
    return result


def _evaluate(
    model: SASRec | FrozenSemanticOnly | DualViewSASRec | FrozenRawSemanticSASRec,
    data: YelpSequenceData,
    config: YelpSASRecRunConfig,
    *,
    split: str,
    device: torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = SASRecEvaluationDataset(
        data,
        split=split,
        max_length=config.max_length,
        num_negatives=config.evaluation_negatives,
        seed=config.evaluation_seed,
        max_users=config.max_eval_users,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    all_ranks: list[np.ndarray] = []
    all_users: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    processed = 0
    with torch.no_grad():
        for sequences, candidates, user_offsets, targets in loader:
            scores = model.score_candidates(sequences.to(device), candidates.to(device))
            order = torch.argsort(scores, dim=1, descending=True, stable=True)
            ranks = torch.argsort(order, dim=1, stable=True)[:, 0]
            all_ranks.append(ranks.cpu().numpy())
            all_users.append(user_offsets.numpy())
            all_targets.append(targets.numpy())
            processed += int(sequences.shape[0])
            if progress_callback is not None:
                progress_callback(processed, len(dataset))
    rank_array = np.concatenate(all_ranks)
    user_offsets = np.concatenate(all_users)
    target_ids = np.concatenate(all_targets)
    user_frequency = [data.user_frequency_buckets[index] for index in user_offsets]
    user_paper = [data.user_paper_buckets[index] for index in user_offsets]
    item_frequency = [data.item_frequency_buckets[target - 1] for target in target_ids]
    item_paper = [data.item_paper_buckets[target - 1] for target in target_ids]
    metrics = {
        "user_frequency": _bucket_metrics(rank_array, user_frequency, top_k=config.top_k),
        "user_paper": _bucket_metrics(rank_array, user_paper, top_k=config.top_k),
        "item_frequency": _bucket_metrics(rank_array, item_frequency, top_k=config.top_k),
        "item_paper": _bucket_metrics(rank_array, item_paper, top_k=config.top_k),
    }
    records = [
        {
            "split": split,
            "user_idx": int(data.user_indices[user_offset]),
            "target_item_idx": int(target_id - 1),
            "rank": int(rank),
            "user_frequency_bucket": user_frequency[row],
            "user_paper_bucket": user_paper[row],
            "item_frequency_bucket": item_frequency[row],
            "item_paper_bucket": item_paper[row],
        }
        for row, (user_offset, target_id, rank) in enumerate(
            zip(user_offsets, target_ids, rank_array, strict=True)
        )
    ]
    return metrics, records


def _serialized_config(config: YelpSASRecRunConfig) -> dict[str, Any]:
    values = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def run_yelp_sasrec(
    config: YelpSASRecRunConfig,
    *,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train with validation-only early stopping and optionally evaluate test once."""
    seed_everything(config.seed)
    device = _device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_yelp_author_sequences(
        config.processed_dir,
        report_path=config.report_path,
        max_users=config.max_users,
    )
    semantic_items = _load_semantics(config, data.num_items)
    raw_semantic_items = _load_raw_semantics(config, data.num_items)
    model = _build_model(
        config, data, semantic_items, raw_semantic_items
    ).to(device)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    if isinstance(model, FrozenSemanticOnly):
        valid_metrics, _ = _evaluate(model, data, config, split="valid", device=device)
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        training_users = 0
        print(
            json.dumps(
                {
                    "model": config.model,
                    "validation_NDCG": valid_metrics["item_frequency"]["overall"][
                        f"NDCG@{config.top_k}"
                    ],
                }
            ),
            flush=True,
        )
    else:
        training_data = SASRecTrainingDataset(
            data, max_length=config.max_length, seed=config.seed
        )
        training_users = len(training_data)
        shuffle_generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(
            training_data,
            batch_size=config.batch_size,
            shuffle=True,
            generator=shuffle_generator,
            num_workers=0,
        )
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        best_score = -1.0
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        for epoch in range(1, config.max_epochs + 1):
            training_data.set_epoch(epoch)
            model.train()
            losses: list[float] = []
            for sequences, positives, negatives in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(
                    sequences.to(device),
                    positives.to(device),
                    negatives.to(device),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            valid_metrics, _ = _evaluate(model, data, config, split="valid", device=device)
            valid_score = float(
                valid_metrics["item_frequency"]["overall"][f"NDCG@{config.top_k}"]
            )
            epoch_record = {
                "epoch": epoch,
                "train_bpr": float(np.mean(losses)),
                f"valid_NDCG@{config.top_k}": valid_score,
            }
            history.append(epoch_record)
            if valid_score > best_score:
                best_score = valid_score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {"state_dict": best_state},
                    config.output_dir / "best_model.pt",
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
            progress_record = {
                **epoch_record,
                f"best_NDCG@{config.top_k}": best_score,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
            }
            if epoch_callback is None:
                print(json.dumps(progress_record), flush=True)
            else:
                epoch_callback(progress_record)
            if stale_epochs >= config.patience:
                break
        if best_state is None:
            raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_metrics, validation_records = _evaluate(
        model, data, config, split="valid", device=device
    )
    if config.test_after_selection:
        test_metrics, test_records = _evaluate(
            model, data, config, split="test", device=device
        )
    else:
        test_metrics, test_records = None, []
    torch.save({"state_dict": best_state}, config.output_dir / "best_model.pt")
    pd.DataFrame([*validation_records, *test_records]).to_parquet(
        config.output_dir / "predictions.parquet", index=False
    )
    summary: dict[str, Any] = {
        "model": config.model,
        "seed": config.seed,
        "device": str(device),
        "data_fingerprint": data.fingerprint,
        "semantic_sha256": (
            sha256_file(config.semantic_path)
            if semantic_items is not None and config.semantic_path is not None
            else None
        ),
        "raw_semantic_sha256": (
            sha256_file(config.raw_semantic_path)
            if raw_semantic_items is not None
            and config.raw_semantic_path is not None
            else None
        ),
        "selected_users": len(data.train_histories),
        "training_users": training_users,
        "num_items": data.num_items,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "frozen_semantic_values": (
                int(model.semantic_items.numel())
                if isinstance(model, FrozenSemanticLateFusion | FrozenSemanticOnly)
                else int(getattr(model, "frozen_semantic_values", 0))
            ),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "protocol": {
            "train_objective": (
                "none; parameter-free semantic baseline"
                if isinstance(model, FrozenSemanticOnly)
                else "BPR over next-item positions"
            ),
            "validation_history": "train only",
            "test_history": "train plus validation target",
            "evaluation": f"target plus {config.evaluation_negatives} fixed unseen negatives",
            "evaluation_seed": config.evaluation_seed,
            "checkpoint_selection": (
                "not applicable"
                if isinstance(model, FrozenSemanticOnly)
                else f"validation NDCG@{config.top_k}"
            ),
            "test_usage": (
                "once after checkpoint selection"
                if config.test_after_selection
                else "not evaluated; validation-only run"
            ),
            "architecture": (
                {
                    "views": "frozen raw semantic adapter plus trainable PCA64 collaborative",
                    "shared_sequence_encoder": model.share_encoder,
                    "bidirectional_cross_attention": model.use_cross_attention,
                    "capacity_matched_positionwise_control": model.capacity_control,
                    "cross_attention_mask": (
                        "causal plus padding"
                        if model.use_cross_attention
                        else "not applicable"
                    ),
                    "author_code_difference": (
                        "cross attention adds a causal mask to prevent future leakage"
                    ),
                }
                if isinstance(model, DualViewSASRec)
                else (
                    {
                        "views": "frozen raw semantic adapter only",
                        "shared_sequence_encoder": False,
                        "bidirectional_cross_attention": False,
                        "capacity_matched_positionwise_control": False,
                        "cross_attention_mask": "not applicable",
                        "author_code_difference": None,
                    }
                    if isinstance(model, FrozenRawSemanticSASRec)
                    else None
                )
            ),
        },
    }
    (config.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(_serialized_config(config), sort_keys=False), encoding="utf-8"
    )
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_environment(config.output_dir / "environment.json", Path.cwd())
    return summary


def evaluate_yelp_test_checkpoint(
    config: YelpSASRecRunConfig,
    *,
    checkpoint_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Evaluate test once for a checkpoint selected strictly from validation results."""
    seed_everything(config.seed)
    device = _device(config.device)
    data = load_yelp_author_sequences(
        config.processed_dir,
        report_path=config.report_path,
        max_users=config.max_users,
    )
    semantic_items = _load_semantics(config, data.num_items)
    raw_semantic_items = _load_raw_semantics(config, data.num_items)
    model = _build_model(
        config, data, semantic_items, raw_semantic_items
    ).to(device)
    checkpoint = checkpoint_path or config.output_dir / "best_model.pt"
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    test_metrics, test_records = _evaluate(
        model,
        data,
        config,
        split="test",
        device=device,
        progress_callback=progress_callback,
    )
    prediction_path = config.output_dir / "predictions.parquet"
    validation_frame = pd.read_parquet(prediction_path)
    validation_frame = validation_frame[validation_frame["split"] != "test"]
    pd.concat((validation_frame, pd.DataFrame(test_records)), ignore_index=True).to_parquet(
        prediction_path,
        index=False,
    )
    summary_path = config.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test") is not None:
        raise ValueError("test metrics already exist for this run")
    summary["test"] = test_metrics
    summary["protocol"]["test_usage"] = "once after external validation-only selection"
    summary["protocol"]["final_test_evaluation_negatives"] = (
        config.evaluation_negatives
    )
    summary["protocol"]["final_test_evaluation"] = (
        f"target plus {config.evaluation_negatives} fixed unseen negatives"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return test_metrics
