"""Semantic controls, embedding drift, and full-catalog sequence evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from caged_ltr.data.sequential import YelpSequenceData, load_yelp_author_sequences
from caged_ltr.models import FrozenSemanticOnly, SASRec, SASRecConfig
from caged_ltr.sequential.yelp_runner import YelpSASRecRunConfig


@dataclass(frozen=True, slots=True)
class FullCatalogEvaluation:
    """Full-catalog ranks, bucket metrics, and protocol metadata."""

    ranks: dict[str, np.ndarray]
    metrics: dict[str, dict[str, Any]]
    protocol: dict[str, Any]


def semantic_control(
    semantics: np.ndarray,
    *,
    kind: str,
    seed: int,
) -> np.ndarray:
    """Create a deterministic real, row-shuffled, or moment-matched control."""
    values = np.asarray(semantics, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("semantics must be a finite matrix")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if kind == "real":
        return values.copy()
    generator = np.random.default_rng(seed)
    if kind == "shuffled":
        return values[generator.permutation(values.shape[0])].copy()
    if kind == "matched_random":
        random = generator.standard_normal(values.shape)
        random -= random.mean(axis=0, keepdims=True)
        random_scale = random.std(axis=0, keepdims=True)
        random = np.divide(
            random,
            random_scale,
            out=np.zeros_like(random),
            where=random_scale > 1e-12,
        )
        matched = random * values.std(axis=0, keepdims=True) + values.mean(
            axis=0, keepdims=True
        )
        return matched.astype(np.float32)
    raise ValueError("kind must be real, shuffled, or matched_random")


def _statistics(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "median": float(np.median(array)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def checkpoint_embedding_drift(
    checkpoint_path: Path,
    initial_semantics: np.ndarray,
) -> dict[str, Any]:
    """Measure per-item movement from LLMInit initialization to a checkpoint."""
    initial = np.asarray(initial_semantics, dtype=np.float64)
    if initial.ndim != 2 or not np.isfinite(initial).all():
        raise ValueError("initial_semantics must be a finite matrix")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or "item_embedding.weight" not in state:
        raise ValueError("checkpoint does not contain item_embedding.weight")
    trained = state["item_embedding.weight"][1:].detach().cpu().numpy().astype(np.float64)
    if trained.shape != initial.shape or not np.isfinite(trained).all():
        raise ValueError("checkpoint item embeddings do not align with initialization")
    initial_norm = np.linalg.norm(initial, axis=1)
    trained_norm = np.linalg.norm(trained, axis=1)
    denominator = initial_norm * trained_norm
    cosine = np.divide(
        (initial * trained).sum(axis=1),
        denominator,
        out=np.zeros_like(initial_norm),
        where=denominator > 1e-12,
    )
    displacement = np.linalg.norm(trained - initial, axis=1)
    relative_displacement = np.divide(
        displacement,
        initial_norm,
        out=np.zeros_like(displacement),
        where=initial_norm > 1e-12,
    )
    return {
        "cosine_initial_to_checkpoint": _statistics(cosine),
        "l2_displacement": _statistics(displacement),
        "relative_l2_displacement": _statistics(relative_displacement),
        "fraction_cosine_below_0p9": float((cosine < 0.9).mean()),
        "fraction_cosine_below_0p5": float((cosine < 0.5).mean()),
    }


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _left_pad(values: np.ndarray, max_length: int) -> np.ndarray:
    output = np.zeros(max_length, dtype=np.int64)
    selected = values[-max_length:]
    output[-len(selected) :] = selected
    return output


def _bucket_metrics(
    ranks: np.ndarray,
    buckets: np.ndarray,
    *,
    top_k: int,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for bucket in ("overall", *sorted(set(buckets.tolist()))):
        mask = np.ones(ranks.shape, dtype=bool) if bucket == "overall" else buckets == bucket
        selected = ranks[mask]
        hits = selected < top_k
        output[bucket] = {
            "count": int(selected.size),
            f"Hit@{top_k}": float(hits.mean()) if selected.size else 0.0,
            f"NDCG@{top_k}": (
                float(np.where(hits, 1.0 / np.log2(selected + 2.0), 0.0).mean())
                if selected.size
                else 0.0
            ),
        }
    return output


def _metrics(
    ranks: np.ndarray,
    data: YelpSequenceData,
    user_offsets: np.ndarray,
    targets: np.ndarray,
    *,
    top_k: int,
) -> dict[str, Any]:
    target_offsets = targets - 1
    return {
        "user_frequency": _bucket_metrics(
            ranks,
            np.asarray(data.user_frequency_buckets)[user_offsets],
            top_k=top_k,
        ),
        "user_paper": _bucket_metrics(
            ranks,
            np.asarray(data.user_paper_buckets)[user_offsets],
            top_k=top_k,
        ),
        "item_frequency": _bucket_metrics(
            ranks,
            np.asarray(data.item_frequency_buckets)[target_offsets],
            top_k=top_k,
        ),
        "item_paper": _bucket_metrics(
            ranks,
            np.asarray(data.item_paper_buckets)[target_offsets],
            top_k=top_k,
        ),
    }


def _masked_zscore(scores: torch.Tensor, excluded: torch.Tensor) -> torch.Tensor:
    eligible = ~excluded
    counts = eligible.sum(dim=1, keepdim=True).clamp_min(1)
    selected = scores.masked_fill(excluded, 0.0)
    mean = selected.sum(dim=1, keepdim=True) / counts
    centered = (scores - mean).masked_fill(excluded, 0.0)
    scale = torch.sqrt(centered.square().sum(dim=1, keepdim=True) / counts)
    return torch.where(scale > 1e-12, centered / scale.clamp_min(1e-12), centered)


def _stable_ranks(
    scores: torch.Tensor,
    targets: torch.Tensor,
    excluded: torch.Tensor,
) -> torch.Tensor:
    values = scores.masked_fill(excluded, -torch.inf)
    target_columns = targets - 1
    target_scores = values.gather(1, target_columns.unsqueeze(1))
    candidate_columns = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
    ahead = values > target_scores
    tied_ahead = (values == target_scores) & (
        candidate_columns < target_columns.unsqueeze(1)
    )
    return (ahead | tied_ahead).sum(dim=1)


def _model_config(config: YelpSASRecRunConfig, num_items: int) -> SASRecConfig:
    return SASRecConfig(
        num_items=num_items,
        max_length=config.max_length,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )


def evaluate_full_catalog(
    config: YelpSASRecRunConfig,
    *,
    checkpoint_path: Path,
    semantic_variants: Mapping[str, np.ndarray] | None = None,
    semantic_weight: float = 0.25,
    gated_residual_weight: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> FullCatalogEvaluation:
    """Evaluate test against every item, excluding only the user's observed history."""
    if config.model not in {"sasrec", "llm_init"}:
        raise ValueError("full-catalog audit supports sasrec or llm_init checkpoints")
    if semantic_weight < 0:
        raise ValueError("semantic_weight must be non-negative")
    if gated_residual_weight is not None and gated_residual_weight < 0:
        raise ValueError("gated_residual_weight must be non-negative")
    device = _device(config.device)
    data = load_yelp_author_sequences(
        config.processed_dir,
        report_path=config.report_path,
        max_users=config.max_users,
    )
    eligible_users = np.flatnonzero(data.test_targets > 0)
    selected_user_offsets = eligible_users[
        : config.max_eval_users or len(eligible_users)
    ]
    selected_users = len(selected_user_offsets)
    semantics = None
    if config.model == "llm_init":
        if config.semantic_path is None:
            raise ValueError("semantic_path is required for llm_init")
        semantics = np.load(config.semantic_path, allow_pickle=False)
        expected = (data.num_items, config.hidden_dim)
        if semantics.shape != expected or not np.isfinite(semantics).all():
            raise ValueError(f"semantic matrix must be finite with shape {expected}")
    model_config = _model_config(config, data.num_items)
    collaborative = SASRec(
        model_config,
        item_initialization=semantics if config.model == "llm_init" else None,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    collaborative.load_state_dict(state["state_dict"])
    collaborative.eval()

    semantic_models: dict[str, FrozenSemanticOnly] = {}
    for name, variant in (semantic_variants or {}).items():
        array = np.asarray(variant, dtype=np.float32)
        expected = (data.num_items, config.hidden_dim)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"semantic variant {name!r} must have shape {expected}")
        semantic_models[name] = FrozenSemanticOnly(model_config, array).to(device).eval()

    rank_batches: dict[str, list[np.ndarray]] = {config.model: []}
    for name in semantic_models:
        rank_batches[f"semantic_only_{name}"] = []
        rank_batches[f"fusion_{name}"] = []
        if gated_residual_weight is not None:
            rank_batches[f"confidence_gate_{name}"] = []
    repeated_target_count = 0
    rarity = torch.zeros(data.num_items, dtype=torch.float32, device=device)
    item_buckets = np.asarray(data.item_frequency_buckets)
    rarity[torch.from_numpy(np.flatnonzero(item_buckets == "torso")).to(device)] = 0.5
    rare_offsets = np.flatnonzero(
        (item_buckets == "tail") | (item_buckets == "cold_start")
    )
    rarity[torch.from_numpy(rare_offsets).to(device)] = 1.0
    batch_size = config.evaluation_batch_size
    with torch.no_grad():
        for start in range(0, selected_users, batch_size):
            stop = min(start + batch_size, selected_users)
            histories: list[np.ndarray] = []
            batch_user_offsets = selected_user_offsets[start:stop]
            targets = data.test_targets[batch_user_offsets]
            excluded = torch.zeros(
                (stop - start, data.num_items),
                dtype=torch.bool,
                device=device,
            )
            for row, user_offset in enumerate(batch_user_offsets):
                history = np.append(
                    data.train_histories[user_offset],
                    data.valid_targets[user_offset],
                )
                target = int(targets[row])
                if target in history:
                    repeated_target_count += 1
                histories.append(_left_pad(history, config.max_length))
                known = np.unique(history[history != target])
                if known.size:
                    excluded[row, torch.from_numpy(known - 1).to(device)] = True
            sequences = torch.from_numpy(np.stack(histories)).to(device)
            target_tensor = torch.from_numpy(targets).to(device)
            collaborative_scores = collaborative.score_catalog(sequences)
            rank_batches[config.model].append(
                _stable_ranks(collaborative_scores, target_tensor, excluded).cpu().numpy()
            )
            normalized_collaborative = _masked_zscore(
                collaborative_scores,
                excluded,
            )
            eligible_collaborative = normalized_collaborative.masked_fill(
                excluded,
                -torch.inf,
            )
            top_two = torch.topk(
                eligible_collaborative,
                k=2,
                dim=1,
            ).values
            uncertainty = 1.0 / (
                1.0 + torch.clamp_min(top_two[:, 0] - top_two[:, 1], 0.0)
            )
            for name, semantic_model in semantic_models.items():
                semantic_scores = semantic_model.score_catalog(sequences)
                rank_batches[f"semantic_only_{name}"].append(
                    _stable_ranks(semantic_scores, target_tensor, excluded).cpu().numpy()
                )
                normalized_semantic = _masked_zscore(
                    semantic_scores,
                    excluded,
                )
                fused = normalized_collaborative + semantic_weight * normalized_semantic
                rank_batches[f"fusion_{name}"].append(
                    _stable_ranks(fused, target_tensor, excluded).cpu().numpy()
                )
                if gated_residual_weight is not None:
                    gated = (
                        fused
                        + gated_residual_weight
                        * uncertainty[:, None]
                        * rarity[None, :]
                        * normalized_semantic
                    )
                    rank_batches[f"confidence_gate_{name}"].append(
                        _stable_ranks(gated, target_tensor, excluded).cpu().numpy()
                    )
            if progress_callback is not None:
                progress_callback(stop, selected_users)

    ranks = {name: np.concatenate(parts) for name, parts in rank_batches.items()}
    targets = data.test_targets[selected_user_offsets]
    metrics = {
        name: _metrics(
            values,
            data,
            selected_user_offsets,
            targets,
            top_k=config.top_k,
        )
        for name, values in ranks.items()
    }
    return FullCatalogEvaluation(
        ranks=ranks,
        metrics=metrics,
        protocol={
            "split": "test",
            "candidate_catalog_size": data.num_items,
            "selected_users": selected_users,
            "history": "train plus validation target",
            "exclusion": "observed history only; repeated test target retained",
            "repeated_test_target_in_history": repeated_target_count,
            "tie_break": "stable ascending item ID",
            "fusion_normalization": "per-query z-score over eligible full catalog",
            "fusion_semantic_weight": semantic_weight,
            "gated_residual_weight": gated_residual_weight,
            "gate_query_uncertainty": (
                "1 / (1 + top1_minus_top2_collaborative_zscore)"
                if gated_residual_weight is not None
                else None
            ),
            "gate_item_rarity": (
                {"head": 0.0, "torso": 0.5, "tail": 1.0, "cold_start": 1.0}
                if gated_residual_weight is not None
                else None
            ),
            "data_fingerprint": data.fingerprint,
        },
    )
