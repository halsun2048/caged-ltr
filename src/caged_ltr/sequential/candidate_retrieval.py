"""Full-catalog candidate-retrieval diagnostics for sequential recommenders."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from caged_ltr.data.sequential import YelpSequenceData, load_yelp_author_sequences
from caged_ltr.models import FrozenSemanticOnly, SASRec, SASRecConfig
from caged_ltr.sequential.yelp_runner import YelpSASRecRunConfig


@dataclass(frozen=True, slots=True)
class CandidateRetrievalEvaluation:
    """Per-user route hits, candidate counts, metrics, and protocol metadata."""

    hits: dict[str, dict[int, np.ndarray]]
    candidate_counts: dict[str, dict[int, np.ndarray]]
    metrics: dict[str, dict[str, Any]]
    protocol: dict[str, Any]


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _left_pad(values: np.ndarray, max_length: int) -> np.ndarray:
    output = np.zeros(max_length, dtype=np.int64)
    selected = values[-max_length:]
    if len(selected):
        output[-len(selected) :] = selected
    return output


def _model_config(config: YelpSASRecRunConfig, num_items: int) -> SASRecConfig:
    return SASRecConfig(
        num_items=num_items,
        max_length=config.max_length,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )


def _bucket_recall(
    hits: np.ndarray,
    buckets: np.ndarray,
    *,
    cutoff: int,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for bucket in ("overall", *sorted(set(buckets.tolist()))):
        mask = np.ones(hits.shape, dtype=bool) if bucket == "overall" else buckets == bucket
        selected = hits[mask]
        result[bucket] = {
            "count": int(selected.size),
            f"Recall@{cutoff}": float(selected.mean()) if selected.size else 0.0,
        }
    return result


def _route_metrics(
    hits: np.ndarray,
    data: YelpSequenceData,
    user_offsets: np.ndarray,
    targets: np.ndarray,
    *,
    cutoff: int,
) -> dict[str, Any]:
    target_offsets = targets - 1
    return {
        "user_frequency": _bucket_recall(
            hits,
            np.asarray(data.user_frequency_buckets)[user_offsets],
            cutoff=cutoff,
        ),
        "user_paper": _bucket_recall(
            hits,
            np.asarray(data.user_paper_buckets)[user_offsets],
            cutoff=cutoff,
        ),
        "item_frequency": _bucket_recall(
            hits,
            np.asarray(data.item_frequency_buckets)[target_offsets],
            cutoff=cutoff,
        ),
        "item_paper": _bucket_recall(
            hits,
            np.asarray(data.item_paper_buckets)[target_offsets],
            cutoff=cutoff,
        ),
    }


def _unique_union_counts(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    combined = torch.cat((left, right), dim=1)
    ordered = torch.sort(combined, dim=1).values
    return 1 + ordered[:, 1:].ne(ordered[:, :-1]).sum(dim=1)


def evaluate_full_catalog_retrieval(
    config: YelpSASRecRunConfig,
    *,
    checkpoint_path: Path,
    semantic_variants: Mapping[str, np.ndarray],
    cutoffs: Sequence[int] = (100, 500),
    split: str = "valid",
    progress_callback: Callable[[int, int], None] | None = None,
) -> CandidateRetrievalEvaluation:
    """Measure collaborative, semantic, and union candidate recall.

    The union route contains each branch's Top-K candidates, so its actual budget
    lies between K and 2K and is reported per user. This diagnostic does not train
    or rerank a model.
    """
    if split != "valid":
        raise ValueError("candidate-route selection is restricted to validation")
    if config.model not in {"sasrec", "llm_init"}:
        raise ValueError("retrieval audit supports sasrec or llm_init checkpoints")
    selected_cutoffs = tuple(sorted(set(int(value) for value in cutoffs)))
    if not selected_cutoffs or selected_cutoffs[0] <= 0:
        raise ValueError("cutoffs must contain positive integers")
    if not semantic_variants:
        raise ValueError("at least one semantic variant is required")

    device = _device(config.device)
    data = load_yelp_author_sequences(
        config.processed_dir,
        report_path=config.report_path,
        max_users=config.max_users,
    )
    if selected_cutoffs[-1] > data.num_items:
        raise ValueError("cutoffs cannot exceed the catalog size")
    eligible_users = np.flatnonzero(data.valid_targets > 0)
    user_offsets = eligible_users[: config.max_eval_users or len(eligible_users)]
    selected_users = len(user_offsets)

    initialization = None
    if config.model == "llm_init":
        if config.semantic_path is None:
            raise ValueError("semantic_path is required for llm_init")
        initialization = np.load(config.semantic_path, allow_pickle=False)
        expected = (data.num_items, config.hidden_dim)
        if initialization.shape != expected or not np.isfinite(initialization).all():
            raise ValueError(f"semantic matrix must be finite with shape {expected}")

    model_config = _model_config(config, data.num_items)
    collaborative = SASRec(model_config, item_initialization=initialization).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    collaborative.load_state_dict(state["state_dict"])
    collaborative.eval()

    semantic_models: dict[str, FrozenSemanticOnly] = {}
    for name, variant in semantic_variants.items():
        array = np.asarray(variant, dtype=np.float32)
        expected = (data.num_items, config.hidden_dim)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"semantic variant {name!r} must have shape {expected}")
        semantic_models[name] = FrozenSemanticOnly(model_config, array).to(device).eval()

    route_names = ["collaborative"]
    for name in semantic_models:
        route_names.extend((f"semantic_{name}", f"union_{name}"))
    hit_batches = {
        route: {cutoff: [] for cutoff in selected_cutoffs} for route in route_names
    }
    count_batches = {
        f"union_{name}": {cutoff: [] for cutoff in selected_cutoffs}
        for name in semantic_models
    }
    repeated_target_count = 0
    max_cutoff = selected_cutoffs[-1]
    batch_size = config.evaluation_batch_size

    with torch.no_grad():
        for start in range(0, selected_users, batch_size):
            stop = min(start + batch_size, selected_users)
            batch_user_offsets = user_offsets[start:stop]
            targets = data.valid_targets[batch_user_offsets]
            histories: list[np.ndarray] = []
            excluded = torch.zeros(
                (stop - start, data.num_items),
                dtype=torch.bool,
                device=device,
            )
            for row, user_offset in enumerate(batch_user_offsets):
                history = data.train_histories[user_offset]
                target = int(targets[row])
                if target in history:
                    repeated_target_count += 1
                histories.append(_left_pad(history, config.max_length))
                known = np.unique(history[history != target])
                if known.size:
                    excluded[row, torch.from_numpy(known - 1).to(device)] = True

            sequences = torch.from_numpy(np.stack(histories)).to(device)
            target_columns = torch.from_numpy(targets - 1).to(device).unsqueeze(1)
            collaborative_scores = collaborative.score_catalog(sequences).masked_fill(
                excluded,
                -torch.inf,
            )
            collaborative_top = torch.topk(
                collaborative_scores,
                k=max_cutoff,
                dim=1,
                largest=True,
                sorted=True,
            ).indices
            for cutoff in selected_cutoffs:
                collaborative_hit = collaborative_top[:, :cutoff].eq(
                    target_columns
                ).any(dim=1)
                hit_batches["collaborative"][cutoff].append(
                    collaborative_hit.cpu().numpy()
                )

            for name, semantic_model in semantic_models.items():
                semantic_scores = semantic_model.score_catalog(sequences).masked_fill(
                    excluded,
                    -torch.inf,
                )
                semantic_top = torch.topk(
                    semantic_scores,
                    k=max_cutoff,
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices
                for cutoff in selected_cutoffs:
                    collaborative_candidates = collaborative_top[:, :cutoff]
                    semantic_candidates = semantic_top[:, :cutoff]
                    collaborative_hit = collaborative_candidates.eq(
                        target_columns
                    ).any(dim=1)
                    semantic_hit = semantic_candidates.eq(target_columns).any(dim=1)
                    hit_batches[f"semantic_{name}"][cutoff].append(
                        semantic_hit.cpu().numpy()
                    )
                    hit_batches[f"union_{name}"][cutoff].append(
                        (collaborative_hit | semantic_hit).cpu().numpy()
                    )
                    count_batches[f"union_{name}"][cutoff].append(
                        _unique_union_counts(
                            collaborative_candidates,
                            semantic_candidates,
                        )
                        .cpu()
                        .numpy()
                    )
            if progress_callback is not None:
                progress_callback(stop, selected_users)

    hits = {
        route: {
            cutoff: np.concatenate(parts).astype(bool, copy=False)
            for cutoff, parts in by_cutoff.items()
        }
        for route, by_cutoff in hit_batches.items()
    }
    candidate_counts = {
        route: {
            cutoff: np.concatenate(parts).astype(np.int64, copy=False)
            for cutoff, parts in by_cutoff.items()
        }
        for route, by_cutoff in count_batches.items()
    }
    targets = data.valid_targets[user_offsets]
    metrics = {
        route: {
            str(cutoff): _route_metrics(
                route_hits,
                data,
                user_offsets,
                targets,
                cutoff=cutoff,
            )
            for cutoff, route_hits in by_cutoff.items()
        }
        for route, by_cutoff in hits.items()
    }
    return CandidateRetrievalEvaluation(
        hits=hits,
        candidate_counts=candidate_counts,
        metrics=metrics,
        protocol={
            "split": "validation",
            "test_accessed": False,
            "candidate_catalog_size": data.num_items,
            "selected_users": selected_users,
            "history": "training interactions only",
            "exclusion": "observed training history only; repeated target retained",
            "repeated_validation_target_in_history": repeated_target_count,
            "cutoffs": list(selected_cutoffs),
            "union_definition": "collaborative Top-K set union semantic Top-K",
            "union_budget": "between K and 2K; actual counts reported per user",
            "topk_tie_behavior": "PyTorch deterministic topk for the active backend",
            "data_fingerprint": data.fingerprint,
        },
    )
