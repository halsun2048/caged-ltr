"""Validation-only score diagnostics and calibrated semantic fusion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from caged_ltr.data.sequential import (
    SASRecEvaluationDataset,
    load_yelp_author_sequences,
)
from caged_ltr.models import FrozenSemanticOnly, SASRec, SASRecConfig
from caged_ltr.reproducibility import sha256_file
from caged_ltr.sequential.yelp_runner import YelpSASRecRunConfig


@dataclass(frozen=True, slots=True)
class ValidationScoreBundle:
    """Fixed candidates and scores from collaborative and semantic branches."""

    collaborative: np.ndarray
    semantic: np.ndarray
    candidates: np.ndarray
    user_offsets: np.ndarray
    targets: np.ndarray
    user_frequency: np.ndarray
    user_paper: np.ndarray
    target_item_frequency: np.ndarray
    target_item_paper: np.ndarray
    item_frequency_table: np.ndarray
    item_paper_table: np.ndarray
    data_fingerprint: str
    checkpoint_sha256: str
    semantic_sha256: str
    seed: int
    evaluation_seed: int

    def __post_init__(self) -> None:
        rows = self.collaborative.shape[0]
        if self.collaborative.ndim != 2 or self.semantic.shape != self.collaborative.shape:
            raise ValueError("score matrices must be aligned and two-dimensional")
        if self.candidates.shape != self.collaborative.shape:
            raise ValueError("candidate matrix must align with score matrices")
        one_dimensional = (
            self.user_offsets,
            self.targets,
            self.user_frequency,
            self.user_paper,
            self.target_item_frequency,
            self.target_item_paper,
        )
        if any(values.shape != (rows,) for values in one_dimensional):
            raise ValueError("row metadata must align with score matrices")
        if not np.array_equal(self.candidates[:, 0], self.targets):
            raise ValueError("the target must be candidate zero")
        if not np.isfinite(self.collaborative).all() or not np.isfinite(self.semantic).all():
            raise ValueError("branch scores must be finite")
        if self.seed < 0 or self.evaluation_seed < 0:
            raise ValueError("seeds must be non-negative")


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    return torch.device(requested)


def _export_scores(
    config: YelpSASRecRunConfig,
    *,
    split: str,
    num_negatives: int,
    checkpoint_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ValidationScoreBundle:
    if config.model != "llm_init":
        raise ValueError("calibrated fusion requires a trained llm_init checkpoint")
    if config.semantic_path is None:
        raise ValueError("semantic_path is required")
    device = _device(config.device)
    data = load_yelp_author_sequences(
        config.processed_dir,
        report_path=config.report_path,
        max_users=config.max_users,
    )
    semantic_items = np.load(config.semantic_path, allow_pickle=False)
    expected = (data.num_items, config.hidden_dim)
    if semantic_items.shape != expected or not np.isfinite(semantic_items).all():
        raise ValueError(f"semantic item matrix must be finite with shape {expected}")
    semantic_items = np.asarray(semantic_items, dtype=np.float32)
    model_config = SASRecConfig(
        num_items=data.num_items,
        max_length=config.max_length,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )
    collaborative_model = SASRec(
        model_config,
        item_initialization=semantic_items,
    ).to(device)
    checkpoint = checkpoint_path or config.output_dir / "best_model.pt"
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    collaborative_model.load_state_dict(state["state_dict"])
    collaborative_model.eval()
    semantic_model = FrozenSemanticOnly(model_config, semantic_items).to(device).eval()
    dataset = SASRecEvaluationDataset(
        data,
        split=split,
        max_length=config.max_length,
        num_negatives=num_negatives,
        seed=config.evaluation_seed,
        max_users=config.max_eval_users,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        num_workers=0,
    )
    collaborative_batches: list[np.ndarray] = []
    semantic_batches: list[np.ndarray] = []
    candidate_batches: list[np.ndarray] = []
    user_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    processed = 0
    with torch.no_grad():
        for sequences, candidates, user_offsets, targets in loader:
            device_sequences = sequences.to(device)
            device_candidates = candidates.to(device)
            collaborative_batches.append(
                collaborative_model.score_candidates(
                    device_sequences, device_candidates
                )
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            semantic_batches.append(
                semantic_model.score_candidates(device_sequences, device_candidates)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            candidate_batches.append(candidates.numpy())
            user_batches.append(user_offsets.numpy())
            target_batches.append(targets.numpy())
            processed += int(sequences.shape[0])
            if progress_callback is not None:
                progress_callback(processed, len(dataset))
    candidates = np.concatenate(candidate_batches)
    users = np.concatenate(user_batches)
    targets = np.concatenate(target_batches)
    item_frequency_table = np.asarray(data.item_frequency_buckets)
    item_paper_table = np.asarray(data.item_paper_buckets)
    return ValidationScoreBundle(
        collaborative=np.concatenate(collaborative_batches),
        semantic=np.concatenate(semantic_batches),
        candidates=candidates,
        user_offsets=users,
        targets=targets,
        user_frequency=np.asarray(data.user_frequency_buckets)[users],
        user_paper=np.asarray(data.user_paper_buckets)[users],
        target_item_frequency=item_frequency_table[targets - 1],
        target_item_paper=item_paper_table[targets - 1],
        item_frequency_table=item_frequency_table,
        item_paper_table=item_paper_table,
        data_fingerprint=data.fingerprint,
        checkpoint_sha256=sha256_file(checkpoint),
        semantic_sha256=sha256_file(config.semantic_path),
        seed=config.seed,
        evaluation_seed=config.evaluation_seed,
    )


def export_validation_scores(
    config: YelpSASRecRunConfig,
    *,
    checkpoint_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ValidationScoreBundle:
    """Export validation scores; this API intentionally has no test-split option."""
    return _export_scores(
        config,
        split="valid",
        num_negatives=config.evaluation_negatives,
        checkpoint_path=checkpoint_path,
        progress_callback=progress_callback,
    )


def export_locked_test_scores(
    config: YelpSASRecRunConfig,
    *,
    num_negatives: int,
    checkpoint_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ValidationScoreBundle:
    """Score a locked LLMInit checkpoint and semantic branch on test once."""
    if config.test_after_selection:
        raise ValueError("set test_after_selection=false before locked test scoring")
    return _export_scores(
        config,
        split="test",
        num_negatives=num_negatives,
        checkpoint_path=checkpoint_path,
        progress_callback=progress_callback,
    )


def save_validation_scores(path: Path, bundle: ValidationScoreBundle) -> None:
    """Persist a validation-only score cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        collaborative=bundle.collaborative,
        semantic=bundle.semantic,
        candidates=bundle.candidates,
        user_offsets=bundle.user_offsets,
        targets=bundle.targets,
        user_frequency=bundle.user_frequency,
        user_paper=bundle.user_paper,
        target_item_frequency=bundle.target_item_frequency,
        target_item_paper=bundle.target_item_paper,
        item_frequency_table=bundle.item_frequency_table,
        item_paper_table=bundle.item_paper_table,
        data_fingerprint=np.asarray(bundle.data_fingerprint),
        checkpoint_sha256=np.asarray(bundle.checkpoint_sha256),
        semantic_sha256=np.asarray(bundle.semantic_sha256),
        seed=np.asarray(bundle.seed, dtype=np.int64),
        evaluation_seed=np.asarray(bundle.evaluation_seed, dtype=np.int64),
    )


def load_validation_scores(path: Path) -> ValidationScoreBundle:
    """Load a validation-only score cache with pickle explicitly disabled."""
    with np.load(path, allow_pickle=False) as payload:
        return ValidationScoreBundle(
            collaborative=payload["collaborative"],
            semantic=payload["semantic"],
            candidates=payload["candidates"],
            user_offsets=payload["user_offsets"],
            targets=payload["targets"],
            user_frequency=payload["user_frequency"],
            user_paper=payload["user_paper"],
            target_item_frequency=payload["target_item_frequency"],
            target_item_paper=payload["target_item_paper"],
            item_frequency_table=payload["item_frequency_table"],
            item_paper_table=payload["item_paper_table"],
            data_fingerprint=str(payload["data_fingerprint"].item()),
            checkpoint_sha256=str(payload["checkpoint_sha256"].item()),
            semantic_sha256=str(payload["semantic_sha256"].item()),
            seed=int(payload["seed"].item()),
            evaluation_seed=int(payload["evaluation_seed"].item()),
        )


def normalize_branch_scores(scores: np.ndarray, method: str) -> np.ndarray:
    """Normalize scores within each sampled candidate set."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite matrix with at least two candidates")
    if method == "zscore":
        centered = values - values.mean(axis=1, keepdims=True)
        scale = values.std(axis=1, keepdims=True)
        normalized = np.divide(
            centered,
            scale,
            out=np.zeros_like(centered),
            where=scale > 1e-12,
        )
        # Remove accumulated floating-point drift so row means are exactly stable
        # for reproducibility checks and downstream calibration.
        return normalized - normalized.mean(axis=1, keepdims=True)
    if method == "rank":
        order = np.argsort(-values, axis=1, kind="stable")
        ranks = np.argsort(order, axis=1, kind="stable")
        return 1.0 - ranks / (values.shape[1] - 1)
    raise ValueError("method must be 'zscore' or 'rank'")


def calibrated_scores(
    collaborative: np.ndarray,
    semantic: np.ndarray,
    *,
    method: str,
    semantic_weight: float,
) -> np.ndarray:
    """Fuse query-normalized collaborative and frozen semantic branches."""
    if semantic_weight < 0.0:
        raise ValueError("semantic_weight must be non-negative")
    if collaborative.shape != semantic.shape:
        raise ValueError("branch score matrices must align")
    return normalize_branch_scores(
        collaborative, method
    ) + semantic_weight * normalize_branch_scores(semantic, method)


def confidence_aware_scores(
    collaborative: np.ndarray,
    semantic: np.ndarray,
    candidate_buckets: np.ndarray,
    *,
    semantic_weight: float,
    base_semantic_weight: float = 0.25,
) -> np.ndarray:
    """Add an uncertainty-and-rarity semantic residual to fixed late fusion."""
    if semantic_weight < 0.0 or base_semantic_weight < 0.0:
        raise ValueError("semantic weights must be non-negative")
    if collaborative.shape != semantic.shape:
        raise ValueError("branch score matrices must align")
    buckets = np.asarray(candidate_buckets)
    if buckets.shape != collaborative.shape:
        raise ValueError("candidate_buckets must align with branch scores")
    allowed = {"head", "torso", "tail", "cold_start"}
    observed = set(buckets.ravel().tolist())
    if not observed <= allowed:
        raise ValueError(f"candidate_buckets contains unknown values: {observed - allowed}")
    collaborative_z = normalize_branch_scores(collaborative, "zscore")
    semantic_z = normalize_branch_scores(semantic, "zscore")
    top_two = np.partition(collaborative_z, -2, axis=1)[:, -2:]
    margin = top_two.max(axis=1) - top_two.min(axis=1)
    uncertainty = 1.0 / (1.0 + np.maximum(margin, 0.0))
    rarity = np.zeros(buckets.shape, dtype=np.float64)
    rarity[buckets == "torso"] = 0.5
    rarity[(buckets == "tail") | (buckets == "cold_start")] = 1.0
    gate = uncertainty[:, None] * rarity
    return (
        collaborative_z
        + base_semantic_weight * semantic_z
        + semantic_weight * gate * semantic_z
    )


def _bucket_metrics(
    ranks: np.ndarray,
    buckets: np.ndarray,
    *,
    top_k: int,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for bucket in ("overall", *sorted(set(buckets.tolist()))):
        mask = np.ones(ranks.shape, dtype=bool) if bucket == "overall" else buckets == bucket
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


def validation_metrics(
    bundle: ValidationScoreBundle,
    scores: np.ndarray,
    *,
    top_k: int,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Evaluate ranks for a cached target at candidate position zero."""
    values = np.asarray(scores)
    if values.shape != bundle.collaborative.shape or not np.isfinite(values).all():
        raise ValueError("scores must be finite and align with the cached validation bundle")
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.argsort(order, axis=1, kind="stable")[:, 0]
    return {
        "user_frequency": _bucket_metrics(ranks, bundle.user_frequency, top_k=top_k),
        "user_paper": _bucket_metrics(ranks, bundle.user_paper, top_k=top_k),
        "item_frequency": _bucket_metrics(
            ranks, bundle.target_item_frequency, top_k=top_k
        ),
        "item_paper": _bucket_metrics(ranks, bundle.target_item_paper, top_k=top_k),
    }


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


def _row_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_centered = first - first.mean(axis=1, keepdims=True)
    second_centered = second - second.mean(axis=1, keepdims=True)
    numerator = (first_centered * second_centered).sum(axis=1)
    denominator = np.linalg.norm(first_centered, axis=1) * np.linalg.norm(
        second_centered, axis=1
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def score_diagnostics(
    bundle: ValidationScoreBundle,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Describe branch scale, agreement, and popularity exposure on validation only."""
    candidate_buckets = bundle.item_frequency_table[bundle.candidates - 1]
    branch_statistics: dict[str, Any] = {}
    exposure: dict[str, Any] = {}
    for name, scores in (
        ("collaborative", bundle.collaborative),
        ("semantic", bundle.semantic),
    ):
        branch_statistics[name] = {
            "all_candidates": _statistics(scores),
            "targets": _statistics(scores[:, 0]),
            "candidate_item_frequency": {
                bucket: _statistics(scores[candidate_buckets == bucket])
                for bucket in sorted(set(candidate_buckets.ravel().tolist()))
            },
            "target_item_frequency": {
                bucket: _statistics(scores[:, 0][bundle.target_item_frequency == bucket])
                for bucket in sorted(set(bundle.target_item_frequency.tolist()))
            },
            "mean_query_std": float(scores.std(axis=1).mean()),
        }
        top_indices = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        top_candidates = np.take_along_axis(bundle.candidates, top_indices, axis=1)
        top_buckets = bundle.item_frequency_table[top_candidates - 1]
        exposure[name] = {
            bucket: float((top_buckets == bucket).mean())
            for bucket in sorted(set(bundle.item_frequency_table.tolist()))
        }
    score_correlations = _row_correlation(
        bundle.collaborative.astype(np.float64),
        bundle.semantic.astype(np.float64),
    )
    rank_correlations = _row_correlation(
        normalize_branch_scores(bundle.collaborative, "rank"),
        normalize_branch_scores(bundle.semantic, "rank"),
    )
    return {
        "split": "validation",
        "rows": int(bundle.collaborative.shape[0]),
        "candidates_per_row": int(bundle.collaborative.shape[1]),
        "branch_statistics": branch_statistics,
        "mean_query_std_ratio_collaborative_to_semantic": float(
            bundle.collaborative.std(axis=1).mean()
            / max(bundle.semantic.std(axis=1).mean(), 1e-12)
        ),
        "branch_score_correlation": _statistics(score_correlations),
        "branch_rank_correlation": _statistics(rank_correlations),
        f"top_{top_k}_item_frequency_exposure": exposure,
    }
