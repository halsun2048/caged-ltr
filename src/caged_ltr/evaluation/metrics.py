"""Public ranking, discrimination, and calibration metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def _arrays(
    labels: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(labels, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.ndim != 1 or score_array.shape != label_array.shape or label_array.size == 0:
        raise ValueError("labels and scores must be non-empty one-dimensional arrays of equal size")
    if not np.isfinite(label_array).all() or not np.isfinite(score_array).all():
        raise ValueError("labels and scores must contain only finite values")
    return label_array, score_array


def _group_indices(group_ids: Sequence[str] | np.ndarray, size: int) -> list[np.ndarray]:
    if len(group_ids) != size:
        raise ValueError("group_ids must match labels")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[str(group_id)].append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def group_auc(
    labels: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    group_ids: Sequence[str] | np.ndarray,
) -> float:
    """Return impression-weighted GAUC, skipping groups with only one class."""
    label_array, score_array = _arrays(labels, scores)
    if not np.isin(label_array, (0.0, 1.0)).all():
        raise ValueError("GAUC requires binary labels")
    weighted_auc = 0.0
    valid_weight = 0
    for indices in _group_indices(group_ids, label_array.size):
        group_labels = label_array[indices]
        if np.unique(group_labels).size < 2:
            continue
        weight = indices.size
        weighted_auc += float(roc_auc_score(group_labels, score_array[indices])) * weight
        valid_weight += weight
    return weighted_auc / valid_weight if valid_weight else float("nan")


def expected_calibration_error(
    labels: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    num_bins: int = 15,
) -> float:
    """Compute equal-width ECE with bins in [0, 1]."""
    label_array, probability_array = _arrays(labels, probabilities)
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    if not np.isin(label_array, (0.0, 1.0)).all():
        raise ValueError("ECE requires binary labels")
    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1]")

    bin_ids = np.minimum((probability_array * num_bins).astype(np.int64), num_bins - 1)
    error = 0.0
    for bin_id in range(num_bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probability_array[mask].mean()) - float(label_array[mask].mean())
            )
    return error


def reliability_diagram(
    labels: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    num_bins: int = 15,
) -> list[dict[str, float | int]]:
    """Return raw bin statistics used to draw a reliability diagram."""
    label_array, probability_array = _arrays(labels, probabilities)
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    if not np.isin(label_array, (0.0, 1.0)).all():
        raise ValueError("reliability diagrams require binary labels")
    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1]")

    bin_ids = np.minimum((probability_array * num_bins).astype(np.int64), num_bins - 1)
    bins: list[dict[str, float | int]] = []
    for bin_id in range(num_bins):
        mask = bin_ids == bin_id
        bins.append(
            {
                "bin": bin_id,
                "lower": bin_id / num_bins,
                "upper": (bin_id + 1) / num_bins,
                "count": int(mask.sum()),
                "mean_probability": float(probability_array[mask].mean()) if mask.any() else 0.0,
                "positive_rate": float(label_array[mask].mean()) if mask.any() else 0.0,
            }
        )
    return bins


def classification_metrics(
    labels: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    num_bins: int = 15,
) -> dict[str, float]:
    """Compute AUC, LogLoss, Brier score, and ECE."""
    label_array, score_array = _arrays(labels, scores)
    _, probability_array = _arrays(labels, probabilities)
    if not np.isin(label_array, (0.0, 1.0)).all():
        raise ValueError("classification metrics require binary labels")
    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    clipped = np.clip(probability_array, 1e-7, 1.0 - 1e-7)
    auc = (
        float(roc_auc_score(label_array, score_array))
        if np.unique(label_array).size == 2
        else float("nan")
    )
    return {
        "AUC": auc,
        "LogLoss": float(
            -(label_array * np.log(clipped) + (1.0 - label_array) * np.log(1.0 - clipped)).mean()
        ),
        "Brier": float(np.mean((probability_array - label_array) ** 2)),
        "ECE": expected_calibration_error(label_array, probability_array, num_bins=num_bins),
    }


def ranking_metrics(
    labels: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    group_ids: Sequence[str] | np.ndarray,
    *,
    cutoffs: Sequence[int] = (5, 10, 20),
) -> dict[str, float]:
    """Compute macro request-level MRR, Recall@K, and NDCG@K."""
    label_array, score_array = _arrays(labels, scores)
    if (label_array < 0).any():
        raise ValueError("ranking labels must be non-negative")
    normalized_cutoffs = tuple(sorted(set(int(cutoff) for cutoff in cutoffs)))
    if not normalized_cutoffs or normalized_cutoffs[0] <= 0:
        raise ValueError("cutoffs must contain positive integers")

    groups = _group_indices(group_ids, label_array.size)
    mrr_values: list[float] = []
    recall_values: dict[int, list[float]] = {cutoff: [] for cutoff in normalized_cutoffs}
    ndcg_values: dict[int, list[float]] = {cutoff: [] for cutoff in normalized_cutoffs}

    for indices in groups:
        group_labels = label_array[indices]
        group_scores = score_array[indices]
        order = np.argsort(-group_scores, kind="stable")
        ranked_labels = group_labels[order]
        relevant = ranked_labels > 0
        relevant_positions = np.flatnonzero(relevant)
        reciprocal_rank = (
            1.0 / (int(relevant_positions[0]) + 1) if relevant_positions.size else 0.0
        )
        mrr_values.append(reciprocal_rank)

        total_relevant = int((group_labels > 0).sum())
        ideal_labels = np.sort(group_labels)[::-1]
        for cutoff in normalized_cutoffs:
            top_labels = ranked_labels[:cutoff]
            recall_values[cutoff].append(
                float((top_labels > 0).sum() / total_relevant) if total_relevant else 0.0
            )
            discounts = np.log2(np.arange(2, top_labels.size + 2, dtype=np.float64))
            dcg = float(np.sum(np.expm1(np.log(2.0) * top_labels) / discounts))
            ideal_top = ideal_labels[:cutoff]
            ideal_discounts = np.log2(np.arange(2, ideal_top.size + 2, dtype=np.float64))
            idcg = float(np.sum(np.expm1(np.log(2.0) * ideal_top) / ideal_discounts))
            ndcg_values[cutoff].append(dcg / idcg if idcg > 0.0 else 0.0)

    metrics = {"MRR": float(np.mean(mrr_values))}
    for cutoff in normalized_cutoffs:
        metrics[f"Recall@{cutoff}"] = float(np.mean(recall_values[cutoff]))
        metrics[f"NDCG@{cutoff}"] = float(np.mean(ndcg_values[cutoff]))
    return metrics


def evaluate_predictions(
    labels: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    group_ids: Sequence[str] | np.ndarray,
    *,
    cutoffs: Sequence[int] = (5, 10, 20),
    num_bins: int = 15,
) -> dict[str, float]:
    """Compute the full public metric suite for one prediction set."""
    metrics = classification_metrics(labels, scores, probabilities, num_bins=num_bins)
    metrics["GAUC"] = group_auc(labels, scores, group_ids)
    metrics.update(ranking_metrics(labels, scores, group_ids, cutoffs=cutoffs))
    return metrics
