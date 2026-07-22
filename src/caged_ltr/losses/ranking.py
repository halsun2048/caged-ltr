"""Tested objectives shared by reproduction and new-experiment work packages."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bayesian Personalized Ranking loss over aligned positive/negative scores."""
    if positive_scores.shape != negative_scores.shape or positive_scores.ndim == 0:
        raise ValueError("positive_scores and negative_scores must have equal non-scalar shapes")
    if mask is None:
        selected = torch.ones_like(positive_scores, dtype=torch.bool)
    else:
        if mask.shape != positive_scores.shape or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean and match score shape")
        selected = mask
    if not selected.any():
        return (positive_scores.sum() + negative_scores.sum()) * 0.0
    return functional.softplus(-(positive_scores[selected] - negative_scores[selected])).mean()


def _validate_flat_scores(scores: torch.Tensor, group_sizes: torch.Tensor) -> list[int]:
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if group_sizes.ndim != 1 or group_sizes.numel() == 0:
        raise ValueError("group_sizes must be a non-empty one-dimensional tensor")
    sizes = [int(size) for size in group_sizes.detach().cpu().tolist()]
    if any(size <= 0 for size in sizes) or sum(sizes) != scores.numel():
        raise ValueError("group_sizes must be positive and sum to the number of scores")
    return sizes


def pointwise_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross entropy on logits with an optional normalized sample weight."""
    if logits.ndim != 1 or targets.shape != logits.shape:
        raise ValueError("logits and targets must be equal one-dimensional tensors")
    if ((targets < 0) | (targets > 1)).any():
        raise ValueError("targets must lie in [0, 1]")
    losses = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weights is None:
        return losses.mean()
    if weights.shape != logits.shape or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be non-negative, non-zero, and match logits")
    return (losses * weights).sum() / weights.sum()


def ranknet_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    weight_by_label_gap: bool = False,
) -> torch.Tensor:
    """Macro request-level RankNet loss over every strictly ordered label pair."""
    if labels.shape != scores.shape:
        raise ValueError("labels must match scores")
    sizes = _validate_flat_scores(scores, group_sizes)
    group_losses: list[torch.Tensor] = []
    start = 0
    for size in sizes:
        group_scores = scores[start : start + size]
        group_labels = labels[start : start + size]
        label_gaps = group_labels[:, None] - group_labels[None, :]
        preferred = label_gaps > 0
        if preferred.any():
            score_gaps = group_scores[:, None] - group_scores[None, :]
            pair_losses = functional.softplus(-score_gaps[preferred])
            if weight_by_label_gap:
                pair_weights = label_gaps[preferred]
                pair_losses = pair_losses * pair_weights / pair_weights.mean()
            group_losses.append(pair_losses.mean())
        start += size
    return torch.stack(group_losses).mean() if group_losses else scores.sum() * 0.0


def listwise_kl_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Macro request-level KL(teacher || student) over list distributions."""
    if teacher_scores.shape != student_scores.shape:
        raise ValueError("teacher_scores must match student_scores")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    sizes = _validate_flat_scores(student_scores, group_sizes)
    group_losses: list[torch.Tensor] = []
    start = 0
    for size in sizes:
        student = student_scores[start : start + size] / temperature
        teacher = teacher_scores[start : start + size] / temperature
        teacher_probabilities = functional.softmax(teacher, dim=0)
        student_log_probabilities = functional.log_softmax(student, dim=0)
        teacher_log_probabilities = functional.log_softmax(teacher, dim=0)
        group_losses.append(
            torch.sum(
                teacher_probabilities * (teacher_log_probabilities - student_log_probabilities)
            )
            * temperature**2
        )
        start += size
    return torch.stack(group_losses).mean()
