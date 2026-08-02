"""Versioned feature builder shared by offline Gate training and serving."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .r16_service import Candidate, RankedCandidate, lexical_score

FEATURE_BUILDER_VERSION = "r19.0-exact-v1"
FEATURES = [
    "query_characters",
    "candidate_count",
    "margin",
    "top1_score",
    "top3_gap",
    "top5_gap",
    "score_mean",
    "score_std",
    "top1_lexical_overlap",
    "max_lexical_overlap",
    "mean_lexical_overlap",
    "top1_passage_characters",
    "mean_passage_characters",
]


def vector_from_ranked(
    query: str, candidates: Sequence[Candidate], ranked: Sequence[RankedCandidate]
) -> list[float]:
    """Build exactly the vector consumed by the deployable post-student Gate."""
    scores = [float(item.score) for item in ranked]
    top = scores[0] if scores else 0.0
    second = scores[1] if len(scores) > 1 else top
    top3 = scores[2] if len(scores) > 2 else second
    top5 = scores[4] if len(scores) > 4 else second
    overlaps = [lexical_score(query, item.text) for item in candidates]
    by_id = {item.item_id: item for item in candidates}
    top_item = by_id.get(ranked[0].item_id) if ranked else None
    mean_score = sum(scores) / max(len(scores), 1)
    variance = sum((value - mean_score) ** 2 for value in scores) / max(len(scores), 1)
    return [
        float(len(query)),
        float(len(candidates)),
        top - second,
        top,
        top - top3,
        top - top5,
        mean_score,
        math.sqrt(variance),
        lexical_score(query, top_item.text) if top_item else 0.0,
        max(overlaps, default=0.0),
        sum(overlaps) / max(len(overlaps), 1),
        float(len(top_item.text)) if top_item else 0.0,
        sum(len(item.text) for item in candidates) / max(len(candidates), 1),
    ]


def frame_matrix(frame: Any) -> np.ndarray:
    """Return a finite matrix from an R12/R19 metrics dataframe."""
    values = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return values.to_numpy(float)


def manifest_features(payload: Mapping[str, Any]) -> list[str]:
    return list(payload.get("features", []))
