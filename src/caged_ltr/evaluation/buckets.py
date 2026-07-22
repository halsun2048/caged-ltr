"""Frequency-based Head/Torso/Tail assignment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FrequencyBucketBoundaries:
    tail_max: float
    head_min: float
    tail_quantile: float
    head_quantile: float


class FrequencyBucketer:
    """Fit bucket thresholds on training frequencies and reuse them unchanged."""

    def __init__(self, *, tail_quantile: float = 0.2, head_quantile: float = 0.8) -> None:
        if not 0.0 < tail_quantile < head_quantile < 1.0:
            raise ValueError("quantiles must satisfy 0 < tail < head < 1")
        self.tail_quantile = tail_quantile
        self.head_quantile = head_quantile
        self.boundaries: FrequencyBucketBoundaries | None = None

    def fit(self, frequencies: Sequence[int] | np.ndarray) -> FrequencyBucketer:
        values = np.asarray(frequencies, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("frequencies must be a finite, non-empty one-dimensional array")
        if (values < 0).any():
            raise ValueError("frequencies must be non-negative")
        tail_max, head_min = np.quantile(
            values,
            (self.tail_quantile, self.head_quantile),
            method="nearest",
        )
        self.boundaries = FrequencyBucketBoundaries(
            tail_max=float(tail_max),
            head_min=float(head_min),
            tail_quantile=self.tail_quantile,
            head_quantile=self.head_quantile,
        )
        return self

    def transform(self, frequencies: Sequence[int] | np.ndarray) -> np.ndarray:
        if self.boundaries is None:
            raise RuntimeError("fit must be called before transform")
        values = np.asarray(frequencies, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("frequencies must be a finite, non-negative one-dimensional array")
        if self.boundaries.tail_max >= self.boundaries.head_min:
            return np.full(values.shape, "torso", dtype="<U5")
        return np.where(
            values <= self.boundaries.tail_max,
            "tail",
            np.where(values >= self.boundaries.head_min, "head", "torso"),
        )

    def fit_transform(self, frequencies: Sequence[int] | np.ndarray) -> np.ndarray:
        return self.fit(frequencies).transform(frequencies)
