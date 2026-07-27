"""Deterministic paired bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def paired_bootstrap_mean(
    differences: np.ndarray,
    *,
    iterations: int,
    seed: int,
    batch_size: int = 25,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Bootstrap column means by resampling aligned rows with replacement."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("differences must be a finite matrix with at least two rows")
    if iterations <= 0 or batch_size <= 0 or seed < 0:
        raise ValueError("iterations and batch_size must be positive; seed non-negative")

    rows, columns = values.shape
    probabilities = np.full(rows, 1.0 / rows, dtype=np.float64)
    samples = np.empty((iterations, columns), dtype=np.float64)
    generator = np.random.default_rng(seed)
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        weights = generator.multinomial(
            rows,
            probabilities,
            size=stop - start,
        )
        samples[start:stop] = weights @ values / rows
        if progress_callback is not None:
            progress_callback(stop, iterations)

    observed = values.mean(axis=0)
    lower, upper = np.quantile(samples, (0.025, 0.975), axis=0)
    probability_positive = (samples > 0.0).mean(axis=0)
    probability_non_positive = (samples <= 0.0).mean(axis=0)
    probability_negative = (samples < 0.0).mean(axis=0)
    probability_non_negative = (samples >= 0.0).mean(axis=0)
    two_sided_p = np.minimum(
        1.0,
        2.0
        * np.minimum(
            (probability_non_positive * iterations + 1.0) / (iterations + 1.0),
            (probability_non_negative * iterations + 1.0) / (iterations + 1.0),
        ),
    )
    return {
        "rows": rows,
        "iterations": iterations,
        "seed": seed,
        "mean": observed.tolist(),
        "bootstrap_standard_error": samples.std(axis=0, ddof=1).tolist(),
        "ci95_percentile": {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        },
        "probability_positive": probability_positive.tolist(),
        "probability_negative": probability_negative.tolist(),
        "two_sided_p": two_sided_p.tolist(),
    }
