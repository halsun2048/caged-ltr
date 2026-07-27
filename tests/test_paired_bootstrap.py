from __future__ import annotations

import numpy as np
import pytest

from caged_ltr.evaluation import paired_bootstrap_mean


def test_paired_bootstrap_is_deterministic_and_detects_positive_mean() -> None:
    differences = np.column_stack(
        (
            np.linspace(0.1, 0.3, 40),
            np.linspace(-0.2, 0.2, 40),
        )
    )
    progress: list[tuple[int, int]] = []
    first = paired_bootstrap_mean(
        differences,
        iterations=200,
        seed=17,
        batch_size=30,
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    second = paired_bootstrap_mean(
        differences,
        iterations=200,
        seed=17,
        batch_size=30,
    )

    assert first == second
    assert progress[-1] == (200, 200)
    assert first["mean"][0] == pytest.approx(0.2)
    assert first["ci95_percentile"]["lower"][0] > 0.0
    assert first["two_sided_p"][0] < 0.02
    assert first["ci95_percentile"]["lower"][1] < 0.0
    assert first["ci95_percentile"]["upper"][1] > 0.0


@pytest.mark.parametrize(
    ("differences", "iterations", "seed", "batch_size"),
    [
        (np.asarray([1.0]), 10, 1, 2),
        (np.asarray([1.0, np.nan]), 10, 1, 2),
        (np.asarray([1.0, 2.0]), 0, 1, 2),
        (np.asarray([1.0, 2.0]), 10, -1, 2),
        (np.asarray([1.0, 2.0]), 10, 1, 0),
    ],
)
def test_paired_bootstrap_rejects_invalid_inputs(
    differences: np.ndarray,
    iterations: int,
    seed: int,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_mean(
            differences,
            iterations=iterations,
            seed=seed,
            batch_size=batch_size,
        )
