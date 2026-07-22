"""Calibration visualizations generated directly from raw predictions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

from caged_ltr.evaluation.metrics import reliability_diagram

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def save_reliability_diagram(
    labels: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    output: Path,
    *,
    num_bins: int = 15,
    title: str = "Reliability diagram",
) -> None:
    """Save an equal-width reliability diagram as a deterministic PNG."""
    bins = reliability_diagram(labels, probabilities, num_bins=num_bins)
    centers = np.asarray([(row["lower"] + row["upper"]) / 2.0 for row in bins])
    positive_rates = np.asarray([row["positive_rate"] for row in bins])
    counts = np.asarray([row["count"] for row in bins])
    positive_rates = np.where(counts > 0, positive_rates, np.nan)

    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    axis.plot(centers, positive_rates, marker="o", color="#1f77b4", label="Observed")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Positive rate")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, metadata={"Software": "caged-ltr"})
    plt.close(figure)
