from __future__ import annotations

import numpy as np
import pytest

from caged_ltr.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    group_auc,
    ranking_metrics,
    reliability_diagram,
)

LABELS = np.asarray([1, 0, 0, 0, 1], dtype=np.float64)
SCORES = np.asarray([0.9, 0.2, 0.1, 0.9, 0.8], dtype=np.float64)
GROUPS = np.asarray(["q1", "q1", "q1", "q2", "q2"])


def test_ranking_metrics_match_hand_calculation() -> None:
    metrics = ranking_metrics(LABELS, SCORES, GROUPS, cutoffs=(1, 2))

    assert metrics["MRR"] == pytest.approx(0.75)
    assert metrics["Recall@1"] == pytest.approx(0.5)
    assert metrics["Recall@2"] == pytest.approx(1.0)
    assert metrics["NDCG@1"] == pytest.approx(0.5)
    assert metrics["NDCG@2"] == pytest.approx((1.0 + 1.0 / np.log2(3.0)) / 2.0)


def test_auc_gauc_and_calibration_match_hand_calculation() -> None:
    probabilities = np.asarray([0.8, 0.2, 0.1, 0.7, 0.6])
    metrics = classification_metrics(LABELS, SCORES, probabilities, num_bins=2)

    assert metrics["AUC"] == pytest.approx(0.75)
    assert group_auc(LABELS, SCORES, GROUPS) == pytest.approx(0.6)
    assert metrics["Brier"] == pytest.approx(np.mean((probabilities - LABELS) ** 2))
    assert expected_calibration_error(LABELS, LABELS, num_bins=2) == pytest.approx(0.0)
    reliability_count = sum(
        item["count"] for item in reliability_diagram(LABELS, probabilities, num_bins=2)
    )
    assert reliability_count == 5


def test_gauc_skips_single_class_groups() -> None:
    labels = [1, 0, 1, 1]
    scores = [0.8, 0.2, 0.4, 0.3]
    groups = ["valid", "valid", "single", "single"]

    assert group_auc(labels, scores, groups) == pytest.approx(1.0)
