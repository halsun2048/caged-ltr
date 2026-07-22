from __future__ import annotations

import numpy as np
import pytest

from caged_ltr.evaluation.buckets import FrequencyBucketer


def test_frequency_buckets_use_frozen_training_thresholds() -> None:
    bucketer = FrequencyBucketer().fit([1, 2, 3, 4, 5])

    assert bucketer.transform([1, 2, 3, 4, 100]).tolist() == [
        "tail",
        "tail",
        "torso",
        "head",
        "head",
    ]
    assert bucketer.boundaries is not None
    assert bucketer.boundaries.tail_max == 2
    assert bucketer.boundaries.head_min == 4


def test_tied_quantiles_fall_back_to_torso() -> None:
    assert FrequencyBucketer().fit_transform(np.ones(5, dtype=int)).tolist() == ["torso"] * 5


def test_frequency_buckets_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        FrequencyBucketer().fit([1, -1])
