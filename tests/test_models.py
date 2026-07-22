from __future__ import annotations

import numpy as np
import torch

from caged_ltr.models import DCNv2Student, LambdaMARTRanker, MLPStudent


def test_neural_students_return_one_logit_per_candidate() -> None:
    features = torch.randn(5, 4)

    assert MLPStudent(4, hidden_dims=(8, 4))(features).shape == (5,)
    assert DCNv2Student(4, num_cross_layers=2, deep_dims=(8, 4))(features).shape == (5,)


def test_lambda_mart_fits_explicit_query_groups() -> None:
    features = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 1.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [4.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.float32)
    model = LambdaMARTRanker(n_estimators=5, min_child_samples=1).fit(
        features, labels, [2, 2, 2]
    )

    predictions = model.predict(features)

    assert predictions.shape == (6,)
    assert np.isfinite(predictions).all()
