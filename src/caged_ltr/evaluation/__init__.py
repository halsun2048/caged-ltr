"""Ranking, calibration, bucket, and efficiency evaluation."""

from caged_ltr.evaluation.metrics import (
    classification_metrics,
    evaluate_predictions,
    group_auc,
    ranking_metrics,
)
from caged_ltr.evaluation.paired_bootstrap import paired_bootstrap_mean
from caged_ltr.evaluation.r4_test_once import (
    evaluate_r4_test_once,
    merge_r4_prediction_shards,
    per_query_linear_ndcg,
    validate_r4_predictions,
)

__all__ = [
    "classification_metrics",
    "evaluate_predictions",
    "evaluate_r4_test_once",
    "group_auc",
    "merge_r4_prediction_shards",
    "paired_bootstrap_mean",
    "per_query_linear_ndcg",
    "ranking_metrics",
    "validate_r4_predictions",
]
