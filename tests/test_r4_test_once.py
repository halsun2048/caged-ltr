from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from caged_ltr.evaluation.r4_test_once import (
    R4_PREDICTION_COLUMNS,
    evaluate_r4_test_once,
    merge_r4_prediction_shards,
    per_query_linear_ndcg,
    validate_r4_predictions,
)


def _predictions(control: str) -> pd.DataFrame:
    rows = []
    for query_index in range(2):
        for candidate_index in range(100):
            grade = (
                3 - candidate_index
                if candidate_index < 3
                else 0
            )
            if control == "prp":
                score = float(grade * 10 - candidate_index / 1000)
            elif control == "bm25":
                score = float(2 - abs(candidate_index - 1))
            elif control == "random":
                score = float((candidate_index * 37) % 100)
            else:
                score = float(candidate_index)
            rows.append(
                {
                    "control": control,
                    "request_id": f"dl2019-q{query_index}",
                    "year": 2019,
                    "query_id": f"q{query_index}",
                    "passage_id": f"q{query_index}-d{candidate_index}",
                    "bm25_rank": candidate_index + 1,
                    "raw_score": score,
                    "probability": 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, score)))),
                    "query_elapsed_seconds": 0.2 + query_index / 10,
                }
            )
    return pd.DataFrame(rows, columns=R4_PREDICTION_COLUMNS)


def _qrels() -> pd.DataFrame:
    rows = []
    for query_index in range(2):
        for candidate_index in range(100):
            rows.append(
                {
                    "request_id": f"dl2019-q{query_index}",
                    "year": 2019,
                    "query_id": f"q{query_index}",
                    "passage_id": f"q{query_index}-d{candidate_index}",
                    "graded_relevance": (
                        3 - candidate_index if candidate_index < 3 else 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_validate_and_merge_r4_prediction_shards() -> None:
    predictions = _predictions("prp")
    validate_r4_predictions(predictions, control="prp")
    merged = merge_r4_prediction_shards(
        [
            predictions[predictions["request_id"] == "dl2019-q0"],
            predictions[predictions["request_id"] == "dl2019-q1"],
        ],
        control="prp",
        expected_request_ids={"dl2019-q0", "dl2019-q1"},
    )
    assert len(merged) == 200
    assert merged["request_id"].tolist()[:2] == ["dl2019-q0", "dl2019-q0"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda frame: frame.assign(graded_relevance=0),
            "unexpected or evaluation-only",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate",
        ),
        (
            lambda frame: frame.assign(probability=2.0),
            "out of range",
        ),
        (
            lambda frame: frame.assign(query_elapsed_seconds=np.nan),
            "finite",
        ),
    ],
)
def test_validate_r4_predictions_rejects_unsafe_frames(
    mutator,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_r4_predictions(mutator(_predictions("prp")), control="prp")


def test_merge_rejects_query_coverage_mismatch() -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        merge_r4_prediction_shards(
            [_predictions("prp")],
            control="prp",
            expected_request_ids={"dl2019-q0", "dl2019-q1", "missing"},
        )


def test_linear_ndcg_uses_complete_qrels_ideal_and_stable_ties() -> None:
    predictions = _predictions("prp")
    values = per_query_linear_ndcg(
        predictions,
        _qrels(),
        score_column="raw_score",
    )
    assert values == pytest.approx({"dl2019-q0": 1.0, "dl2019-q1": 1.0})
    tied = predictions.assign(raw_score=0.0)
    tied_values = per_query_linear_ndcg(
        tied,
        _qrels(),
        score_column="raw_score",
    )
    assert tied_values == pytest.approx(values)


def test_evaluate_r4_test_once_reports_controls_calibration_and_bootstrap() -> None:
    predictions = {
        control: _predictions(control)
        for control in ("vanilla", "bm25", "random", "prp")
    }
    result = evaluate_r4_test_once(
        predictions,
        _qrels(),
        bootstrap_iterations=100,
        seed=42,
    )
    controls = result["controls"]
    assert controls["prp"]["overall"]["trec_eval_ndcg_at_10"] == pytest.approx(1.0)
    assert controls["prp"]["by_year"]["2019"]["queries"] == 2
    assert controls["prp"]["efficiency"]["mean_seconds_per_query"] == pytest.approx(
        0.25
    )
    assert 0.0 <= controls["prp"]["calibration_binary_grade_at_least_2"]["ECE"] <= 1.0
    assert result["paired_bootstrap"]["prp_minus_vanilla"]["iterations"] == 100
    assert controls["bm25_initial"]["overall"]["trec_eval_ndcg_at_10"] == pytest.approx(
        1.0
    )


def test_evaluate_rejects_missing_control_and_mismatched_queries() -> None:
    predictions = {
        control: _predictions(control)
        for control in ("vanilla", "bm25", "random", "prp")
    }
    with pytest.raises(ValueError, match="all four"):
        evaluate_r4_test_once(
            {key: value for key, value in predictions.items() if key != "random"},
            _qrels(),
            bootstrap_iterations=10,
        )
    predictions["random"] = predictions["random"][
        predictions["random"]["request_id"] != "dl2019-q1"
    ]
    with pytest.raises(ValueError, match="candidate count|same test queries"):
        evaluate_r4_test_once(
            predictions,
            _qrels(),
            bootstrap_iterations=10,
        )
