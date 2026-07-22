from __future__ import annotations

import numpy as np
import pytest

from caged_ltr.calibration.plots import save_reliability_diagram
from caged_ltr.data import CandidateList, collate_candidate_lists
from caged_ltr.evaluation.predictions import (
    prediction_records,
    read_predictions,
    write_predictions,
)
from caged_ltr.evaluation.reporting import build_bucket_report, write_bucket_report


def test_prediction_records_rank_within_request_and_round_trip(tmp_path) -> None:
    batch = collate_candidate_lists(
        [
            CandidateList(
                request_id="q1",
                candidate_ids=["a", "b"],
                features=np.ones((2, 2)),
                labels=np.asarray([0, 1]),
            ),
            CandidateList(
                request_id="q2",
                candidate_ids=["c", "d"],
                features=np.ones((2, 2)),
                labels=np.asarray([1, 0]),
            ),
        ]
    )
    records = prediction_records(batch, [0.1, 0.9, 0.8, 0.2], query_buckets=["tail", "head"])

    assert [record.rank for record in records] == [2, 1, 1, 2]
    output = tmp_path / "predictions.parquet"
    write_predictions(records, output)
    assert read_predictions(output) == records

    rows = build_bucket_report(records, cutoffs=(1, 2), num_bins=2)
    assert [row["bucket"] for row in rows] == ["Overall", "head", "tail"]
    csv_path = tmp_path / "metrics.csv"
    json_path = tmp_path / "metrics.json"
    write_bucket_report(rows, csv_path=csv_path, json_path=json_path)
    assert csv_path.exists() and json_path.exists()
    figure_path = tmp_path / "reliability.png"
    save_reliability_diagram(
        [record.label for record in records],
        [record.probability for record in records],
        figure_path,
        num_bins=2,
    )
    assert figure_path.stat().st_size > 0


def test_prediction_records_validate_lengths() -> None:
    batch = collate_candidate_lists(
        [
            CandidateList(
                request_id="q1",
                candidate_ids=["a"],
                features=np.ones((1, 1)),
                labels=np.asarray([1]),
            )
        ]
    )

    with pytest.raises(ValueError, match="flattened"):
        prediction_records(batch, [])
