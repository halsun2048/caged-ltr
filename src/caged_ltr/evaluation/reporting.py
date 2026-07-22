"""Generate Overall and frequency-bucket tables from raw predictions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from caged_ltr.evaluation.metrics import evaluate_predictions
from caged_ltr.evaluation.predictions import PredictionRecord


def build_bucket_report(
    records: Sequence[PredictionRecord],
    *,
    bucket_field: str = "query_bucket",
    cutoffs: Sequence[int] = (5, 10, 20),
    num_bins: int = 15,
) -> list[dict[str, float | int | str]]:
    """Build Overall plus Head/Torso/Tail rows from request-level buckets."""
    if not records:
        raise ValueError("records must not be empty")
    if bucket_field not in {"query_bucket", "user_bucket"}:
        raise ValueError("bucket_field must be query_bucket or user_bucket")

    rows: list[dict[str, float | int | str]] = []
    for bucket in ("Overall", "head", "torso", "tail"):
        selected = (
            list(records)
            if bucket == "Overall"
            else [record for record in records if getattr(record, bucket_field) == bucket]
        )
        if not selected:
            continue
        metrics = evaluate_predictions(
            [record.label for record in selected],
            [record.score for record in selected],
            [record.probability for record in selected],
            [record.request_id for record in selected],
            cutoffs=cutoffs,
            num_bins=num_bins,
        )
        rows.append(
            {
                "bucket": bucket,
                "requests": len({record.request_id for record in selected}),
                "candidates": len(selected),
                **metrics,
            }
        )
    return rows


def write_bucket_report(
    rows: Sequence[dict[str, float | int | str]],
    *,
    csv_path: Path,
    json_path: Path,
) -> None:
    """Write the same generated metric table in machine- and reader-friendly forms."""
    if not rows:
        raise ValueError("rows must not be empty")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(list(rows), ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
