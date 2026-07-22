"""Canonical per-candidate prediction records and persistence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from caged_ltr.data import CandidateBatch


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    request_id: str
    candidate_id: str
    label: float
    score: float
    probability: float
    rank: int
    query_bucket: str | None = None
    candidate_bucket: str | None = None
    user_bucket: str | None = None


def _as_numpy(values: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def prediction_records(
    batch: CandidateBatch,
    scores: torch.Tensor | np.ndarray | Sequence[float],
    probabilities: torch.Tensor | np.ndarray | Sequence[float] | None = None,
    *,
    query_buckets: Sequence[str] | None = None,
    candidate_buckets: Sequence[str] | None = None,
    user_buckets: Sequence[str] | None = None,
) -> list[PredictionRecord]:
    """Convert one model batch to stable, ranked, per-candidate records."""
    score_array = _as_numpy(scores).astype(np.float64, copy=False)
    probability_array = (
        1.0 / (1.0 + np.exp(-score_array))
        if probabilities is None
        else _as_numpy(probabilities).astype(np.float64, copy=False)
    )
    candidate_count = len(batch.candidate_ids)
    group_count = len(batch.request_ids)
    if score_array.shape != (candidate_count,) or probability_array.shape != (candidate_count,):
        raise ValueError("scores and probabilities must match the flattened candidates")
    if query_buckets is not None and len(query_buckets) != group_count:
        raise ValueError("query_buckets must have one entry per request")
    if user_buckets is not None and len(user_buckets) != group_count:
        raise ValueError("user_buckets must have one entry per request")
    if candidate_buckets is not None and len(candidate_buckets) != candidate_count:
        raise ValueError("candidate_buckets must have one entry per candidate")

    labels = batch.labels.detach().cpu().numpy()
    records: list[PredictionRecord] = []
    for group_index, group_slice in enumerate(batch.group_slices()):
        order = np.argsort(-score_array[group_slice], kind="stable")
        ranks = np.empty(order.size, dtype=np.int64)
        ranks[order] = np.arange(1, order.size + 1)
        for local_index, flat_index in enumerate(range(group_slice.start, group_slice.stop)):
            records.append(
                PredictionRecord(
                    request_id=batch.request_ids[group_index],
                    candidate_id=batch.candidate_ids[flat_index],
                    label=float(labels[flat_index]),
                    score=float(score_array[flat_index]),
                    probability=float(probability_array[flat_index]),
                    rank=int(ranks[local_index]),
                    query_bucket=(
                        query_buckets[group_index] if query_buckets is not None else None
                    ),
                    candidate_bucket=(
                        candidate_buckets[flat_index]
                        if candidate_buckets is not None
                        else None
                    ),
                    user_bucket=user_buckets[group_index] if user_buckets is not None else None,
                )
            )
    return records


def write_predictions(records: Sequence[PredictionRecord], output: Path) -> None:
    """Persist raw predictions in Parquet for later table generation."""
    if not records:
        raise ValueError("records must not be empty")
    if output.suffix != ".parquet":
        raise ValueError("prediction output must use the .parquet extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([asdict(record) for record in records])
    pq.write_table(table, output, compression="zstd")


def read_predictions(path: Path) -> list[PredictionRecord]:
    """Load prediction records written by :func:`write_predictions`."""
    return [PredictionRecord(**row) for row in pq.read_table(path).to_pylist()]
