"""Leakage-safe prediction validation and test-once metrics for R4."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from caged_ltr.evaluation.metrics import classification_metrics
from caged_ltr.evaluation.paired_bootstrap import paired_bootstrap_mean

R4_CONTROLS = ("vanilla", "bm25", "random", "prp")
R4_PREDICTION_COLUMNS = (
    "control",
    "request_id",
    "year",
    "query_id",
    "passage_id",
    "bm25_rank",
    "raw_score",
    "probability",
    "query_elapsed_seconds",
)


def validate_r4_predictions(
    predictions: pd.DataFrame,
    *,
    control: str,
    expected_candidates_per_query: int = 100,
) -> None:
    """Reject incomplete, duplicated, non-finite, or label-bearing predictions."""
    if control not in R4_CONTROLS:
        raise ValueError(f"unknown R4 control: {control}")
    if tuple(predictions.columns) != R4_PREDICTION_COLUMNS:
        raise ValueError("R4 predictions have an unexpected or evaluation-only column")
    if predictions.empty or set(predictions["control"]) != {control}:
        raise ValueError("R4 predictions must contain exactly the requested control")
    if predictions.duplicated(["request_id", "passage_id"]).any():
        raise ValueError("R4 predictions contain duplicate request/passage pairs")
    numeric = predictions[
        ["year", "bm25_rank", "raw_score", "probability", "query_elapsed_seconds"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("R4 prediction numeric fields must be finite")
    if (
        (predictions["probability"] < 0.0).any()
        or (predictions["probability"] > 1.0).any()
        or (predictions["query_elapsed_seconds"] <= 0.0).any()
    ):
        raise ValueError("R4 probabilities or query latencies are out of range")
    sizes = predictions.groupby("request_id", sort=False).size()
    if not (sizes == expected_candidates_per_query).all():
        raise ValueError("every R4 query must have the frozen candidate count")
    for _, group in predictions.groupby("request_id", sort=False):
        if group["query_id"].nunique() != 1 or group["year"].nunique() != 1:
            raise ValueError("query identity changed within an R4 candidate list")
        if sorted(group["bm25_rank"].astype(int)) != list(
            range(1, expected_candidates_per_query + 1)
        ):
            raise ValueError("R4 BM25 ranks must be contiguous within each query")
        if group["query_elapsed_seconds"].nunique() != 1:
            raise ValueError("R4 query latency must be identical across its candidates")


def merge_r4_prediction_shards(
    shards: Sequence[pd.DataFrame],
    *,
    control: str,
    expected_request_ids: set[str],
    expected_candidates_per_query: int = 100,
) -> pd.DataFrame:
    """Merge disjoint qrels-free shards and prove exact query coverage."""
    if not shards:
        raise ValueError("at least one R4 prediction shard is required")
    merged = pd.concat(shards, ignore_index=True)
    validate_r4_predictions(
        merged,
        control=control,
        expected_candidates_per_query=expected_candidates_per_query,
    )
    observed = set(merged["request_id"].astype(str))
    if observed != expected_request_ids:
        missing = sorted(expected_request_ids - observed)
        extra = sorted(observed - expected_request_ids)
        raise ValueError(f"R4 prediction query coverage mismatch: {missing=}, {extra=}")
    return merged.sort_values(
        ["request_id", "bm25_rank"],
        kind="stable",
    ).reset_index(drop=True)


def _validated_qrels(qrels: pd.DataFrame) -> pd.DataFrame:
    required = {
        "request_id",
        "year",
        "query_id",
        "passage_id",
        "graded_relevance",
    }
    if not required.issubset(qrels.columns) or qrels.empty:
        raise ValueError("official qrels are missing required R4 columns")
    if qrels.duplicated(["request_id", "passage_id"]).any():
        raise ValueError("official qrels contain duplicate request/passage pairs")
    if (qrels["graded_relevance"] < 0).any():
        raise ValueError("official qrels contain negative grades")
    return qrels


def per_query_linear_ndcg(
    predictions: pd.DataFrame,
    qrels: pd.DataFrame,
    *,
    score_column: str,
    cutoff: int = 10,
) -> dict[str, float]:
    """Return trec_eval-style linear-gain NDCG for every frozen query."""
    if cutoff <= 0 or score_column not in predictions.columns:
        raise ValueError("invalid R4 score column or cutoff")
    qrels = _validated_qrels(qrels)
    qrels_by_request = {
        str(request_id): {
            str(row.passage_id): int(row.graded_relevance)
            for row in group.itertuples()
        }
        for request_id, group in qrels.groupby("request_id", sort=False)
    }
    values: dict[str, float] = {}
    for request_id, group in predictions.groupby("request_id", sort=False):
        request_key = str(request_id)
        if request_key not in qrels_by_request:
            raise ValueError(f"R4 predictions contain a query absent from qrels: {request_key}")
        relevance = qrels_by_request[request_key]
        ranked = group.sort_values(
            [score_column, "bm25_rank"],
            ascending=[False, True],
            kind="stable",
        )
        observed = [
            relevance.get(str(passage_id), 0)
            for passage_id in ranked["passage_id"].iloc[:cutoff]
        ]
        ideal = sorted(relevance.values(), reverse=True)[:cutoff]
        dcg = sum(
            grade / math.log2(rank + 1)
            for rank, grade in enumerate(observed, start=1)
        )
        idcg = sum(
            grade / math.log2(rank + 1)
            for rank, grade in enumerate(ideal, start=1)
        )
        values[request_key] = dcg / idcg if idcg else 0.0
    return values


def _latency_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    per_query = (
        predictions[["request_id", "query_elapsed_seconds"]]
        .drop_duplicates("request_id")
        .sort_values("request_id")
    )
    seconds = per_query["query_elapsed_seconds"].to_numpy(dtype=np.float64)
    total_candidates = len(predictions)
    total_seconds = float(seconds.sum())
    return {
        "queries": len(per_query),
        "candidates": total_candidates,
        "mean_seconds_per_query": float(seconds.mean()),
        "p50_seconds_per_query": float(np.percentile(seconds, 50)),
        "p95_seconds_per_query": float(np.percentile(seconds, 95)),
        "total_single_worker_seconds": total_seconds,
        "candidates_per_second": total_candidates / total_seconds,
    }


def _calibration_metrics(
    predictions: pd.DataFrame,
    qrels: pd.DataFrame,
) -> dict[str, float]:
    relevance = {
        (str(row.request_id), str(row.passage_id)): int(row.graded_relevance)
        for row in qrels.itertuples()
    }
    labels = np.asarray(
        [
            relevance.get((str(row.request_id), str(row.passage_id)), 0) >= 2
            for row in predictions.itertuples()
        ],
        dtype=np.float64,
    )
    return classification_metrics(
        labels,
        predictions["raw_score"].to_numpy(dtype=np.float64),
        predictions["probability"].to_numpy(dtype=np.float64),
        num_bins=15,
    )


def evaluate_r4_test_once(
    predictions: Mapping[str, pd.DataFrame],
    qrels: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate frozen four-control predictions after explicit qrels access."""
    if set(predictions) != set(R4_CONTROLS):
        raise ValueError("R4 test-once evaluation requires all four frozen controls")
    qrels = _validated_qrels(qrels)
    request_sets: dict[str, set[str]] = {}
    per_query: dict[str, dict[str, float]] = {}
    results: dict[str, object] = {}
    for control in R4_CONTROLS:
        frame = predictions[control]
        validate_r4_predictions(frame, control=control)
        request_sets[control] = set(frame["request_id"].astype(str))
        control_per_query = per_query_linear_ndcg(
            frame,
            qrels,
            score_column="raw_score",
        )
        per_query[control] = control_per_query
        by_year: dict[str, object] = {}
        for year, year_frame in frame.groupby("year", sort=True):
            year_values = per_query_linear_ndcg(
                year_frame,
                qrels[qrels["year"] == year],
                score_column="raw_score",
            )
            by_year[str(int(year))] = {
                "queries": len(year_values),
                "trec_eval_ndcg_at_10": float(np.mean(list(year_values.values()))),
            }
        results[control] = {
            "overall": {
                "queries": len(control_per_query),
                "trec_eval_ndcg_at_10": float(
                    np.mean(list(control_per_query.values()))
                ),
            },
            "by_year": by_year,
            "calibration_binary_grade_at_least_2": _calibration_metrics(
                frame,
                qrels,
            ),
            "efficiency": _latency_metrics(frame),
        }
    if len({frozenset(values) for values in request_sets.values()}) != 1:
        raise ValueError("R4 controls do not cover the same test queries")

    reference = predictions["prp"]
    bm25_per_query = per_query_linear_ndcg(
        reference.assign(bm25_score=-reference["bm25_rank"].astype(float)),
        qrels,
        score_column="bm25_score",
    )
    bm25_by_year: dict[str, object] = {}
    for year, year_frame in reference.groupby("year", sort=True):
        year_values = per_query_linear_ndcg(
            year_frame.assign(bm25_score=-year_frame["bm25_rank"].astype(float)),
            qrels[qrels["year"] == year],
            score_column="bm25_score",
        )
        bm25_by_year[str(int(year))] = {
            "queries": len(year_values),
            "trec_eval_ndcg_at_10": float(np.mean(list(year_values.values()))),
        }
    results["bm25_initial"] = {
        "overall": {
            "queries": len(bm25_per_query),
            "trec_eval_ndcg_at_10": float(np.mean(list(bm25_per_query.values()))),
        },
        "by_year": bm25_by_year,
    }

    ordered_requests = sorted(per_query["prp"])
    significance: dict[str, object] = {}
    for control in ("vanilla", "bm25", "random"):
        differences = np.asarray(
            [
                per_query["prp"][request_id] - per_query[control][request_id]
                for request_id in ordered_requests
            ]
        )
        significance[f"prp_minus_{control}"] = paired_bootstrap_mean(
            differences,
            iterations=bootstrap_iterations,
            seed=seed,
        )
    return {
        "metric": "linear graded NDCG@10 over complete official qrels",
        "calibration_rule": (
            "unjudged candidates are non-relevant; only NIST grades 2 and 3 "
            "are binary positives; probabilities are raw sigmoid scores "
            "without test-set calibration"
        ),
        "controls": results,
        "paired_bootstrap": significance,
    }
