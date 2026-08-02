"""Offline Tail failure attribution for the already-consumed fresh-confirm split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from select_mind_r8_6_gate import merge, metrics

FEATURES = [
    "margin", "top1_score", "top3_gap", "top5_gap", "score_mean", "score_std",
    "score_entropy", "student_top1_source_rank", "student_top3_mean_source_rank",
    "student_source_top1_agreement", "student_source_top3_overlap",
    "top1_lexical_overlap", "max_lexical_overlap", "mean_lexical_overlap",
    "query_characters", "candidate_count", "top1_passage_characters",
    "mean_passage_characters",
]


def fixed_budget(prediction: np.ndarray, budget: float) -> np.ndarray:
    count = round(len(prediction) * budget)
    route = np.zeros(len(prediction), dtype=bool)
    route[np.argsort(-prediction, kind="stable")[:count]] = True
    return route


def tail_priority(frame: pd.DataFrame, budget: float) -> np.ndarray:
    """Oracle diagnostic: spend calls on Tail gain first, then other gain."""
    gain = (frame.first_ndcg10 - frame.ndcg10).to_numpy()
    priority = np.where(frame.frequency_bucket.to_numpy() == "tail", 1, 0)
    order = np.lexsort((-gain, -priority))
    route = np.zeros(len(frame), dtype=bool)
    route[order[: round(len(frame) * budget)]] = True
    return route


def group_summary(frame: pd.DataFrame, route: np.ndarray) -> dict[str, object]:
    output: dict[str, object] = {}
    for bucket, group in frame.groupby("frequency_bucket", sort=True):
        positions = group.index.to_numpy()
        output[str(bucket)] = {
            "queries": len(group),
            "route_rate": float(route[positions].mean()),
            "student": metrics(group, np.zeros(len(group), dtype=bool)),
            "first": metrics(group, np.ones(len(group), dtype=bool)),
            "gate": metrics(group, route[positions]),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", type=Path, default=Path("runs/mind_r8_8b"))
    parser.add_argument("--pool-root", type=Path, default=Path("data/processed/mind_r8_8b"))
    parser.add_argument("--first-root", type=Path, default=Path("runs/mind_r8_8b"))
    parser.add_argument(
        "--gate-model", type=Path, default=Path("artifacts/mind_r8_6_gate_v2.joblib")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiments/mind_r8_10_tail_attribution.json"),
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    frame = merge("fresh_confirm", args.metrics_root, args.pool_root, args.first_root)
    frozen = joblib.load(args.gate_model)
    x = frame[frozen["features"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prediction = frozen["model"].predict(x)
    route = prediction >= frozen["selected"]["threshold"]
    gain = frame.first_ndcg10.to_numpy() - frame.ndcg10.to_numpy()
    tail = frame.frequency_bucket.to_numpy() == "tail"
    missed_tail = tail & ~route
    if args.progress:
        print(f"[1/4] loaded {len(frame):,} fresh-confirm queries", flush=True)
    budgets = {}
    for budget in (0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 1.00):
        global_route = fixed_budget(prediction, budget)
        tail_route = tail_priority(frame, budget)
        budgets[str(budget)] = {
            "global_gain_oracle": metrics(frame, fixed_budget(gain, budget)),
            "global_predicted_route": metrics(frame, global_route),
            "tail_priority_oracle": metrics(frame, tail_route),
            "tail_priority_by_bucket": group_summary(frame, tail_route),
        }
    if args.progress:
        print("[2/4] computed global and Tail-priority budget curves", flush=True)
    correlations = {}
    for feature in frozen["features"]:
        correlations[feature] = float(frame[feature].corr(pd.Series(gain)))
    tail_gain = gain[tail]
    payload = {
        "schema": "mind_r8_10_tail_attribution_v1",
        "source_split": "fresh_confirm",
        "queries": len(frame),
        "current_route": {
            "overall": metrics(frame, route),
            "by_bucket": group_summary(frame, route),
            "tail_route_rate": float(route[tail].mean()),
            "tail_missed_queries": int(missed_tail.sum()),
            "tail_missed_positive_gain_rate": float((tail_gain[~route[tail]] > 0).mean()),
            "tail_missed_mean_gain": float(gain[missed_tail].mean()),
        },
        "gain_distribution": {
            "overall_mean": float(gain.mean()),
            "tail_mean": float(tail_gain.mean()),
            "tail_quantiles": {
                str(q): float(np.quantile(tail_gain, q))
                for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
            "tail_positive_gain_rate": float((tail_gain > 0).mean()),
        },
        "feature_gain_correlation": correlations,
        "budget_curves": budgets,
        "recommendation": {
            "failure_mode": "global gain gate under-routes high-gain Tail queries",
            "next_strategy": "Tail-aware gain target plus bucket-specific threshold or Tail floor",
            "do_not_access_large_test": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.progress:
        print("[3/4] quantified Tail misses and feature correlations", flush=True)
        print("[4/4] analysis report written; no new evaluation performed", flush=True)
    print(json.dumps({"stage": "complete", "report": str(args.output)}))


if __name__ == "__main__":
    main()
