"""Develop a Tail-floor route on consumed fresh-confirm data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from select_mind_r8_6_gate import merge, metrics


def route(frame, prediction: np.ndarray, budget: float, tail_floor: float) -> np.ndarray:
    total = round(len(frame) * budget)
    tail = frame.frequency_bucket.to_numpy() == "tail"
    tail_count = min(round(tail.sum() * tail_floor), total)
    route = np.zeros(len(frame), dtype=bool)
    tail_gain_order = np.argsort(-prediction[tail], kind="stable")
    tail_positions = np.flatnonzero(tail)
    route[tail_positions[tail_gain_order[:tail_count]]] = True
    remaining = total - int(route.sum())
    other_positions = np.flatnonzero(~route)
    order = other_positions[np.argsort(-prediction[other_positions], kind="stable")]
    route[order[:remaining]] = True
    return route


def buckets(frame, route):
    result = {}
    for name, group in frame.groupby("frequency_bucket", sort=True):
        positions = group.index.to_numpy()
        result[str(name)] = {
            "queries": len(group),
            "route_rate": float(route[positions].mean()),
            "student": metrics(group, np.zeros(len(group), dtype=bool)),
            "first": metrics(group, np.ones(len(group), dtype=bool)),
            "gate": metrics(group, route[positions]),
        }
    return result


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
        default=Path("reports/experiments/mind_r8_11_tail_floor.json"),
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    frame = merge("fresh_confirm", args.metrics_root, args.pool_root, args.first_root)
    frozen = joblib.load(args.gate_model)
    x = frame[frozen["features"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prediction = frozen["model"].predict(x)
    candidates = []
    for budget in (0.35, 0.40, 0.45, 0.50, 0.55):
        for floor in (0.75, 0.90, 1.00):
            selected = route(frame, prediction, budget, floor)
            overall = metrics(frame, selected)
            by_bucket = buckets(frame, selected)
            candidates.append(
                {
                    "budget": budget,
                    "tail_floor": floor,
                    "overall": overall,
                    "by_bucket": by_bucket,
                    "overall_gap_vs_first": float(
                        by_bucket["head"]["first"]["ndcg10"] * 0
                        + frame.first_ndcg10.mean() - overall["ndcg10"]
                    ),
                    "tail_gap_vs_first": float(
                        by_bucket["tail"]["first"]["ndcg10"]
                        - by_bucket["tail"]["gate"]["ndcg10"]
                    ),
                }
            )
    eligible = [
        value
        for value in candidates
        if value["overall_gap_vs_first"] <= 0.003
        and value["tail_gap_vs_first"] <= 0.003
        and value["overall"]["first_call_rate"] <= 0.55
    ]
    selected = (
        min(eligible, key=lambda value: (value["budget"], -value["tail_floor"]))
        if eligible
        else None
    )
    payload = {
        "schema": "mind_r8_11_tail_floor_development_v1",
        "source_split": "fresh_confirm_consumed",
        "candidates": candidates,
        "selected": selected,
        "acceptance": {
            "eligible_on_consumed_data": selected is not None,
            "large_test_accessed": False,
        },
        "next": "preregister a new confirm before any formal claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.progress:
        print(f"[1/2] evaluated {len(candidates)} Tail-floor policies", flush=True)
        print(f"[2/2] selected={selected is not None}; no new split accessed", flush=True)
    print(
        json.dumps(
            {"stage": "complete", "selected": selected is not None, "report": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
