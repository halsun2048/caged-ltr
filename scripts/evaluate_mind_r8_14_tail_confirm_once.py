"""Evaluate the frozen Tail-floor policy once on R8.14 fresh-confirm-v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from select_mind_r8_6_gate import merge, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", type=Path, default=Path("runs/mind_r14"))
    parser.add_argument("--pool-root", type=Path, default=Path("data/processed/mind_r14"))
    parser.add_argument("--first-root", type=Path, default=Path("runs/mind_r14"))
    parser.add_argument("--split", default="fresh_confirm_v2")
    parser.add_argument(
        "--gate-model", type=Path, default=Path("artifacts/mind_r8_6_gate_v2.joblib")
    )
    parser.add_argument(
        "--guard", type=Path, default=Path("artifacts/mind_r8_14_confirm_guard.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/experiments/mind_r8_14_tail_confirm.json")
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    guard = json.loads(args.guard.read_text())
    if guard["evaluation_count"] == 1:
        print(json.dumps({"stage": "cached_closed", "report": str(args.output)}))
        return
    frame = merge(args.split, args.metrics_root, args.pool_root, args.first_root)
    frozen = joblib.load(args.gate_model)
    x = frame[frozen["features"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prediction = frozen["model"].predict(x)
    tail = frame.frequency_bucket.to_numpy() == "tail"
    total = round(len(frame) * 0.4)
    tail_count = round(tail.sum() * 0.75)
    route = np.zeros(len(frame), dtype=bool)
    tail_positions = np.flatnonzero(tail)
    route[tail_positions[np.argsort(-prediction[tail], kind="stable")[:tail_count]]] = True
    remaining = total - int(route.sum())
    other = np.flatnonzero(~route)
    route[other[np.argsort(-prediction[other], kind="stable")[:remaining]]] = True
    overall = {
        "student": metrics(frame, np.zeros(len(frame), dtype=bool)),
        "first": metrics(frame, np.ones(len(frame), dtype=bool)),
        "gate": metrics(frame, route),
    }
    buckets = {}
    for name, group in frame.groupby("frequency_bucket", sort=True):
        positions = group.index.to_numpy()
        buckets[str(name)] = {
            "queries": len(group),
            "route_rate": float(route[positions].mean()),
            "student": metrics(group, np.zeros(len(group), dtype=bool)),
            "first": metrics(group, np.ones(len(group), dtype=bool)),
            "gate": metrics(group, route[positions]),
        }
    acceptance = {
        "overall_gap_at_most_0p003": overall["first"]["ndcg10"] - overall["gate"]["ndcg10"]
        <= 0.003,
        "tail_gap_at_most_0p003": buckets["tail"]["first"]["ndcg10"]
        - buckets["tail"]["gate"]["ndcg10"]
        <= 0.003,
        "first_call_rate_at_most_0p55": overall["gate"]["first_call_rate"] <= 0.55,
        "gate_at_least_student": overall["gate"]["ndcg10"] >= overall["student"]["ndcg10"],
        "large_test_accessed": args.split == "large_test",
    }
    acceptance["all_passed"] = all(
        value for key, value in acceptance.items() if key != "large_test_accessed"
    )
    frozen_policy = guard.get("policy", {})
    if not isinstance(frozen_policy, dict):
        frozen_policy = {}
    payload = {
        "schema": "mind_r8_14_tail_confirm_once_v1",
        "split": args.split,
        "queries": len(frame),
        "policy": {
            "budget": 0.4,
            "tail_floor": 0.75,
            "gate_model_sha256": frozen_policy.get("gate_model_sha256", "frozen-r8.14"),
        },
        "overall": overall,
        "frequency_buckets": buckets,
        "acceptance": acceptance,
        "large_test_accessed": args.split == "large_test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    guard.update(
        {
            "status": "consumed_closed",
            "evaluation_count": 1,
            "result": str(args.output),
            "admitted": acceptance["all_passed"],
        }
    )
    args.guard.write_text(json.dumps(guard, indent=2) + "\n")
    if args.progress:
        print("[1/2] applied frozen Tail-floor route", flush=True)
        print(f"[2/2] complete admitted={acceptance['all_passed']}; fresh guard closed", flush=True)
    print(
        json.dumps(
            {"stage": "complete", "admitted": acceptance["all_passed"], "report": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
