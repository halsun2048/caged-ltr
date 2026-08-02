"""Apply the frozen gain gate exactly once on the fresh R8.8 confirmation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from select_mind_r8_6_gate import merge, metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(values: np.ndarray, seed: int = 20260803) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(2_000)
    for offset in range(0, len(means), 100):
        size = min(100, len(means) - offset)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[offset : offset + size] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", type=Path, default=Path("runs/mind_r8_8b"))
    parser.add_argument("--pool-root", type=Path, default=Path("data/processed/mind_r8_8b"))
    parser.add_argument("--first-root", type=Path, default=Path("runs/mind_r8_8b"))
    parser.add_argument("--split", default="fresh_confirm")
    parser.add_argument(
        "--gate-model", type=Path, default=Path("artifacts/mind_r8_6_gate_v2.joblib")
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("reports/data/mind_r8_8a_preregistration.json"),
    )
    parser.add_argument(
        "--guard", type=Path, default=Path("artifacts/mind_r8_8a_confirm_guard.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/experiments/mind_r8_8b_fresh_confirm.json")
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    guard = json.loads(args.guard.read_text())
    if guard["evaluation_count"] == 1 and args.output.exists():
        print(json.dumps({"stage": "cached_closed", "report": str(args.output)}))
        return
    if guard["status"] not in {
        "preregistered_unaccessed",
        "materialized_pending_evaluation",
        "locked_unaccessed",
    }:
        raise RuntimeError(f"evaluation guard is not evaluable: {guard['status']}")
    prereg = json.loads(args.preregistration.read_text())
    if sha256(args.gate_model) != prereg["hashes"]["gate_model"]:
        raise RuntimeError("frozen gate model hash changed after preregistration")
    frozen = joblib.load(args.gate_model)
    policy = prereg["frozen_policy"]
    if frozen["features"] != policy["features"]:
        raise RuntimeError("frozen feature order changed after preregistration")
    if frozen["selected"]["threshold"] != policy["threshold"]:
        raise RuntimeError("frozen threshold changed after preregistration")
    if args.progress:
        print("[1/5] verified preregistered model, features, and absolute threshold", flush=True)
    frame = merge(args.split, args.metrics_root, args.pool_root, args.first_root)
    x = frame[policy["features"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prediction = frozen["model"].predict(x)
    route = prediction >= policy["threshold"]
    if args.progress:
        print(f"[2/5] applied frozen route to {len(frame):,} queries", flush=True)
    student_values = frame.ndcg10.to_numpy()
    first_values = frame.first_ndcg10.to_numpy()
    gate_values = np.where(route, first_values, student_values)
    overall = {
        "student": metrics(frame, np.zeros(len(frame), dtype=bool)),
        "first": metrics(frame, np.ones(len(frame), dtype=bool)),
        "gate": metrics(frame, route),
    }
    buckets = {}
    for bucket, group in frame.groupby("frequency_bucket", sort=True):
        positions = group.index.to_numpy()
        local_route = route[positions]
        buckets[str(bucket)] = {
            "queries": len(group),
            "student": metrics(group, np.zeros(len(group), dtype=bool)),
            "first": metrics(group, np.ones(len(group), dtype=bool)),
            "gate": metrics(group, local_route),
        }
    if args.progress:
        print("[3/5] computed overall and Head/Torso/Tail metrics", flush=True)
    tail = buckets["tail"]
    acceptance = {
        "overall_gap_vs_first_at_most_0p003": (
            overall["first"]["ndcg10"] - overall["gate"]["ndcg10"] <= 0.003
        ),
        "first_call_rate_at_most_0p55": overall["gate"]["first_call_rate"] <= 0.55,
        "gate_at_least_student": overall["gate"]["ndcg10"] >= overall["student"]["ndcg10"],
        "tail_gap_vs_first_at_most_0p003": (
            tail["first"]["ndcg10"] - tail["gate"]["ndcg10"] <= 0.003
        ),
        "hash_and_boundary_checks": True,
        "large_test_accessed": args.split == "large_test",
    }
    acceptance["all_passed"] = all(
        value for key, value in acceptance.items() if key != "large_test_accessed"
    )
    if args.progress:
        print("[4/5] running 2,000-sample paired bootstrap", flush=True)
    payload = {
        "schema": "mind_r8_guarded_evaluation_once_v1",
        "split": args.split,
        "queries": len(frame),
        "frozen_policy": policy,
        "overall": overall,
        "frequency_buckets": buckets,
        "bootstrap_95pct": {
            "gate_minus_first_ndcg10": bootstrap(gate_values - first_values),
            "gate_minus_student_ndcg10": bootstrap(gate_values - student_values),
        },
        "acceptance": acceptance,
        "boundaries": {
            "evaluation_count": 1,
            "large_test_accessed": args.split == "large_test",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    guard.update(
        {
            "status": "consumed_closed",
            "evaluation_count": 1,
            "labels_materialized": True,
            "result": str(args.output),
            "admitted": acceptance["all_passed"],
        }
    )
    args.guard.write_text(json.dumps(guard, indent=2) + "\n")
    if args.progress:
        print(f"[5/5] result committed and {args.split} guard permanently closed", flush=True)
    print(
        json.dumps(
            {"stage": "complete", "admitted": acceptance["all_passed"], "report": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
