"""Apply the frozen R8.6 gate once and permanently consume the R8 large-test guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from select_mind_r8_6_gate import first_frame, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--guard", type=Path, default=Path("artifacts/mind_r8_0_large_test_guard.json")
    )
    parser.add_argument(
        "--gate-report", type=Path, default=Path("reports/experiments/mind_r8_6_gate.json")
    )
    parser.add_argument("--gate-model", type=Path, default=Path("artifacts/mind_r8_6_gate.joblib"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/experiments/mind_r8_9_final_once.json")
    )
    args = parser.parse_args()
    guard = json.loads(args.guard.read_text())
    if guard["status"] != "locked_unaccessed" or guard["evaluation_count"] != 0:
        raise RuntimeError("R8.9 guard has already been consumed or is invalid")
    gate_report = json.loads(args.gate_report.read_text())
    if not gate_report["acceptance"]["gate_confirm_passed"]:
        raise RuntimeError("R8.8 did not admit the one-shot test")
    frozen = joblib.load(args.gate_model)
    if not frozen["selected"]:
        raise RuntimeError("frozen gate policy is absent")
    first = first_frame(
        Path("data/processed/mind_r8_9/large_test.parquet"),
        Path("runs/mind_r8_9/large_test_prompts.jsonl"),
        Path("runs/mind_r8_9/large_test_first/results.jsonl"),
    )
    student = pd.read_parquet("runs/mind_r8_9/large_test_query_metrics.parquet")
    frame = student.merge(first, on="query_id", validate="one_to_one")
    probability = frozen["model"].predict_proba(frame[frozen["features"]])[:, 1]
    route = probability >= frozen["selected"]["threshold"]
    gate = metrics(frame, route)
    delta = np.where(route, frame.first_ndcg10, frame.ndcg10) - frame.ndcg10
    payload = {
        "schema": "mind_r8_9_final_once_v1",
        "queries": len(frame),
        "student": {
            "ndcg10": float(frame.ndcg10.mean()),
            "hit10": float(frame.hit10.mean()),
            "mrr": float(frame.mrr.mean()),
        },
        "first": {
            "ndcg10": float(frame.first_ndcg10.mean()),
            "hit10": float(frame.first_hit10.mean()),
            "mrr": float(frame.first_mrr.mean()),
        },
        "gate": gate,
        "gate_delta_ndcg10_vs_student": float(delta.mean()),
        "evaluation_count": 1,
        "policy_frozen_before_test": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    guard.update(
        {
            "status": "consumed_closed",
            "evaluation_count": 1,
            "labels_materialized": True,
            "predictions_exist": True,
        }
    )
    temporary = args.guard.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(guard, indent=2) + "\n")
    temporary.replace(args.guard)
    print(json.dumps({"stage": "complete", "report": str(args.output), "gate": gate}))


if __name__ == "__main__":
    main()
