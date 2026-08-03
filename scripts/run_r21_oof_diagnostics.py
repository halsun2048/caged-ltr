"""R21 diagnostics for R20 OOF candidates without opening confirm/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_mind_r11_gate_search import merge
from sklearn.metrics import mean_absolute_error

from caged_ltr.r18_gate_features import FEATURES, frame_matrix


def bootstrap(delta: np.ndarray, seed: int = 20260821) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(2000)
    for index in range(len(means)):
        means[index] = rng.choice(delta, len(delta), replace=True).mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def comparison(frame: pd.DataFrame, route: np.ndarray) -> dict[str, object]:
    student = frame.ndcg10.to_numpy(float)
    first = frame.first_ndcg10.to_numpy(float)
    gate = np.where(route, first, student)
    delta_student, delta_first = gate - student, gate - first
    return {
        "gate_ndcg10": float(gate.mean()),
        "student_ndcg10": float(student.mean()),
        "first_ndcg10": float(first.mean()),
        "gate_minus_student": float(delta_student.mean()),
        "gate_minus_student_95ci": bootstrap(delta_student),
        "gate_minus_first": float(delta_first.mean()),
        "gate_minus_first_95ci": bootstrap(delta_first, 20260822),
        "win_tie_loss_vs_first": {
            "win": float((delta_first > 0).mean()),
            "tie": float((delta_first == 0).mean()),
            "loss": float((delta_first < 0).mean()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", type=Path, default=Path("runs/mind_r20_dev_routes.parquet"))
    ap.add_argument(
        "--metrics", type=Path, default=Path("runs/mind_r12_0/r12_dev_query_metrics.parquet")
    )
    ap.add_argument("--pool-root", type=Path, default=Path("data/processed/mind_r12_0"))
    ap.add_argument("--first-root", type=Path, default=Path("runs/mind_r12_0"))
    ap.add_argument(
        "--output", type=Path, default=Path("reports/experiments/mind_r21_oof_diagnostics.json")
    )
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    frame = merge("dev", args.metrics, args.pool_root, args.first_root).reset_index(drop=True)
    routes = pd.read_parquet(args.routes)
    if not np.array_equal(frame.query_id.astype(str), routes.query_id.astype(str)):
        raise SystemExit("R20 routes are not aligned with dev metrics")
    policies = {}
    for policy in ("logistic", "gain_extra_trees", "tail_floor_0.90"):
        route = routes[f"route_{policy}"].to_numpy(bool)
        policies[policy] = {"overall": comparison(frame, route), "by_bucket": {}}
        for bucket, group in frame.groupby("frequency_bucket", sort=True):
            policies[policy]["by_bucket"][str(bucket)] = comparison(
                group, route[group.index.to_numpy()]
            )
        if args.progress:
            print(f"[R21] paired bootstrap {policy}", flush=True)
    prediction = routes.prediction_gain_extra_trees.to_numpy(float)
    actual_gain = frame.first_ndcg10.to_numpy(float) - frame.ndcg10.to_numpy(float)
    quantiles = pd.qcut(prediction, 10, labels=False, duplicates="drop")
    calibration = []
    for bucket in sorted(pd.unique(quantiles)):
        mask = np.asarray(quantiles == bucket)
        calibration.append(
            {
                "bin": int(bucket),
                "queries": int(mask.sum()),
                "predicted_gain": float(prediction[mask].mean()),
                "observed_gain": float(actual_gain[mask].mean()),
            }
        )
    # Correlation-based attribution is reported instead of refitting on held-out labels.
    x = frame_matrix(frame)
    attribution = {
        feature: float(np.corrcoef(x[:, index], prediction)[0, 1])
        if np.std(x[:, index]) > 0
        else 0.0
        for index, feature in enumerate(FEATURES)
    }
    payload = {
        "schema": "mind_r21_oof_diagnostics_v1",
        "scope": "R20 dev OOF diagnostics only; not model selection",
        "policies": policies,
        "gain_calibration": {
            "mae": float(mean_absolute_error(actual_gain, prediction)),
            "bins": calibration,
        },
        "feature_prediction_correlation": attribution,
        "drift": {
            "status": "not_run",
            "reason": "R20 OOF admission failed, so confirm was not opened for drift analysis",
        },
        "boundaries": {
            "dev_accessed": True,
            "confirm_accessed": False,
            "large_test_accessed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "report": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
