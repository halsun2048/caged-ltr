"""Evaluate the frozen R19 Gate once on the fixed confirm split (no tuning)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run_mind_r11_gate_search import merge

from caged_ltr.r18_gate_features import FEATURES, frame_matrix


def score(frame, route):
    values = np.where(route, frame.first_ndcg10, frame.ndcg10)
    return {
        "queries": len(frame),
        "ndcg10": float(values.mean()),
        "hit10": float(np.where(route, frame.first_hit10, frame.hit10).mean()),
        "mrr": float(np.where(route, frame.first_mrr, frame.mrr).mean()),
        "first_call_rate": float(route.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest", type=Path, default=Path("artifacts/r19_post_student_gate_oof.json")
    )
    ap.add_argument(
        "--metrics", type=Path, default=Path("runs/mind_r12_0/r12_confirm_query_metrics.parquet")
    )
    ap.add_argument("--pool-root", type=Path, default=Path("data/processed/mind_r12_0"))
    ap.add_argument("--first-root", type=Path, default=Path("runs/mind_r12_0"))
    ap.add_argument(
        "--output", type=Path, default=Path("reports/experiments/mind_r19_gate_confirm.json")
    )
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    payload = json.loads(args.manifest.read_text())
    if payload.get("features") != FEATURES:
        raise SystemExit("manifest feature builder mismatch")
    frame = merge("confirm", args.metrics, args.pool_root, args.first_root)
    x = frame_matrix(frame)
    mean, scale = np.asarray(payload["mean"]), np.asarray(payload["scale"])
    linear = (x - mean) / scale @ np.asarray(payload["coef"]) + float(payload["intercept"])
    probability = 1.0 / (1.0 + np.exp(-linear))
    route = probability >= float(payload["threshold"])
    buckets = {
        str(bucket): score(group, route[group.index.to_numpy()])
        for bucket, group in frame.groupby("frequency_bucket", sort=True)
    }
    result = {
        "schema": "caged_ltr_r19_gate_confirm_v1",
        "selection": "frozen R19 dev OOF manifest; no confirm tuning",
        "manifest_sha256": __import__("hashlib").sha256(args.manifest.read_bytes()).hexdigest(),
        "overall": score(frame, route),
        "by_bucket": buckets,
        "student": score(frame, np.zeros(len(frame), dtype=bool)),
        "first": score(frame, np.ones(len(frame), dtype=bool)),
        "boundaries": {
            "dev_accessed": True,
            "confirm_accessed": True,
            "large_test_accessed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.progress:
        print(
            f"[R19.2] confirm scored {len(frame):,} queries; calls={route.mean():.3f}", flush=True
        )
    print(
        json.dumps(
            {"stage": "complete", "report": str(args.output), "overall": result["overall"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
