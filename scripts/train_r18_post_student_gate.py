"""Train a deployable post-Student gain router on R12 dev only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from run_mind_r11_gate_search import merge
from sklearn.linear_model import LogisticRegression

from caged_ltr.r18_gate_features import FEATURE_BUILDER_VERSION, FEATURES, frame_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", type=Path, default=Path("runs/mind_r12_0/r12_dev_query_metrics.parquet")
    )
    parser.add_argument("--pool-root", type=Path, default=Path("data/processed/mind_r12_0"))
    parser.add_argument("--first-root", type=Path, default=Path("runs/mind_r12_0"))
    parser.add_argument("--budget", type=float, default=0.4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/r18_post_student_gate.json"))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    frame = merge("dev", args.metrics, args.pool_root, args.first_root)
    x = frame_matrix(frame)
    y = (frame.first_ndcg10.to_numpy(float) > frame.ndcg10.to_numpy(float)).astype(int)
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260818)
    model.fit((x - mean) / scale, y)
    probability = model.predict_proba((x - mean) / scale)[:, 1]
    count = round(len(probability) * args.budget)
    threshold = float(np.sort(probability)[::-1][count - 1])
    chosen = probability >= threshold
    payload = {
        "schema": "caged_ltr_r18_post_student_gate_v1",
        "protocol": (
            "R12 dev only; labels are FIRST-vs-Student gain and are never request-time inputs"
        ),
        "features": FEATURES,
        "feature_builder_version": FEATURE_BUILDER_VERSION,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "threshold": threshold,
        "budget": args.budget,
        "dev": {
            "queries": len(frame),
            "positive_rate": float(y.mean()),
            "first_call_rate": float(chosen.mean()),
            "label_accuracy": float(((probability >= 0.5) == y).mean()),
        },
        "source_sha256": hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
        "boundaries": {
            "r12_dev_accessed": True,
            "confirm_accessed": False,
            "large_test_accessed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.progress:
        print(f"[R18.1] fitted {len(frame)} dev queries; call_rate={chosen.mean():.3f}", flush=True)
    print(
        json.dumps(
            {
                "stage": "complete",
                "report": str(args.output),
                "first_call_rate": float(chosen.mean()),
                "label_accuracy": payload["dev"]["label_accuracy"],
            }
        )
    )


if __name__ == "__main__":
    main()
