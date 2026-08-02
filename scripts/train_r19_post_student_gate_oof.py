"""R19.1: train a post-Student gain Gate with deterministic five-fold OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from run_mind_r11_gate_search import merge
from sklearn.linear_model import LogisticRegression

from caged_ltr.r18_gate_features import FEATURE_BUILDER_VERSION, FEATURES, frame_matrix


def fold_for(query_id: object, folds: int) -> int:
    digest = hashlib.sha256(str(query_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def metrics(frame, route: np.ndarray) -> dict[str, float]:
    student = frame.ndcg10.to_numpy(float)
    first = frame.first_ndcg10.to_numpy(float)
    values = np.where(route, first, student)
    return {
        "queries": len(values),
        "ndcg10": float(values.mean()),
        "hit10": float(np.where(route, frame.first_hit10, frame.hit10).mean()),
        "mrr": float(np.where(route, frame.first_mrr, frame.mrr).mean()),
        "first_call_rate": float(route.mean()) if len(route) else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics", type=Path, default=Path("runs/mind_r12_0/r12_dev_query_metrics.parquet")
    )
    ap.add_argument("--pool-root", type=Path, default=Path("data/processed/mind_r12_0"))
    ap.add_argument("--first-root", type=Path, default=Path("runs/mind_r12_0"))
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output", type=Path, default=Path("artifacts/r19_post_student_gate_oof.json"))
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    if not 0 < args.budget <= 1 or args.folds < 2:
        raise SystemExit("budget must be in (0,1], folds must be >=2")
    frame = merge("dev", args.metrics, args.pool_root, args.first_root)
    x = frame_matrix(frame)
    y = (frame.first_ndcg10.to_numpy(float) > frame.ndcg10.to_numpy(float)).astype(int)
    folds = np.array([fold_for(value, args.folds) for value in frame.query_id])
    oof_probability = np.zeros(len(frame), dtype=float)
    fold_reports = []
    for fold in range(args.folds):
        train = folds != fold
        valid = folds == fold
        mean, scale = x[train].mean(0), x[train].std(0)
        scale[scale == 0] = 1
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260818)
        model.fit((x[train] - mean) / scale, y[train])
        oof_probability[valid] = model.predict_proba((x[valid] - mean) / scale)[:, 1]
        fold_reports.append(
            {
                "fold": fold,
                "train": int(train.sum()),
                "valid": int(valid.sum()),
                "positive_rate": float(y[valid].mean()),
            }
        )
        if args.progress:
            print(f"[R19.1] fold {fold + 1}/{args.folds} valid={int(valid.sum())}", flush=True)
    count = max(1, round(len(frame) * args.budget))
    threshold = float(np.sort(oof_probability)[::-1][count - 1])
    oof_route = oof_probability >= threshold
    by_bucket = {
        str(bucket): metrics(group, oof_route[group.index.to_numpy()])
        for bucket, group in frame.groupby("frequency_bucket", sort=True)
    }
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    final = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260818)
    final.fit((x - mean) / scale, y)
    final_probability = final.predict_proba((x - mean) / scale)[:, 1]
    final_threshold = float(np.sort(final_probability)[::-1][count - 1])
    payload = {
        "schema": "caged_ltr_r19_post_student_gate_oof_v1",
        "protocol": (
            "R12 dev only; five deterministic query-id folds; "
            "confirm and large-test untouched"
        ),
        "features": FEATURES,
        "feature_builder_version": FEATURE_BUILDER_VERSION,
        "folds": fold_reports,
        "oof": {
            "overall": metrics(frame, oof_route),
            "by_bucket": by_bucket,
            "positive_rate": float(y.mean()),
            "threshold": threshold,
        },
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": final.coef_[0].tolist(),
        "intercept": float(final.intercept_[0]),
        "threshold": final_threshold,
        "budget": args.budget,
        "source_sha256": hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
        "boundaries": {
            "r12_dev_accessed": True,
            "confirm_accessed": False,
            "large_test_accessed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"stage": "complete", "report": str(args.output), "oof": payload["oof"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
