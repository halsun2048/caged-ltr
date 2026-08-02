"""Select a gain-aware MiniLM-to-FIRST route using deployable features only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from select_mind_r8_6_gate import merge, metrics
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "margin",
    "top1_score",
    "top3_gap",
    "top5_gap",
    "score_mean",
    "score_std",
    "score_entropy",
    "student_top1_source_rank",
    "student_top3_mean_source_rank",
    "student_source_top1_agreement",
    "student_source_top3_overlap",
    "top1_lexical_overlap",
    "max_lexical_overlap",
    "mean_lexical_overlap",
    "query_characters",
    "candidate_count",
    "top1_passage_characters",
    "mean_passage_characters",
]


def estimators() -> dict[str, object]:
    return {
        "extra_trees_leaf10": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=10, max_features=0.8,
            random_state=20260802, n_jobs=-1,
        ),
        "extra_trees_leaf30": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=30, max_features=1.0,
            random_state=20260802, n_jobs=-1,
        ),
        "hist_gradient": make_pipeline(
            StandardScaler(),
            HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=20260802,
            ),
        ),
    }


def route_at_budget(prediction: np.ndarray, budget: float) -> np.ndarray:
    count = round(len(prediction) * budget)
    chosen = np.argsort(-prediction, kind="stable")[:count]
    route = np.zeros(len(prediction), dtype=bool)
    route[chosen] = True
    return route


def oracle(frame: pd.DataFrame, budget: float) -> dict[str, float]:
    gain = (frame.first_ndcg10 - frame.ndcg10).to_numpy()
    return metrics(frame, route_at_budget(gain, budget))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", type=Path, default=Path("runs/mind_r8_6_v2"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/experiments/mind_r8_6_gate_v2.json")
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("artifacts/mind_r8_6_gate_v2.joblib")
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    dev = merge("gate_dev", args.metrics_root)
    missing = sorted(set(FEATURES) - set(dev.columns))
    if missing:
        raise RuntimeError(f"missing deployable gate features: {missing}")
    x = dev[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gain = (dev.first_ndcg10 - dev.ndcg10).to_numpy()
    folds = KFold(n_splits=5, shuffle=True, random_state=20260802)
    candidates = []
    predictions: dict[str, np.ndarray] = {}
    for name, prototype in estimators().items():
        oof = np.zeros(len(dev), dtype=float)
        for fold_number, (train_index, valid_index) in enumerate(folds.split(x), 1):
            model = clone(prototype)
            model.fit(x.iloc[train_index], gain[train_index])
            oof[valid_index] = model.predict(x.iloc[valid_index])
            if args.progress:
                print(f"[gate] {name} fold={fold_number}/5", flush=True)
        predictions[name] = oof
        for budget in (0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55):
            result = metrics(dev, route_at_budget(oof, budget))
            candidates.append({"model": name, "budget": budget, **result})
    first_quality = float(dev.first_ndcg10.mean())
    eligible = [
        value for value in candidates
        if value["first_call_rate"] <= 0.55 and value["ndcg10"] >= first_quality - 0.003
    ]
    # Freeze the cheapest policy that is already quality-equivalent to FIRST.
    selected = min(eligible, key=lambda value: value["first_call_rate"]) if eligible else None
    confirm_result = None
    admitted = False
    frozen = None
    if selected:
        model = estimators()[selected["model"]]
        model.fit(x, gain)
        dev_prediction = model.predict(x)
        threshold = float(np.quantile(dev_prediction, 1 - selected["budget"]))
        selected = {**selected, "threshold": threshold}
        # Confirmation is evaluated exactly once after all model/budget choices are frozen.
        confirm = merge("gate_confirm", args.metrics_root)
        confirm_x = confirm[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        confirm_prediction = model.predict(confirm_x)
        route = confirm_prediction >= threshold
        confirm_result = metrics(confirm, route)
        admitted = bool(
            confirm_result["first_call_rate"] <= 0.55
            and confirm_result["ndcg10"] >= float(confirm.first_ndcg10.mean()) - 0.003
            and confirm_result["ndcg10"] >= float(confirm.ndcg10.mean())
        )
        frozen = {"model": model, "features": FEATURES, "selected": selected}
    payload = {
        "schema": "mind_r8_6_gain_gate_v2",
        "target": "per-query FIRST NDCG@10 minus student NDCG@10",
        "features": FEATURES,
        "selection": "five-fold out-of-fold gate-dev predictions; confirmation once after freeze",
        "dev": {
            "student_ndcg10": float(dev.ndcg10.mean()),
            "first_ndcg10": first_quality,
            "oracle_at_50pct": oracle(dev, 0.5),
            "candidates": candidates,
        },
        "selected": selected,
        "confirm": confirm_result,
        "acceptance": {
            "gate_confirm_passed": admitted,
            "first_call_rate_reduced_at_least_45pct": bool(
                confirm_result and confirm_result["first_call_rate"] <= 0.55
            ),
            "large_test_accessed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(frozen, args.model_output)
    print(json.dumps({"stage": "complete", "admitted": admitted, "report": str(args.output)}))


if __name__ == "__main__":
    main()
