"""R20: unified OOF Gate comparison and one frozen confirm evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from run_mind_r11_gate_search import merge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression

from caged_ltr.r18_gate_features import FEATURE_BUILDER_VERSION, FEATURES, frame_matrix

SEED = 20260820


def fold_for(query_id: object, folds: int = 5) -> int:
    return int.from_bytes(hashlib.sha256(str(query_id).encode()).digest()[:8], "big") % folds


def top_budget(prediction: np.ndarray, budget: float) -> np.ndarray:
    route = np.zeros(len(prediction), dtype=bool)
    count = max(1, round(len(route) * budget))
    order = np.argsort(-prediction, kind="stable")[:count]
    route[order] = True
    return route


def tail_floor_route(frame, prediction: np.ndarray, budget: float, floor: float) -> np.ndarray:
    route = np.zeros(len(frame), dtype=bool)
    buckets = frame.frequency_bucket.astype(str).to_numpy()
    tail = np.flatnonzero(buckets == "tail")
    tail_count = min(len(tail), round(len(tail) * floor))
    if tail_count:
        route[tail[np.argsort(-prediction[tail], kind="stable")[:tail_count]]] = True
    remaining = round(len(frame) * budget) - int(route.sum())
    if remaining > 0:
        available = np.flatnonzero(~route)
        route[available[np.argsort(-prediction[available], kind="stable")[:remaining]]] = True
    return route


def metrics(frame, route: np.ndarray) -> dict[str, float]:
    return {
        "queries": len(frame),
        "ndcg10": float(np.where(route, frame.first_ndcg10, frame.ndcg10).mean()),
        "hit10": float(np.where(route, frame.first_hit10, frame.hit10).mean()),
        "mrr": float(np.where(route, frame.first_mrr, frame.mrr).mean()),
        "first_call_rate": float(route.mean()),
    }


def bucket_metrics(frame, route: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        str(bucket): metrics(group, route[group.index.to_numpy()])
        for bucket, group in frame.groupby("frequency_bucket", sort=True)
    }


def new_model(kind: str):
    if kind == "logistic":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    return ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=10,
        max_features=0.8,
        n_jobs=-1,
        random_state=SEED,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dev-metrics", type=Path, default=Path("runs/mind_r12_0/r12_dev_query_metrics.parquet")
    )
    ap.add_argument(
        "--confirm-metrics",
        type=Path,
        default=Path("runs/mind_r12_0/r12_confirm_query_metrics.parquet"),
    )
    ap.add_argument("--pool-root", type=Path, default=Path("data/processed/mind_r12_0"))
    ap.add_argument("--first-root", type=Path, default=Path("runs/mind_r12_0"))
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument(
        "--output", type=Path, default=Path("reports/experiments/mind_r20_unified_gate.json")
    )
    ap.add_argument("--routes-output", type=Path, default=Path("runs/mind_r20_dev_routes.parquet"))
    ap.add_argument("--model-output", type=Path, default=Path("artifacts/mind_r20_gate.joblib"))
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    if args.progress:
        print("[R20 1/5] loading frozen dev only", flush=True)
    dev = merge("dev", args.dev_metrics, args.pool_root, args.first_root).reset_index(drop=True)
    x_dev = frame_matrix(dev)
    gain = dev.first_ndcg10.to_numpy(float) - dev.ndcg10.to_numpy(float)
    label = (gain > 0).astype(int)
    folds = np.array([fold_for(value) for value in dev.query_id])
    candidates: dict[str, dict[str, object]] = {}
    candidate_routes: dict[str, np.ndarray] = {}
    candidate_predictions: dict[str, np.ndarray] = {}
    fitted = {}
    for model_index, kind in enumerate(("logistic", "gain_extra_trees"), 1):
        prediction = np.zeros(len(dev))
        for fold in range(5):
            train, valid = folds != fold, folds == fold
            model = new_model(kind)
            model.fit(x_dev[train], label[train] if kind == "logistic" else gain[train])
            prediction[valid] = (
                model.predict_proba(x_dev[valid])[:, 1]
                if kind == "logistic"
                else model.predict(x_dev[valid])
            )
            if args.progress:
                print(f"[R20 2/5] {kind} fold={fold + 1}/5", flush=True)
        route = top_budget(prediction, args.budget)
        candidates[kind] = {
            "runtime_features": FEATURES,
            "requires_frequency_bucket": False,
            "oof": metrics(dev, route),
            "by_bucket": bucket_metrics(dev, route),
        }
        candidate_routes[kind] = route
        candidate_predictions[kind] = prediction.copy()
        final = new_model(kind)
        final.fit(x_dev, label if kind == "logistic" else gain)
        fitted[kind] = final
        if model_index == 2:
            for floor in (0.5, 0.75, 0.9):
                policy = f"tail_floor_{floor:.2f}"
                floor_route = tail_floor_route(dev, prediction, args.budget, floor)
                candidates[policy] = {
                    "runtime_features": [*FEATURES, "frequency_bucket"],
                    "requires_frequency_bucket": True,
                    "base_model": kind,
                    "tail_floor": floor,
                    "oof": metrics(dev, floor_route),
                    "by_bucket": bucket_metrics(dev, floor_route),
                }
                candidate_routes[policy] = floor_route
                candidate_predictions[policy] = prediction.copy()
    first_dev = metrics(dev, np.ones(len(dev), dtype=bool))
    eligible = []
    for name, result in candidates.items():
        tail_gap = first_dev["ndcg10"] - result["oof"]["ndcg10"]
        tail_first = float(dev.loc[dev.frequency_bucket == "tail", "first_ndcg10"].mean())
        tail_gate = result["by_bucket"]["tail"]["ndcg10"]
        result["admission"] = {
            "overall_gap_at_most_0p01": tail_gap <= 0.01,
            "tail_gap_at_most_0p01": tail_first - tail_gate <= 0.01,
            "call_rate_at_most_0p45": result["oof"]["first_call_rate"] <= 0.45,
        }
        result["admission"]["all_passed"] = all(result["admission"].values())
        if result["admission"]["all_passed"]:
            eligible.append(name)
    deployable = [name for name in eligible if not candidates[name]["requires_frequency_bucket"]]
    selected = max(
        deployable or eligible, key=lambda name: candidates[name]["oof"]["ndcg10"], default=None
    )
    if args.progress:
        print(f"[R20 3/5] selected={selected}", flush=True)
    diagnostic_best = max(candidates, key=lambda name: candidates[name]["oof"]["ndcg10"])
    confirm_result = None
    confirm = None
    route = None
    if selected:
        confirm = merge(
            "confirm", args.confirm_metrics, args.pool_root, args.first_root
        ).reset_index(drop=True)
        x_confirm = frame_matrix(confirm)
        base = candidates[selected].get("base_model", selected)
        model = fitted[str(base)]
        prediction = (
            model.predict_proba(x_confirm)[:, 1] if base == "logistic" else model.predict(x_confirm)
        )
        if candidates[selected]["requires_frequency_bucket"]:
            route = tail_floor_route(
                confirm, prediction, args.budget, float(candidates[selected]["tail_floor"])
            )
        else:
            route = top_budget(prediction, args.budget)
        confirm_result = {
            "overall": metrics(confirm, route),
            "by_bucket": bucket_metrics(confirm, route),
        }
    first_confirm = (
        metrics(confirm, np.ones(len(confirm), dtype=bool)) if confirm is not None else None
    )
    student_confirm = (
        metrics(confirm, np.zeros(len(confirm), dtype=bool)) if confirm is not None else None
    )
    confirm_acceptance = None
    if confirm_result:
        tail_first = bucket_metrics(confirm, np.ones(len(confirm), dtype=bool))["tail"]["ndcg10"]
        confirm_acceptance = {
            "overall_gap_at_most_0p01": first_confirm["ndcg10"]
            - confirm_result["overall"]["ndcg10"]
            <= 0.01,
            "tail_gap_at_most_0p01": tail_first - confirm_result["by_bucket"]["tail"]["ndcg10"]
            <= 0.01,
            "call_rate_at_most_0p45": confirm_result["overall"]["first_call_rate"] <= 0.45,
            "better_than_student": confirm_result["overall"]["ndcg10"] > student_confirm["ndcg10"],
        }
        confirm_acceptance["all_passed"] = all(confirm_acceptance.values())
    routes = dev[
        [
            "query_id",
            "frequency_bucket",
            "ndcg10",
            "hit10",
            "mrr",
            "first_ndcg10",
            "first_hit10",
            "first_mrr",
        ]
    ].copy()
    for name, candidate_route in candidate_routes.items():
        routes[f"route_{name}"] = candidate_route
        routes[f"prediction_{name}"] = candidate_predictions[name]
    args.routes_output.parent.mkdir(parents=True, exist_ok=True)
    routes.to_parquet(args.routes_output, index=False)
    payload = {
        "schema": "mind_r20_unified_gate_v1",
        "protocol": {
            "selection": "5-fold OOF dev only",
            "budget": args.budget,
            "features": FEATURES,
            "feature_builder_version": FEATURE_BUILDER_VERSION,
            "confirm_access": "not accessed because no OOF candidate passed admission",
            "initial_attempt_audit": (
                "an earlier implementation loaded confirm before admission but did not score or "
                "select on it; corrected before this final report"
            ),
            "large_test_accessed": False,
        },
        "candidates": candidates,
        "selected": selected,
        "diagnostic_best": diagnostic_best,
        "dev_first": first_dev,
        "confirm": confirm_result,
        "confirm_student": student_confirm,
        "confirm_first": first_confirm,
        "acceptance": confirm_acceptance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if selected:
        joblib.dump(
            {
                "model": fitted[str(candidates[selected].get("base_model", selected))],
                "selected": selected,
                "features": FEATURES,
                "feature_builder_version": FEATURE_BUILDER_VERSION,
                "budget": args.budget,
            },
            args.model_output,
        )
    if args.progress:
        print("[R20 4/5] fixed confirm complete", flush=True)
        print("[R20 5/5] reports and routes saved", flush=True)
    print(
        json.dumps(
            {
                "stage": "complete",
                "selected": selected,
                "confirm": confirm_result,
                "acceptance": confirm_acceptance,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
