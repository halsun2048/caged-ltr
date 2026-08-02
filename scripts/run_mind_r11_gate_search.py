#!/usr/bin/env python3
"""R10 dev-only gate search followed by one fixed confirm evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from select_mind_r8_6_gate import first_frame, metrics

FEATURES = [
    "margin", "top1_score", "top3_gap", "top5_gap", "score_mean", "score_std",
    "score_entropy", "student_top1_source_rank", "student_top3_mean_source_rank",
    "student_source_top1_agreement", "student_source_top3_overlap", "top1_lexical_overlap",
    "max_lexical_overlap", "mean_lexical_overlap", "query_characters", "candidate_count",
    "top1_passage_characters", "mean_passage_characters",
]


def merge(split: str, metrics_path: Path, pool_root: Path, first_root: Path) -> pd.DataFrame:
    first = first_frame(pool_root / f"{split}.parquet", first_root / f"{split}_prompts.jsonl", first_root / f"{split}_first/results.jsonl")
    student = pd.read_parquet(metrics_path)
    return student.merge(first, on="query_id", validate="one_to_one")


def route_for(frame: pd.DataFrame, prediction: np.ndarray, budget: float, tail_floor: float, torso_floor: float) -> np.ndarray:
    n = len(frame); route = np.zeros(n, dtype=bool); total = round(n * budget)
    for bucket, floor in (("tail", tail_floor), ("torso", torso_floor)):
        idx = np.flatnonzero(frame.frequency_bucket.to_numpy() == bucket)
        count = min(len(idx), round(len(idx) * floor))
        if count:
            route[idx[np.argsort(-prediction[idx], kind="stable")[:count]]] = True
    remaining = total - int(route.sum())
    if remaining > 0:
        idx = np.flatnonzero(~route)
        route[idx[np.argsort(-prediction[idx], kind="stable")[:min(remaining, len(idx))]]] = True
    return route


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--pool-root", type=Path, default=Path("data/processed/mind_r10_0")); ap.add_argument("--first-root", type=Path, default=Path("runs/mind_r10_0")); ap.add_argument("--dev-metrics", type=Path, default=Path("runs/mind_r10_0/r10_dev_query_metrics.parquet")); ap.add_argument("--confirm-metrics", type=Path, default=Path("runs/mind_r10_0/r10_confirm_query_metrics.parquet")); ap.add_argument("--output", type=Path, default=Path("reports/experiments/mind_r10_gate_search.json")); ap.add_argument("--progress", action="store_true"); args = ap.parse_args()
    if args.progress: print("[R10 1/4] loading dev and confirm metrics", flush=True)
    dev = merge("dev", args.dev_metrics, args.pool_root, args.first_root); confirm = merge("confirm", args.confirm_metrics, args.pool_root, args.first_root)
    x = dev[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target = dev.first_ndcg10.to_numpy() - dev.ndcg10.to_numpy()
    model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10, random_state=20260802, n_jobs=-1).fit(x, target)
    pred_dev = model.predict(x); pred_confirm = model.predict(confirm[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    first_dev = metrics(dev, np.ones(len(dev), dtype=bool)); first_confirm = metrics(confirm, np.ones(len(confirm), dtype=bool))
    first_dev_buckets = {k: metrics(g, np.ones(len(g), dtype=bool)) for k, g in dev.groupby("frequency_bucket", sort=True)}
    if args.progress: print("[R10 2/4] dev-only budget and floor search", flush=True)
    candidates = []
    for budget in (0.3, 0.4, 0.5):
        for tail in (0.5, 0.75, 0.9, 1.0):
            for torso in (0.0, 0.35, 0.5, 0.65):
                route = route_for(dev, pred_dev, budget, tail, torso); m = metrics(dev, route); b = {k: metrics(g, route[g.index.to_numpy()]) for k, g in dev.groupby("frequency_bucket", sort=True)}
                candidates.append({"budget": budget, "tail_floor": tail, "torso_floor": torso, "overall": m, "by_bucket": b, "dev_gap": first_dev["ndcg10"] - m["ndcg10"], "dev_tail_gap": float(first_dev_buckets["tail"]["ndcg10"] - b["tail"]["ndcg10"])})
    eligible = [c for c in candidates if c["dev_gap"] <= 0.003 and c["dev_tail_gap"] <= 0.003 and c["overall"]["first_call_rate"] <= 0.5]
    selected = max(eligible, key=lambda c: c["overall"]["ndcg10"]) if eligible else min(candidates, key=lambda c: c["dev_gap"])
    if args.progress: print("[R10 3/4] applying frozen policy to confirm", flush=True)
    route = route_for(confirm, pred_confirm, selected["budget"], selected["tail_floor"], selected["torso_floor"])
    confirm_gate = metrics(confirm, route)
    confirm_buckets = {k: {"route_rate": float(route[g.index.to_numpy()].mean()), "gate": metrics(g, route[g.index.to_numpy()]), "student": metrics(g, np.zeros(len(g), dtype=bool)), "first": metrics(g, np.ones(len(g), dtype=bool))} for k, g in confirm.groupby("frequency_bucket", sort=True)}
    payload = {"schema": "mind_r10_gate_search_v1", "selection_split": "dev", "confirm_split": "confirm", "features": FEATURES, "selected": selected, "dev_first": first_dev, "confirm_first": first_confirm, "confirm": {"gate": confirm_gate, "buckets": confirm_buckets}, "ablation": candidates, "acceptance": {"confirm_overall_gap_at_most_0p003": first_confirm["ndcg10"] - confirm_gate["ndcg10"] <= 0.003, "confirm_tail_gap_at_most_0p003": confirm_buckets.get("tail", {}).get("first", {}).get("ndcg10", 0.0) - confirm_buckets.get("tail", {}).get("gate", {}).get("ndcg10", 0.0) <= 0.003, "confirm_call_rate_at_most_0p55": confirm_gate["first_call_rate"] <= 0.55, "large_test_accessed": False}}
    payload["acceptance"]["all_passed"] = all(payload["acceptance"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.progress: print("[R10 4/4] complete", flush=True)
    print(json.dumps({"stage": "complete", "selected": selected, "confirm": confirm_gate, "acceptance": payload["acceptance"]}, ensure_ascii=False))


if __name__ == "__main__": main()
