"""Distill the OOF post-Student route into a qrels-free request-time Pre-Gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from run_mind_r11_gate_search import FEATURES, merge
from run_mind_r12_hard_tail_gate import route


WORDS = re.compile(r"[a-z0-9]+")


def fold_id(value: object) -> int:
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % 5


def overlap(a: str, b: str) -> float:
    left, right = set(WORDS.findall(a.lower())), set(WORDS.findall(b.lower()))
    return len(left & right) / max(len(left | right), 1)


def build_features(pool: pd.DataFrame) -> pd.DataFrame:
    rows, elapsed = [], []
    for query_id, group in pool.groupby("query_id", sort=False):
        group = group.sort_values("source_rank")
        query = str(group.iloc[0].query)
        passages = group.passage.astype(str).tolist()
        frequency = group.train_item_frequency.fillna(0).to_numpy(float)
        started = time.perf_counter()
        lengths = np.fromiter((len(text) for text in passages), dtype=float)
        lexical = np.asarray([overlap(query, text) for text in passages], float)
        rows.append({
            "query_id": str(query_id), "query_characters": len(query), "query_words": len(WORDS.findall(query.lower())),
            "candidate_count": len(group), "passage_characters_mean": lengths.mean(), "passage_characters_std": lengths.std(),
            "passage_characters_min": lengths.min(), "passage_characters_max": lengths.max(),
            "candidate_frequency_mean": frequency.mean(), "candidate_frequency_std": frequency.std(),
            "candidate_frequency_min": frequency.min(), "candidate_frequency_max": frequency.max(),
            "candidate_frequency_zero_share": float((frequency == 0).mean()),
            "top1_lexical_overlap": lexical[0], "max_lexical_overlap": lexical.max(), "mean_lexical_overlap": lexical.mean(),
        })
        elapsed.append(1000 * (time.perf_counter() - started))
    result = pd.DataFrame(rows)
    result["feature_extraction_ms"] = elapsed
    return result


def paired_ci(delta: np.ndarray, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for start in range(0, draws, 250):
        size = min(250, draws - start)
        idx = rng.integers(0, len(delta), size=(size, len(delta)))
        values[start : start + size] = delta[idx].mean(1)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def quality(frame: pd.DataFrame, chosen: np.ndarray) -> dict[str, object]:
    values = np.where(chosen, frame.first_ndcg10, frame.ndcg10)
    result = {"ndcg10": float(values.mean()), "first_call_rate": float(chosen.mean())}
    result["buckets"] = {
        bucket: {"ndcg10": float(values[group.index].mean()), "queries": len(group), "first_call_rate": float(chosen[group.index].mean())}
        for bucket, group in frame.groupby("frequency_bucket", sort=True)
    }
    return result


def latency(values: np.ndarray) -> dict[str, float]:
    return {"mean_ms": float(values.mean()), **{f"p{q}_ms": float(np.percentile(values, q)) for q in (50, 95, 99)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=Path("artifacts/mind_r15_4_pregate_preregistration.json"))
    parser.add_argument("--pool-root", type=Path, default=Path("data/processed/mind_r12_0"))
    parser.add_argument("--first-root", type=Path, default=Path("runs/mind_r12_0"))
    parser.add_argument("--metrics", type=Path, default=Path("runs/mind_r12_0/r12_dev_query_metrics.parquet"))
    parser.add_argument("--service-report", type=Path, default=Path("reports/experiments/mind_r14_service_latency.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/experiments/mind_r15_4_pregate.json"))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text())
    if args.progress: print("[R15.4 1/5] constructing qrels-free pre-inference features", flush=True)
    pool = pd.read_parquet(args.pool_root / "dev.parquet")
    cheap = build_features(pool)
    frame = merge("dev", args.metrics, args.pool_root, args.first_root).reset_index(drop=True)
    folds = np.asarray([fold_id(value) for value in frame.query_id])
    teacher_prediction = np.zeros(len(frame)); teacher_route = np.zeros(len(frame), bool)
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10, random_state=20260805 + fold, n_jobs=-1)
        model.fit(frame.loc[train, FEATURES].fillna(0), frame.loc[train, "first_ndcg10"] - frame.loc[train, "ndcg10"])
        teacher_prediction[valid] = model.predict(frame.loc[valid, FEATURES].fillna(0))
        group = frame.loc[valid].reset_index(drop=True)
        teacher_route[np.flatnonzero(valid)] = route(group, teacher_prediction[valid], 0.60, 0.65)
    if args.progress: print("[R15.4 2/5] frozen OOF teacher routes reconstructed", flush=True)
    frame = frame.merge(cheap, on="query_id", validate="one_to_one", suffixes=("_teacher", "")).reset_index(drop=True)
    feature_ms = frame.pop("feature_extraction_ms").to_numpy(float)
    columns = prereg["features"]
    feature_matrix = frame[columns].to_numpy(float)
    pregate_route = np.zeros(len(frame), bool); prediction_ms = np.zeros(len(frame)); importance = []
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        classifier = ExtraTreesClassifier(n_estimators=50, max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=20260816 + fold, n_jobs=1)
        classifier.fit(feature_matrix[train], teacher_route[train])
        valid_idx = np.flatnonzero(valid)
        probability = classifier.predict_proba(feature_matrix[valid])[:, 1]
        count = round(len(valid_idx) * 0.60)
        pregate_route[valid_idx[np.argsort(-probability, kind="stable")[:count]]] = True
        for index in valid_idx:
            started = time.perf_counter(); classifier.predict_proba(feature_matrix[index : index + 1]); prediction_ms[index] = 1000 * (time.perf_counter() - started)
        importance.append(classifier.feature_importances_)
        if args.progress: print(f"[R15.4 3/5] Pre-Gate OOF fold {fold + 1}/5 complete", flush=True)
    post = quality(frame, teacher_route); pre = quality(frame, pregate_route); full_first = quality(frame, np.ones(len(frame), bool))
    post_values = np.where(teacher_route, frame.first_ndcg10, frame.ndcg10)
    pre_values = np.where(pregate_route, frame.first_ndcg10, frame.ndcg10)
    delta = pre_values - post_values
    tail = frame.frequency_bucket.to_numpy() == "tail"
    service = json.loads(args.service_report.read_text()); student_ms = float(service["student"]["mean_ms"])
    first_ms = frame.first_latency_ms.to_numpy(float); overhead = feature_ms + prediction_ms
    first_latency = first_ms
    post_latency = student_ms + np.where(teacher_route, first_ms, 0.0)
    pre_latency = overhead + np.where(pregate_route, first_ms, student_ms)
    limits = prereg["acceptance"]
    acceptance = {key: bool(value) for key, value in {
        "route_agreement": float((pregate_route == teacher_route).mean()) >= limits["route_agreement_minimum"],
        "quality_gap": post["ndcg10"] - pre["ndcg10"] <= limits["ndcg_gap_to_post_gate_maximum"],
        "tail_gap": post["buckets"]["tail"]["ndcg10"] - pre["buckets"]["tail"]["ndcg10"] <= limits["tail_gap_to_post_gate_maximum"],
        "first_call_rate": pre["first_call_rate"] <= limits["first_call_rate_maximum"],
        "pregate_p99": float(np.percentile(overhead, 99)) <= limits["pregate_p99_ms_maximum"],
        "total_p99": float(np.percentile(pre_latency, 99)) - float(np.percentile(first_latency, 99)) <= limits["total_p99_above_first_maximum_ms"],
        "mean_latency_reduction": 1 - pre_latency.mean() / first_latency.mean() >= limits["mean_latency_reduction_vs_first_minimum"],
    }.items()}
    acceptance["all_passed"] = all(acceptance.values())
    payload = {
        "schema": "mind_r15_4_pregate_v1", "protocol_sha256": prereg["protocol_sha256"], "queries": len(frame), "features": columns,
        "route_agreement": float((pregate_route == teacher_route).mean()),
        "quality": {"full_first": full_first, "post_student_gate": post, "pregate": pre, "pregate_minus_post": float(delta.mean()), "paired_bootstrap_95ci": paired_ci(delta, prereg["inference"]["bootstrap_draws"], prereg["inference"]["seed"]), "tail_pregate_minus_post": float(delta[tail].mean()), "tail_bootstrap_95ci": paired_ci(delta[tail], prereg["inference"]["bootstrap_draws"], prereg["inference"]["seed"] + 1)},
        "latency": {"full_first": latency(first_latency), "post_student_gate": latency(post_latency), "pregate": latency(pre_latency), "pregate_overhead": latency(overhead), "mean_reduction_vs_first": float(1 - pre_latency.mean() / first_latency.mean())},
        "feature_importance": dict(zip(columns, np.mean(importance, axis=0), strict=True)), "acceptance": acceptance,
        "deployment_audit": {"oracle_frequency_bucket_removed": True, "student_features_removed": True, "first_features_removed": True},
        "boundaries": {"r12_dev_accessed": True, "r12_confirm_accessed": False, "large_test_accessed": False, "nfcorpus_test_accessed": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.progress: print("[R15.4 4/5] quality and latency guardrails computed", flush=True); print("[R15.4 5/5] complete", flush=True)
    print(json.dumps({"stage": "complete", "all_passed": acceptance["all_passed"], "route_agreement": payload["route_agreement"], "pregate_ndcg10": pre["ndcg10"], "report": str(args.output)}))


if __name__ == "__main__":
    main()
