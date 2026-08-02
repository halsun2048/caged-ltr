"""Run preregistered A/A checks and randomized offline A/B replay on R12 dev."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from run_mind_r11_gate_search import FEATURES, merge
from run_mind_r12_hard_tail_gate import route


def fold_id(value: object) -> int:
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % 5


def arm(value: object, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{value}".encode()).digest()
    return "treatment" if int.from_bytes(digest[:8], "big") / 2**64 < 0.5 else "control"


def paired_ci(delta: np.ndarray, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for start in range(0, draws, 250):
        size = min(250, draws - start)
        idx = rng.integers(0, len(delta), size=(size, len(delta)))
        means[start : start + size] = delta[idx].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def independent_ci(a: np.ndarray, b: np.ndarray, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for start in range(0, draws, 250):
        size = min(250, draws - start)
        ai = rng.integers(0, len(a), size=(size, len(a)))
        bi = rng.integers(0, len(b), size=(size, len(b)))
        values[start : start + size] = a[ai].mean(1) - b[bi].mean(1)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def percentile(values: np.ndarray) -> dict[str, float]:
    return {f"p{q}_ms": float(np.percentile(values, q)) for q in (50, 95, 99)} | {"mean_ms": float(values.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=Path("artifacts/mind_r15_ab_preregistration.json"))
    parser.add_argument("--pool-root", type=Path, default=Path("data/processed/mind_r12_0"))
    parser.add_argument("--first-root", type=Path, default=Path("runs/mind_r12_0"))
    parser.add_argument("--metrics", type=Path, default=Path("runs/mind_r12_0/r12_dev_query_metrics.parquet"))
    parser.add_argument("--service-report", type=Path, default=Path("reports/experiments/mind_r14_service_latency.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/experiments/mind_r15_offline_ab.json"))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text())
    protocol = prereg["protocol_sha256"]
    draws = int(prereg["inference"]["bootstrap_draws"])
    seed = int(prereg["inference"]["seed"])
    if args.progress:
        print("[R15 1/5] merge frozen R12 dev student/FIRST outputs", flush=True)
    frame = merge("dev", args.metrics, args.pool_root, args.first_root).reset_index(drop=True)
    folds = np.asarray([fold_id(value) for value in frame.query_id])
    prediction = np.zeros(len(frame))
    routes = np.zeros(len(frame), dtype=bool)
    for fold in range(5):
        train = folds != fold
        valid = folds == fold
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10, random_state=20260805 + fold, n_jobs=-1)
        model.fit(frame.loc[train, FEATURES].fillna(0), frame.loc[train, "first_ndcg10"] - frame.loc[train, "ndcg10"])
        prediction[valid] = model.predict(frame.loc[valid, FEATURES].fillna(0))
        group = frame.loc[valid].reset_index(drop=True)
        routes[np.flatnonzero(valid)] = route(group, prediction[valid], 0.60, 0.65)
        if args.progress:
            print(f"[R15 2/5] OOF fold {fold + 1}/5 complete", flush=True)
    assignment = np.asarray([arm(value, prereg["randomization"]["salt"]) for value in frame.query_id])
    treatment = assignment == "treatment"
    control = ~treatment
    n_t, n_c = int(treatment.sum()), int(control.sum())
    z = abs(n_t - len(frame) / 2) / math.sqrt(len(frame) * 0.25)
    srm_p = math.erfc(z / math.sqrt(2))
    first = frame.first_ndcg10.to_numpy(float)
    gate = np.where(routes, first, frame.ndcg10.to_numpy(float))
    aa_difference = float(first[treatment].mean() - first[control].mean())
    bucket_balance = {}
    for bucket in ("head", "torso", "tail"):
        flag = frame.frequency_bucket.to_numpy() == bucket
        bucket_balance[bucket] = {"treatment_share": float(flag[treatment].mean()), "control_share": float(flag[control].mean()), "absolute_difference": float(abs(flag[treatment].mean() - flag[control].mean()))}
    aa_ci = independent_ci(first[treatment], first[control], draws, seed)
    if args.progress:
        print("[R15 3/5] A/A SRM and balance checks complete", flush=True)
    randomized_difference = float(gate[treatment].mean() - first[control].mean())
    randomized_ci = independent_ci(gate[treatment], first[control], draws, seed + 1)
    paired_delta = gate - first
    paired = {"difference": float(paired_delta.mean()), "bootstrap_95ci": paired_ci(paired_delta, draws, seed + 2)}
    tail = frame.frequency_bucket.to_numpy() == "tail"
    tail_delta = paired_delta[tail]
    paired_tail = {"difference": float(tail_delta.mean()), "bootstrap_95ci": paired_ci(tail_delta, draws, seed + 3)}
    service = json.loads(args.service_report.read_text())
    student_ms = float(service["student"]["mean_ms"])
    first_latency = frame.first_latency_ms.to_numpy(float)
    treatment_latency = student_ms + np.where(routes, first_latency, 0.0)
    control_latency = first_latency
    call_reduction = float(1 - routes.mean())
    latency_reduction = float(1 - treatment_latency.mean() / control_latency.mean())
    if args.progress:
        print("[R15 4/5] randomized replay, paired effects and latency computed", flush=True)
    aa_limits = prereg["aa_acceptance"]
    ab_limits = prereg["ab_acceptance"]
    aa_acceptance = {
        "srm_pass": srm_p >= aa_limits["srm_p_value_minimum"],
        "bucket_balance_pass": max(x["absolute_difference"] for x in bucket_balance.values()) <= aa_limits["maximum_bucket_share_difference"],
        "ndcg_ci_includes_zero": aa_ci[0] <= 0 <= aa_ci[1],
    }
    aa_acceptance["all_passed"] = all(aa_acceptance.values())
    treatment_p = percentile(treatment_latency)
    control_p = percentile(control_latency)
    ab_acceptance = {
        "quality_noninferiority": paired["bootstrap_95ci"][0] >= ab_limits["paired_quality_ci_lower_bound_minimum"],
        "tail_noninferiority": paired_tail["bootstrap_95ci"][0] >= ab_limits["paired_tail_ci_lower_bound_minimum"],
        "first_call_reduction": call_reduction + 1e-12 >= ab_limits["first_call_reduction_minimum"],
        "mean_latency_reduction": latency_reduction >= ab_limits["mean_latency_reduction_minimum"],
        "p99_guardrail": treatment_p["p99_ms"] - control_p["p99_ms"] <= ab_limits["p99_increase_maximum_ms"],
    }
    ab_acceptance["all_passed"] = all(ab_acceptance.values())
    payload = {
        "schema": "mind_r15_offline_ab_v1", "protocol_sha256": protocol, "queries": len(frame),
        "aa": {"arms": {"treatment": n_t, "control": n_c}, "srm_p_value": srm_p, "ndcg_difference": aa_difference, "bootstrap_95ci": aa_ci, "bucket_balance": bucket_balance, "acceptance": aa_acceptance},
        "ab": {"randomized_replay": {"treatment_gate_ndcg10": float(gate[treatment].mean()), "control_first_ndcg10": float(first[control].mean()), "difference": randomized_difference, "bootstrap_95ci": randomized_ci}, "paired_potential_outcome": paired, "paired_tail": paired_tail, "treatment_first_call_rate": float(routes.mean()), "first_call_reduction": call_reduction, "latency": {"control": control_p, "treatment": treatment_p, "mean_reduction": latency_reduction}, "acceptance": ab_acceptance},
        "online_metrics": prereg["online_only_metrics"],
        "interpretation": "Randomized replay emulates assignment; paired potential-outcome estimates are reported because both offline policies are observable for every dev query.",
        "boundaries": {"r12_dev_accessed": True, "r12_confirm_accessed": False, "large_test_accessed": False, "nfcorpus_test_accessed": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.progress:
        print("[R15 5/5] complete", flush=True)
    print(json.dumps({"stage": "complete", "aa_pass": aa_acceptance["all_passed"], "ab_pass": ab_acceptance["all_passed"], "report": str(args.output)}))


if __name__ == "__main__":
    main()
