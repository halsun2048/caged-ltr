"""Freeze the R15 offline A/A and A/B protocol before outcome replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/mind_r15_ab_preregistration.json"))
    args = parser.parse_args()
    payload = {
        "schema": "mind_r15_ab_preregistration_v1",
        "experiment_id": "mind-r15-tail-safe-gate-vs-first",
        "scope": "offline randomized replay on R12 dev only",
        "randomization": {
            "unit": "query_id",
            "salt": "mind-r15-ab-20260803",
            "control": "full FIRST",
            "treatment": "five-fold OOF Tail-safe Gate",
            "allocation": {"control": 0.5, "treatment": 0.5},
        },
        "frozen_treatment": {"first_budget": 0.60, "tail_floor": 1.0, "torso_floor": 0.65},
        "primary": {
            "quality": "NDCG@10",
            "quality_noninferiority_margin": 0.003,
            "cost": "FIRST call rate",
            "first_call_reduction_minimum": 0.40,
        },
        "secondary": ["Hit@10", "MRR", "mean latency", "P50", "P95", "P99", "throughput proxy"],
        "segments": ["head", "torso", "tail"],
        "aa_acceptance": {
            "srm_p_value_minimum": 0.05,
            "maximum_bucket_share_difference": 0.03,
            "ndcg_difference_ci_must_include_zero": True,
        },
        "ab_acceptance": {
            "paired_quality_ci_lower_bound_minimum": -0.003,
            "paired_tail_ci_lower_bound_minimum": -0.003,
            "first_call_reduction_minimum": 0.40,
            "mean_latency_reduction_minimum": 0.30,
            "p99_increase_maximum_ms": 5.0,
        },
        "inference": {"bootstrap_draws": 10000, "seed": 20260815, "alpha": 0.05},
        "online_only_metrics": {"CTR": None, "long_click_rate": None, "CVR": None, "reason": "not observable in offline MIND replay"},
        "boundaries": {"r12_confirm_accessed": False, "large_test_accessed": False, "nfcorpus_test_accessed": False},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "preregistered", "output": str(args.output), "protocol_sha256": payload["protocol_sha256"]}))


if __name__ == "__main__":
    main()
