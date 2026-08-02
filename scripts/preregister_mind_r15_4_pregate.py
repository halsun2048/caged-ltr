"""Freeze the qrels-free Pre-Gate distillation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    features = [
        "query_characters", "query_words", "candidate_count",
        "passage_characters_mean", "passage_characters_std", "passage_characters_min", "passage_characters_max",
        "candidate_frequency_mean", "candidate_frequency_std", "candidate_frequency_min", "candidate_frequency_max", "candidate_frequency_zero_share",
        "top1_lexical_overlap", "max_lexical_overlap", "mean_lexical_overlap",
    ]
    payload = {
        "schema": "mind_r15_4_pregate_preregistration_v1",
        "experiment_id": "mind-r15-4-qrels-free-pregate",
        "source_split": "R12 dev only",
        "teacher": "five-fold OOF Tail-safe post-Student Gate route",
        "student_router": "five-fold OOF ExtraTreesClassifier with fixed 60% FIRST budget",
        "features": features,
        "forbidden_features": ["relevance", "positive_frequency", "frequency_bucket", "student score", "student margin", "student entropy", "FIRST logits"],
        "acceptance": {
            "route_agreement_minimum": 0.95,
            "ndcg_gap_to_post_gate_maximum": 0.003,
            "tail_gap_to_post_gate_maximum": 0.003,
            "first_call_rate_maximum": 0.60,
            "pregate_p99_ms_maximum": 0.5,
            "total_p99_above_first_maximum_ms": 1.0,
            "mean_latency_reduction_vs_first_minimum": 0.30,
        },
        "inference": {"folds": 5, "seed": 20260816, "bootstrap_draws": 10000},
        "boundaries": {"r12_confirm_accessed": False, "large_test_accessed": False, "nfcorpus_test_accessed": False},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    path = Path("artifacts/mind_r15_4_pregate_preregistration.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "preregistered", "output": str(path), "protocol_sha256": payload["protocol_sha256"]}))


if __name__ == "__main__":
    main()
