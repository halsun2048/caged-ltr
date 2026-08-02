"""Record the independent-dev cross-domain result and its transfer boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    nested_path = Path("reports/experiments/nfcorpus_r7_2_nested_gate.json")
    prior_path = Path("reports/experiments/mind_r13_cross_domain.json")
    nested = json.loads(nested_path.read_text())
    prior = json.loads(prior_path.read_text())
    payload = {
        "schema": "mind_r14_cross_domain_audit_v1",
        "dataset": "NFCorpus independent dev only",
        "protocol": "5-fold outer / 4-fold inner nested validation; no NFCorpus test access",
        "observed": {
            "oof_ndcg10": nested["oof"]["ndcg10"],
            "first_ndcg10": nested["baselines"]["first_ndcg10"],
            "bm25_ndcg10": nested["baselines"]["bm25_ndcg10"],
            "first_call_rate": nested["oof"]["first_call_rate"],
            "gate_minus_first_bootstrap_95ci": nested["paired_bootstrap_95ci"]["gate_minus_first"],
        },
        "decision": {
            "principle_transfers": True,
            "same_mind_thresholds_transferred": False,
            "status": "supportive_but_not_strict_zero_shot_transfer",
            "reason": "NFCorpus confirms the quality-cost routing pattern, but prior protocol used dataset-specific calibration; this is not evidence that MIND thresholds transfer unchanged.",
        },
        "boundaries": {
            "mind_large_test_reopened": False,
            "nfcorpus_test_reopened": False,
        },
        "source_sha256": {str(path): sha(path) for path in (nested_path, prior_path)},
    }
    out = Path("reports/experiments/mind_r14_cross_domain_audit.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "status": payload["decision"]["status"], "report": str(out)}))


if __name__ == "__main__":
    main()
