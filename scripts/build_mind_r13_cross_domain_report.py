"""Synthesize the frozen MIND and independent NFCorpus dev evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    mind_path = Path("reports/experiments/mind_r12_gate_search.json")
    nf_path = Path("reports/experiments/nfcorpus_r6_gate_selection.json")
    nf_formal_path = Path("reports/experiments/nfcorpus_r6_formal_dev.json")
    mind, nf, nf_formal = load(mind_path), load(nf_path), load(nf_formal_path)
    selected = mind["selected"]
    mind_confirm = mind["confirm"]
    nf_validation = nf["validation"]
    nf_first = nf["validation_baselines"]["first"]
    payload = {
        "schema": "mind_r13_cross_domain_v1",
        "datasets": {
            "MIND_independent_confirm": {
                "gate_ndcg10": mind_confirm["gate"]["ndcg10"],
                "first_ndcg10": mind["confirm_first"]["ndcg10"],
                "first_call_rate": mind_confirm["gate"]["first_call_rate"],
                "selected_budget": selected["budget"],
                "protocol": "R12 frozen hard-Tail gate",
            },
            "NFCorpus_independent_dev_validation": {
                "gate_ndcg10": nf_validation["ndcg10"],
                "first_ndcg10": nf_first["ndcg10"],
                "first_call_rate": nf_validation["first_call_rate"],
                "gate_minus_first": nf_validation["ndcg10"] - nf_first["ndcg10"],
                "full_dev_models": {
                    name: values["ndcg10"]
                    for name, values in nf_formal["models"].items()
                },
                "protocol": "previously frozen R6 three-way gate; no re-selection",
            },
        },
        "conclusion": {
            "direction_transfers": nf_validation["ndcg10"] > nf["validation_baselines"]["bm25"]["ndcg10"],
            "near_first_cross_domain": nf_validation["ndcg10"] >= nf_first["ndcg10"] - 0.01,
            "same_gate_parameters_transferred": False,
            "interpretation": "The quality-cost routing principle transfers, but dataset-specific thresholds remain necessary.",
        },
        "boundaries": {
            "nfcorpus_test_reopened": False,
            "mind_large_test_reopened": False,
            "existing_independent_dev_reports_only": True,
        },
        "source_sha256": {str(path): sha(path) for path in (mind_path, nf_path, nf_formal_path)},
    }
    out = Path("reports/experiments/mind_r13_cross_domain.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "report": str(out), "conclusion": payload["conclusion"]}))


if __name__ == "__main__":
    main()
