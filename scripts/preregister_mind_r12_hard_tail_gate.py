#!/usr/bin/env python3
"""Audit R10/R11 failures and freeze the R12 hard-Tail search space.

Reporting/preregistration only: no labels are read and no evaluation is run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/experiments"
ART = ROOT / "artifacts"


def load(name: str) -> dict:
    return json.loads((EXP / name).read_text(encoding="utf-8"))


def main() -> None:
    r10 = load("mind_r10_gate_search.json")
    r11 = load("mind_r11_gate_search.json")
    selected = {"budget": [0.45, 0.50, 0.55, 0.60], "tail_floor": [1.0], "torso_floor": [0.35, 0.50, 0.65], "head_policy": "gain-ranked residual budget"}
    audit = {
        "schema": "mind_r12_0_failure_audit_v1",
        "status": "locked_for_next_split",
        "r10": {"selected": r10["selected"], "confirm": r10["confirm"], "acceptance": r10["acceptance"]},
        "r11": {"selected": r11["selected"], "confirm": r11["confirm"], "acceptance": r11["acceptance"]},
        "observations": [
            "R10 and R11 both pass overall but fail Tail on independent confirm.",
            "R11 execution used tail_floor=0.5 although the intended preregistration was 0.75/0.9/1.0; it is therefore exploratory, not a strict preregistered confirmation.",
            "The next strategy must hard-route all Tail queries to FIRST and allocate residual budget to Torso/Head.",
        ],
        "r12_search_space": selected,
        "confirm_policy": "new independent confirm only; one evaluation; no historical confirm reruns; no large-test access",
    }
    payload = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    (EXP / "mind_r12_failure_audit.json").write_text(payload, encoding="utf-8")
    prereg = {
        "schema": "mind_r12_0_hard_tail_preregistration_v1",
        "source_audit": "reports/experiments/mind_r12_failure_audit.json",
        "selection_split": "new_r12_dev_only",
        "confirm_split": "new_r12_confirm_once",
        "budget": selected["budget"], "tail_floor": selected["tail_floor"], "torso_floor": selected["torso_floor"],
        "selection_rule": ["Tail gap <= 0.003", "Torso gap minimized", "then minimum FIRST call rate", "no confirm-based tuning"],
        "large_test_accessed": False,
        "historical_confirms_reopened": False,
    }
    (ART / "mind_r12_0_hard_tail_preregistration.json").write_text(json.dumps(prereg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256((ART / "mind_r12_0_hard_tail_preregistration.json").read_bytes()).hexdigest()
    print(json.dumps({"stage": "complete", "audit": str(EXP / "mind_r12_failure_audit.json"), "preregistration": str(ART / "mind_r12_0_hard_tail_preregistration.json"), "preregistration_sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
