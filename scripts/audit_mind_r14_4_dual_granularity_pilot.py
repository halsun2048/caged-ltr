"""Pre-registered R14.4 feasibility/go-no-go audit for dual-granularity KD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listwise", type=Path, required=True)
    parser.add_argument("--pairwise", type=Path, required=True)
    parser.add_argument("--teacher-logits", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    listwise = pd.read_parquet(args.listwise)
    pairwise = pd.read_parquet(args.pairwise)
    required_listwise = {"query_id", "query", "passage", "relevance"}
    required_pairwise = {"query_id", "query", "positive_passage", "negative_passage"}
    missing_listwise = sorted(required_listwise - set(listwise.columns))
    missing_pairwise = sorted(required_pairwise - set(pairwise.columns))
    logits_ready = args.teacher_logits is not None and args.teacher_logits.exists()
    payload = {
        "schema": "mind_r14_4_dual_granularity_audit_v1",
        "pilot": {
            "listwise_rows": len(listwise),
            "pairwise_rows": len(pairwise),
            "listwise_query_count": int(listwise.query_id.nunique()),
            "pairwise_query_count": int(pairwise.query_id.nunique()),
            "teacher_logits_present": logits_ready,
            "missing_listwise_columns": missing_listwise,
            "missing_pairwise_columns": missing_pairwise,
        },
        "decision": "go" if logits_ready and not missing_listwise and not missing_pairwise else "no_go_pending_teacher_logit_alignment",
        "reason": "The current listwise package contains relevance labels but no aligned FIRST logits; training a joint KD pilot before alignment would not test the stated method.",
        "large_test_accessed": False,
        "confirm_accessed": False,
        "source_sha256": {str(path): sha256(path) for path in (args.listwise, args.pairwise) if path.exists()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "decision": payload["decision"], "report": str(args.output)}))


if __name__ == "__main__":
    main()
