#!/usr/bin/env python3
"""Validate that a future pool/prompt package is qrels-free before inference."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd

FORBIDDEN = {"relevance", "label", "score", "qrel", "gold"}

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--pool", type=Path, required=True); ap.add_argument("--prompts", type=Path); ap.add_argument("--report", type=Path, required=True); args = ap.parse_args()
    columns = set(pd.read_parquet(args.pool, engine="pyarrow").columns)
    forbidden = sorted(c for c in columns if c.lower() in FORBIDDEN or any(x in c.lower() for x in ("relevance", "qrel", "label")))
    prompt_forbidden = []
    if args.prompts and args.prompts.exists():
        for i, line in enumerate(args.prompts.open(encoding="utf-8"), 1):
            row = json.loads(line)
            if any(k.lower() in FORBIDDEN for k in row): prompt_forbidden.append(i)
    payload = {"schema":"qrels_free_protocol_audit_v1","pool":str(args.pool),"prompts":str(args.prompts) if args.prompts else None,"pool_columns":sorted(columns),"forbidden_pool_columns":forbidden,"forbidden_prompt_lines":prompt_forbidden,"qrels_free":not forbidden and not prompt_forbidden}
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False))
    if not payload["qrels_free"]: raise SystemExit("qrels-free protocol violation")

if __name__ == "__main__": main()
