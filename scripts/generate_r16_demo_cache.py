"""Generate a portable 200-query demo cache from existing R12 dev assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from r16_llm_app import as_json, explain_result
from r16_service import Candidate, MiniLMBackend, ReplayFirstBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--first-results", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    student = MiniLMBackend(str(args.model), str(args.checkpoint))
    first = ReplayFirstBackend(args.first_results)
    rows = []
    for index, (_, group) in enumerate(
        list(pd.read_parquet(args.pool).groupby("query_id", sort=False))[: args.limit], 1
    ):
        query = str(group.iloc[0].query)
        candidates = [
            Candidate(str(index), str(text))
            for index, text in enumerate(group.passage.astype(str).tolist()[:20])
        ]
        student_result = student.rerank(query, candidates)
        first_result = first.rerank(query, candidates)
        rows.append(
            {
                "query": query,
                "understanding": as_json(query),
                "candidates": [item.__dict__ for item in candidates],
                "student": [item.__dict__ for item in student_result],
                "first_replay": [item.__dict__ for item in first_result],
                "explanations": [
                    explain_result(query, item.item_id, item.text, item.score, "minilm")
                    for item in student_result[:3]
                ],
            }
        )
        if args.progress and index % 25 == 0:
            print(f"[R16 cache] {index}/{args.limit}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"schema": "caged_ltr_r16_demo_cache_v1", "queries": len(rows), "rows": rows},
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps({"stage": "complete", "queries": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
