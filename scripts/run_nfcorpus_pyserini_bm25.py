"""Run the pinned Pyserini BM25 control on the frozen NFCorpus queries."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import pandas as pd
from pyserini.search.lucene import LuceneSearcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    queries = pd.read_parquet(args.queries)
    qrels = pd.read_parquet(args.qrels)
    relevance = {
        (str(row.request_id), str(row.passage_id)): int(row.graded_relevance)
        for row in qrels.itertuples()
    }
    searcher = LuceneSearcher.from_prebuilt_index("beir-v1.0.0-nfcorpus.flat")
    rows: list[dict[str, object]] = []
    for row in queries.itertuples():
        hits = searcher.search(str(row.query), k=args.top_k)
        for rank, hit in enumerate(hits, 1):
            rows.append(
                {
                    "request_id": str(row.request_id),
                    "query_id": str(row.query_id),
                    "passage_id": str(hit.docid),
                    "rank": rank,
                    "score": float(hit.score),
                    "graded_relevance": relevance.get(
                        (str(row.request_id), str(hit.docid)), 0
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    summary = {
        "stage": "complete",
        "queries": len(queries),
        "rows": len(rows),
        "top_k": args.top_k,
        "pyserini_version": importlib.metadata.version("pyserini"),
        "index": "beir-v1.0.0-nfcorpus.flat",
        "qrels_accessed_after_retrieval": True,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
