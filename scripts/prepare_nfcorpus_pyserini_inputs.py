"""Create qrels-free NFCorpus inputs from the official Pyserini BM25 index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pyserini.search.lucene import LuceneSearcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    queries = [json.loads(line) for line in (args.root / "queries.jsonl").open()]
    qrels = pd.read_csv(args.root / "qrels" / f"{args.split}.tsv", sep="\t")
    judged = set(qrels["query-id"].astype(str))
    searcher = LuceneSearcher.from_prebuilt_index("beir-v1.0.0-nfcorpus.flat")
    records = []
    for query in queries:
        query_id = str(query["_id"])
        if query_id not in judged:
            continue
        candidates = []
        for rank, hit in enumerate(searcher.search(str(query["text"]), args.top_k), 1):
            raw = json.loads(searcher.doc(hit.docid).raw())
            text = str(raw.get("contents", raw.get("text", "")))
            candidates.append({"passage_id": str(hit.docid), "bm25_rank": rank, "passage": text})
        if len(candidates) < 2:
            continue
        records.append(
            {
                "request_id": f"nfcorpus-{query_id}",
                "year": 0,
                "query_id": query_id,
                "query": str(query["text"]),
                "candidates": candidates,
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "teacher_inputs.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    pd.DataFrame(
        [
            {
                "request_id": r["request_id"],
                "year": 0,
                "query_id": r["query_id"],
                "query": r["query"],
            }
            for r in records
        ]
    ).to_parquet(args.output / "queries.parquet", index=False)
    pd.DataFrame(
        [
            {"request_id": r["request_id"], "year": 0, "query_id": r["query_id"], **candidate}
            for r in records
            for candidate in r["candidates"]
        ]
    ).to_parquet(args.output / "candidates.parquet", index=False)
    qrels_out = qrels.rename(
        columns={"query-id": "query_id", "corpus-id": "passage_id", "score": "graded_relevance"}
    )
    qrels_out["request_id"] = "nfcorpus-" + qrels_out["query_id"].astype(str)
    qrels_out["year"] = 0
    qrels_out.to_parquet(args.output / "qrels.parquet", index=False)
    print(json.dumps({"split": args.split, "queries": len(records), "top_k": args.top_k}))


if __name__ == "__main__":
    main()
