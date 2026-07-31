"""Create a deterministic BM25 top-20 NFCorpus snapshot for FIRST R5."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

TOKEN = re.compile(r"[a-z0-9]+")


def toks(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("train", "dev", "test"), default="test")
    args = p.parse_args()
    root = args.root
    corpus = [json.loads(x) for x in (root / "corpus.jsonl").open()]
    queries = [json.loads(x) for x in (root / "queries.jsonl").open()]
    qrels = pd.read_csv(root / "qrels" / f"{args.split}.tsv", sep="\t")
    qrels = qrels.rename(columns={"query-id": "query_id", "corpus-id": "corpus_id"})
    docs = [(str(d["_id"]), f"{d.get('title', '')} {d.get('text', '')}") for d in corpus]
    df = Counter()
    lengths = {}
    for doc_id, text in docs:
        terms = toks(text)
        lengths[doc_id] = len(terms)
        for term in set(terms):
            df[term] += 1
    avgdl = sum(lengths.values()) / len(lengths)
    n = len(docs)
    postings = {doc_id: Counter(toks(text)) for doc_id, text in docs}
    by_query = defaultdict(list)
    for row in qrels.itertuples():
        by_query[str(row.query_id)].append((str(row.corpus_id), int(row.score)))
    records = []
    for query in queries:
        qid = str(query["_id"])
        if qid not in by_query:
            continue
        qt = Counter(toks(query["text"]))
        scored = []
        for doc_id, _ in docs:
            tf = postings[doc_id]
            score = 0.0
            for term, _qtf in qt.items():
                if term not in tf:
                    continue
                idf = math.log1p((n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
                score += (
                    idf
                    * (tf[term] * 2.0)
                    / (tf[term] + 1.2 * (0.25 + 0.75 * lengths[doc_id] / avgdl))
                )
            scored.append((score, doc_id))
        top = sorted(scored, key=lambda x: (-x[0], x[1]))[:20]
        text_by_id = dict(docs)
        records.append(
            {
                "request_id": f"nfcorpus-{qid}",
                "year": 0,
                "query_id": qid,
                "query": query["text"],
                "candidates": [
                    {"passage_id": doc_id, "bm25_rank": i, "passage": text_by_id[doc_id]}
                    for i, (_, doc_id) in enumerate(top, 1)
                ],
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "teacher_inputs.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    qdf = pd.DataFrame(
        [
            {
                "request_id": f"nfcorpus-{r['query_id']}",
                "year": 0,
                "query_id": r["query_id"],
                "query": r["query"],
            }
            for r in records
        ]
    )
    qdf.to_parquet(args.output / "queries.parquet")
    crows = [
        {
            "request_id": r["request_id"],
            "year": 0,
            "query_id": r["query_id"],
            "passage_id": c["passage_id"],
            "bm25_rank": c["bm25_rank"],
            "passage": c["passage"],
        }
        for r in records
        for c in r["candidates"]
    ]
    pd.DataFrame(crows).to_parquet(args.output / "candidates.parquet")
    qrels_out = qrels.rename(
        columns={"query_id": "query_id", "corpus_id": "passage_id", "score": "graded_relevance"}
    )
    qrels_out["request_id"] = "nfcorpus-" + qrels_out["query_id"].astype(str)
    qrels_out["year"] = 0
    qrels_out.to_parquet(args.output / "qrels.parquet")
    print(
        json.dumps(
            {"queries": len(records), "candidates_per_query": 20, "output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
