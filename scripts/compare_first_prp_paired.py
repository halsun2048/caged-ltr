"""Paired FIRST/PRP comparison on the identical top-10 candidate set."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def ndcg(ranking, relevance):
    observed = [relevance.get(str(pid), 0) for pid in ranking[:10]]
    ideal = sorted(relevance.values(), reverse=True)[:10]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(observed))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first-prompts", type=Path, required=True)
    p.add_argument("--first-results", type=Path, required=True)
    p.add_argument("--prp", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prompts = [json.loads(line) for line in args.first_prompts.open()]
    mapping = {
        (x["query_id"], x["variant"]): {
            m["identifier"]: m["candidate_id"] for m in x["candidate_mapping"]
        }
        for x in prompts
    }
    first = defaultdict(dict)
    for line in args.first_results.open():
        x = json.loads(line)["payload"]
        ids = mapping[(x["query_id"], x["variant"])]
        first[x["query_id"]][x["variant"]] = {ids[k]: v for k, v in x["identifier_logits"].items()}
    qrels = pd.read_parquet(args.qrels)
    queries = pd.read_parquet(args.queries)
    request_for_query = dict(
        zip(queries.query_id.astype(str), queries.request_id.astype(str), strict=True)
    )
    year_for_request = dict(
        zip(queries.request_id.astype(str), queries.year.astype(int), strict=True)
    )
    relevance = {
        str(req): {str(r.passage_id): int(r.graded_relevance) for r in g.itertuples()}
        for req, g in qrels.groupby("request_id", sort=False)
    }
    rows = pd.read_parquet(args.prp)
    values = []
    for query_id, branches in first.items():
        request_id = request_for_query[query_id]
        candidate_ids = set(branches["baseline"])
        prp = (
            rows[
                (rows.request_id.astype(str) == request_id)
                & (rows.passage_id.astype(str).isin(candidate_ids))
            ]
            .sort_values(["raw_score", "bm25_rank"], ascending=[False, True])["passage_id"]
            .astype(str)
            .tolist()
        )
        first_rank = sorted(candidate_ids, key=lambda pid: (-branches["baseline"][pid], pid))
        values.append(
            {
                "first": ndcg(first_rank, relevance[request_id]),
                "prp": ndcg(prp, relevance[request_id]),
                "request_id": request_id,
                "year": year_for_request[request_id],
            }
        )
    rng = random.Random(42)
    diffs = []
    for _ in range(10000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        diffs.append(sum(x["first"] - x["prp"] for x in sample) / len(sample))
    report = {
        "queries": len(values),
        "first_ndcg": sum(x["first"] for x in values) / len(values),
        "prp_ndcg_same_candidate_pool": sum(x["prp"] for x in values) / len(values),
        "first_minus_prp": sum(x["first"] - x["prp"] for x in values) / len(values),
        "paired_bootstrap_95ci_first_minus_prp": [sorted(diffs)[250], sorted(diffs)[9749]],
        "candidate_protocol": (
            "PRP ranking filtered to the same top-20 BM25 candidate pool used by FIRST"
        ),
        "qrels_accessed_after_prediction_freeze": True,
        "by_year": {},
    }
    for year in sorted({x["year"] for x in values}):
        subset = [x for x in values if x["year"] == year]
        year_diffs = []
        for _ in range(10000):
            sample = [subset[rng.randrange(len(subset))] for _ in subset]
            year_diffs.append(sum(x["first"] - x["prp"] for x in sample) / len(sample))
        report["by_year"][str(year)] = {
            "queries": len(subset),
            "first_ndcg": sum(x["first"] for x in subset) / len(subset),
            "prp_ndcg": sum(x["prp"] for x in subset) / len(subset),
            "difference": sum(x["first"] - x["prp"] for x in subset) / len(subset),
            "paired_bootstrap_95ci": [sorted(year_diffs)[250], sorted(year_diffs)[9749]],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
