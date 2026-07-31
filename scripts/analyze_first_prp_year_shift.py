"""Describe observable DL19/DL20 covariate shift for the top-20 audit."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


def ndcg(rank, rel):
    observed = [rel.get(str(x), 0) for x in rank[:10]]
    ideal = sorted(rel.values(), reverse=True)[:10]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(observed))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--prp", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prompts = [json.loads(x) for x in args.prompts.open()]
    mapping = {(x["query_id"], x["variant"]): {m["identifier"]: m["candidate_id"] for m in x["candidate_mapping"]} for x in prompts}
    first = defaultdict(dict)
    for line in args.first.open():
        x = json.loads(line)["payload"]
        ids = mapping[(x["query_id"], x["variant"])]
        first[x["query_id"]][x["variant"]] = {ids[k]: v for k, v in x["identifier_logits"].items()}
    queries = pd.read_parquet(args.queries)
    candidates = pd.read_parquet(args.candidates)
    qrels = pd.read_parquet(args.qrels)
    req = dict(zip(queries.query_id.astype(str), queries.request_id.astype(str), strict=True))
    year = dict(zip(queries.request_id.astype(str), queries.year.astype(int), strict=True))
    text_len = dict(zip(queries.request_id.astype(str), queries["query"].astype(str).str.split().str.len(), strict=True))
    rel = {str(r): {str(x.passage_id): int(x.graded_relevance) for x in g.itertuples()} for r, g in qrels.groupby("request_id", sort=False)}
    prp = pd.read_parquet(args.prp)
    cand_by_req = {str(r): g for r, g in candidates.groupby("request_id", sort=False)}
    rows = []
    for qid, branches in first.items():
        request = req[qid]; pool = set(branches["baseline"])
        first_rank = sorted(pool, key=lambda x: (-branches["baseline"][x], x))
        pr = prp[(prp.request_id.astype(str) == request) & (prp.passage_id.astype(str).isin(pool))].sort_values(["raw_score", "bm25_rank"], ascending=[False, True])["passage_id"].astype(str).tolist()
        cg = cand_by_req[request]
        bm = cg.sort_values("bm25_rank")["passage_id"].astype(str).tolist()
        rows.append({"request_id": request, "year": year[request], "query_words": int(text_len[request]), "judged_rate": float(cg.judged.mean()), "relevant_grade2plus": int((rel[request] and sum(v >= 2 for v in rel[request].values()))), "bm25_ndcg": ndcg(bm, rel[request]), "first_ndcg": ndcg(first_rank, rel[request]), "prp_ndcg": ndcg(pr, rel[request]), "first_minus_prp": ndcg(first_rank, rel[request]) - ndcg(pr, rel[request])})
    frame = pd.DataFrame(rows)
    report = {"queries": len(frame), "by_year": {}}
    for y, group in frame.groupby("year"):
        report["by_year"][str(y)] = {"queries": len(group), **{col: float(group[col].mean()) for col in ["query_words", "judged_rate", "relevant_grade2plus", "bm25_ndcg", "first_ndcg", "prp_ndcg", "first_minus_prp"]}}
    report["correlations_overall"] = {col: float(frame[[col, "first_minus_prp"]].corr().iloc[0, 1]) for col in ["query_words", "judged_rate", "relevant_grade2plus", "bm25_ndcg"]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
