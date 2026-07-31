"""Summarize per-query FIRST gains and simple diagnostic covariates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def ndcg(order: list[str], rel: dict[str, int]) -> float:
    observed = [rel.get(pid, 0) for pid in order[:10]]
    ideal = sorted(rel.values(), reverse=True)[:10]
    dcg = sum(v / math.log2(i + 2) for i, v in enumerate(observed))
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--first-results", type=Path, required=True)
    p.add_argument("--prp", type=Path, required=True)
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    prompts = [json.loads(line) for line in args.prompts.open(encoding="utf-8")]
    baseline = {
        x["query_id"]: [m["candidate_id"] for m in x["candidate_mapping"]]
        for x in prompts
        if x["variant"] == "baseline"
    }
    queries = {
        x["query_id"]: x["query"]
        for x in prompts
        if x["variant"] == "baseline"
    }
    first = {}
    for line in args.first_results.open(encoding="utf-8"):
        payload = json.loads(line)["payload"]
        if payload["variant"] == "baseline":
            mapping = next(
                x["candidate_mapping"]
                for x in prompts
                if x["query_id"] == payload["query_id"]
                and x["variant"] == "baseline"
            )
            ids = {m["identifier"]: m["candidate_id"] for m in mapping}
            first[payload["query_id"]] = [
                ids[i]
                for i, _ in sorted(
                    payload["identifier_logits"].items(),
                    key=lambda pair: (-float(pair[1]), pair[0]),
                )
            ]
    qrels = pd.read_parquet(args.qrels)
    relevance = {
        str(req): {
            str(row.passage_id): int(row.graded_relevance)
            for row in group.itertuples()
        }
        for req, group in qrels.groupby("request_id", sort=False)
    }
    query_to_request = dict(zip(
        qrels.query_id.astype(str), qrels.request_id.astype(str), strict=False
    ))
    prp = pd.read_parquet(args.prp)
    rows = []
    for qid, bm25_order in baseline.items():
        req = query_to_request[qid]
        rel = relevance[req]
        prp_order = (
            prp[prp.request_id.astype(str) == req]
            .sort_values(["raw_score", "bm25_rank"], ascending=[False, True])
            .passage_id.astype(str)
            .tolist()
        )
        relevant = sum(v > 0 for v in rel.values())
        rows.append(
            {
                "query_id": qid,
                "request_id": req,
                "query_words": len(queries[qid].split()),
                "relevant_docs": relevant,
                "bm25_ndcg10": ndcg(bm25_order, rel),
                "first_ndcg10": ndcg(first[qid], rel),
                "prp_ndcg10": ndcg(prp_order, rel),
            }
        )
    frame = pd.DataFrame(rows)
    frame["first_minus_bm25"] = frame.first_ndcg10 - frame.bm25_ndcg10
    frame["first_minus_prp"] = frame.first_ndcg10 - frame.prp_ndcg10
    frame["bucket"] = frame.first_minus_bm25.map(
        lambda x: "gain" if x > 1e-9 else "loss" if x < -1e-9 else "tie"
    )
    summary = {
        "queries": len(frame),
        "gain_queries": int((frame.bucket == "gain").sum()),
        "loss_queries": int((frame.bucket == "loss").sum()),
        "tie_queries": int((frame.bucket == "tie").sum()),
        "mean_first_minus_bm25": float(frame.first_minus_bm25.mean()),
        "mean_first_minus_prp": float(frame.first_minus_prp.mean()),
        "gain_mean_query_words": float(frame.loc[frame.bucket == "gain", "query_words"].mean()),
        "loss_mean_query_words": float(frame.loc[frame.bucket == "loss", "query_words"].mean()),
        "gain_mean_relevant_docs": float(frame.loc[frame.bucket == "gain", "relevant_docs"].mean()),
        "loss_mean_relevant_docs": float(frame.loc[frame.bucket == "loss", "relevant_docs"].mean()),
        "spearman_gain_query_words": float(
            frame.first_minus_bm25.corr(frame.query_words, method="spearman")
        ),
        "spearman_gain_relevant_docs": float(
            frame.first_minus_bm25.corr(frame.relevant_docs, method="spearman")
        ),
        "top_gains": frame.nlargest(10, "first_minus_bm25")[
            ["query_id", "first_minus_bm25", "query_words", "relevant_docs"]
        ].to_dict("records"),
        "top_losses": frame.nsmallest(10, "first_minus_bm25")[
            ["query_id", "first_minus_bm25", "query_words", "relevant_docs"]
        ].to_dict("records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
