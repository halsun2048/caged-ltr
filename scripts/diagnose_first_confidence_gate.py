"""Audit whether FIRST confidence features predict per-query ranking gains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ndcg(labels: list[float], k: int = 10) -> float:
    labels = labels[:k]
    dcg = sum((2**v - 1) / np.log2(i + 2) for i, v in enumerate(labels))
    ideal = sorted(labels, reverse=True)
    idcg = sum((2**v - 1) / np.log2(i + 2) for i, v in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def relevance(qrels: pd.Series, query_id: str, passage_id: str) -> float:
    return float(qrels.get((query_id, passage_id), 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-inputs", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--qrels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    prompts = {}
    with args.prompt_inputs.open() as f:
        for line in f:
            row = json.loads(line)
            if row["variant"] == "baseline":
                prompts[row["fingerprint"]] = row
    qrels = pd.read_parquet(args.qrels).set_index(["query_id", "passage_id"])["graded_relevance"]
    candidates = pd.read_parquet(args.candidates)
    bm25 = candidates.sort_values("bm25_rank").groupby("query_id")["passage_id"].apply(list)
    rows = []
    with args.results.open() as f:
        for line in f:
            record = json.loads(line)
            p = record["payload"]
            if p["variant"] != "baseline":
                continue
            prompt = prompts[record["key"]]
            ids = {x["identifier"]: x["candidate_id"] for x in prompt["candidate_mapping"]}
            first_ids = [ids[x] for x in p["first_token_ranking"]]
            rows.append(
                {
                    "query_id": p["query_id"],
                    "normalized_entropy": p["normalized_entropy"],
                    "top1_top2_margin": p["top1_top2_margin"],
                    "bm25_ndcg10": ndcg(
                        [relevance(qrels, p["query_id"], x) for x in bm25[p["query_id"]]]
                    ),
                    "first_ndcg10": ndcg(
                        [relevance(qrels, p["query_id"], x) for x in first_ids]
                    ),
                }
            )
    frame = pd.DataFrame(rows).drop_duplicates("query_id")
    frame["gain"] = frame["first_ndcg10"] - frame["bm25_ndcg10"]
    frame["confidence_bin"] = pd.qcut(
        frame["normalized_entropy"],
        4,
        labels=["low", "mid_low", "mid_high", "high"],
        duplicates="drop",
    )
    summary = frame.groupby("confidence_bin", observed=True).agg(
        queries=("query_id", "count"), mean_entropy=("normalized_entropy", "mean"),
        mean_margin=("top1_top2_margin", "mean"), mean_gain=("gain", "mean"),
        positive_gain_rate=("gain", lambda x: float((x > 0).mean())),
    ).reset_index()
    result = {
        "queries": len(frame),
        "overall_mean_gain": float(frame["gain"].mean()),
        "bins": summary.to_dict("records"),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
