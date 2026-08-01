"""Dev-only gate using permutation stability plus FIRST confidence features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ndcg(values: list[float]) -> float:
    values = values[:10]
    dcg = sum((2**v - 1) / np.log2(i + 2) for i, v in enumerate(values))
    ideal = sorted(values, reverse=True)
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
    prompts = {
        json.loads(x)["fingerprint"]: json.loads(x)
        for x in args.prompt_inputs.read_text().splitlines()
    }
    qrels = pd.read_parquet(args.qrels).set_index(["query_id", "passage_id"])["graded_relevance"]
    variants: dict[str, dict[str, list[str]]] = {}
    base_rows = []
    for line in args.results.open():
        item = json.loads(line)
        p = item["payload"]
        prompt = prompts[item["key"]]
        mapping = {m["identifier"]: m["candidate_id"] for m in prompt["candidate_mapping"]}
        ranking = [mapping[x] for x in p["first_token_ranking"]]
        variants.setdefault(p["query_id"], {})[p["variant"]] = ranking
        if p["variant"] == "baseline":
            rank = {x: i for i, x in enumerate(ranking)}
            base_rows.append(
                {
                    "query_id": p["query_id"],
                    "passage_id": ranking[0],
                    "teacher_logit": max(p["identifier_logits"].values()),
                    "teacher_rank": 0,
                    "entropy": p["normalized_entropy"],
                    "margin": p["top1_top2_margin"],
                    "rank_map": rank,
                }
            )
    rows = []
    candidates = pd.read_parquet(args.candidates)
    bm25 = (
        candidates.sort_values("bm25_rank").groupby("query_id")["passage_id"].apply(list).to_dict()
    )
    for row in base_rows:
        q = row["query_id"]
        ranks = variants[q]
        base = ranks["baseline"]
        agreements = []
        for name in ("reverse", "random_permutation"):
            other = ranks[name]
            pos = {x: i for i, x in enumerate(other)}
            agreements.append(np.mean([abs(i - pos[x]) <= 2 for i, x in enumerate(base)]))
        stability = float(np.mean(agreements))
        rows.append(
            {
                "query_id": q,
                "entropy": row["entropy"],
                "margin": row["margin"],
                "stability": stability,
                "first_ndcg": ndcg([relevance(qrels, q, x) for x in base]),
                "bm25_ndcg": ndcg([relevance(qrels, q, x) for x in bm25[q]]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["gain"] = frame.first_ndcg - frame.bm25_ndcg
    order = sorted(frame.query_id, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    train_ids = set(order[: int(0.8 * len(order))])
    train = frame[frame.query_id.isin(train_ids)]
    valid = frame[~frame.query_id.isin(train_ids)]
    grid = []
    for entropy in (0.3, 0.5, 0.7):
        for stability in (0.2, 0.4, 0.6, 0.8):
            use = (train.entropy <= entropy) & (train.stability >= stability)
            score = float(np.mean(np.where(use, train.first_ndcg, train.bm25_ndcg)))
            grid.append({"entropy": entropy, "stability": stability, "train_ndcg10": score})
    best = max(grid, key=lambda x: x["train_ndcg10"])
    use_valid = (valid.entropy <= best["entropy"]) & (valid.stability >= best["stability"])
    valid_score = float(np.mean(np.where(use_valid, valid.first_ndcg, valid.bm25_ndcg)))
    result = {
        "split": "dev_only",
        "queries": len(frame),
        "train_queries": len(train),
        "valid_queries": len(valid),
        "grid": grid,
        "selected": best,
        "valid_gate_ndcg10": valid_score,
        "mean_stability": float(frame.stability.mean()),
        "test_accessed": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
