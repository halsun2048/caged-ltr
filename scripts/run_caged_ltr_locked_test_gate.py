"""Apply a frozen dev gate once on the untouched official test pool."""

from __future__ import annotations

import argparse
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
    candidates = pd.read_parquet(args.candidates)
    bm25 = (
        candidates.sort_values("bm25_rank").groupby("query_id")["passage_id"].apply(list).to_dict()
    )
    variants: dict[str, dict[str, list[str]]] = {}
    confidence: dict[str, tuple[float, float]] = {}
    for line in args.results.open():
        item = json.loads(line)
        p = item["payload"]
        prompt = prompts[item["key"]]
        mapping = {m["identifier"]: m["candidate_id"] for m in prompt["candidate_mapping"]}
        ranking = [mapping[x] for x in p["first_token_ranking"]]
        variants.setdefault(p["query_id"], {})[p["variant"]] = ranking
        if p["variant"] == "baseline":
            confidence[p["query_id"]] = (
                float(p["normalized_entropy"]),
                float(p["top1_top2_margin"]),
            )
    rows = []
    for q, ranks in variants.items():
        base = ranks["baseline"]
        agreements = []
        for name in ("reverse", "random_permutation"):
            pos = {x: i for i, x in enumerate(ranks[name])}
            agreements.append(np.mean([abs(i - pos[x]) <= 2 for i, x in enumerate(base)]))
        stability = float(np.mean(agreements))
        entropy, margin = confidence[q]
        use_first = entropy <= 0.7 and stability >= 0.2
        rows.append(
            {
                "query_id": q,
                "entropy": entropy,
                "margin": margin,
                "stability": stability,
                "route": "FIRST" if use_first else "BM25",
                "bm25_ndcg10": ndcg([relevance(qrels, q, x) for x in bm25[q]]),
                "first_ndcg10": ndcg([relevance(qrels, q, x) for x in base]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["gate_ndcg10"] = np.where(frame.route == "FIRST", frame.first_ndcg10, frame.bm25_ndcg10)
    result = {
        "split": "untouched_test",
        "queries": len(frame),
        "thresholds_frozen": {"entropy_max": 0.7, "stability_min": 0.2},
        "route_counts": frame.route.value_counts().to_dict(),
        "bm25_ndcg10": float(frame.bm25_ndcg10.mean()),
        "first_ndcg10": float(frame.first_ndcg10.mean()),
        "gate_ndcg10": float(frame.gate_ndcg10.mean()),
        "gate_minus_bm25": float(frame.gate_ndcg10.mean() - frame.bm25_ndcg10.mean()),
        "test_accessed": True,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
