"""Select and lock a query-length fallback using dev predictions only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def ndcg(order: list[str], relevance: dict[str, int]) -> float:
    observed = [relevance.get(pid, 0) for pid in order[:10]]
    ideal = sorted(relevance.values(), reverse=True)[:10]
    dcg = sum(value / math.log2(i + 2) for i, value in enumerate(observed))
    idcg = sum(value / math.log2(i + 2) for i, value in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = [json.loads(line) for line in args.prompts.open(encoding="utf-8")]
    prompt_by_query = {
        x["query_id"]: x for x in prompts if x["variant"] == "baseline"
    }
    qrels = pd.read_parquet(args.qrels)
    relevance = {
        str(request): {
            str(row.passage_id): int(row.graded_relevance)
            for row in group.itertuples()
        }
        for request, group in qrels.groupby("request_id", sort=False)
    }
    request_by_query = dict(
        zip(qrels.query_id.astype(str), qrels.request_id.astype(str), strict=False)
    )
    rows = []
    for line in args.results.open(encoding="utf-8"):
        payload = json.loads(line)["payload"]
        if payload["variant"] != "baseline":
            continue
        prompt = prompt_by_query[payload["query_id"]]
        identifiers = {
            item["identifier"]: item["candidate_id"]
            for item in prompt["candidate_mapping"]
        }
        first_order = [
            identifiers[identifier]
            for identifier, _ in sorted(
                payload["identifier_logits"].items(),
                key=lambda pair: (-float(pair[1]), pair[0]),
            )
        ]
        bm25_order = [item["candidate_id"] for item in prompt["candidate_mapping"]]
        rel = relevance[request_by_query[payload["query_id"]]]
        rows.append(
            {
                "query_id": payload["query_id"],
                "words": len(prompt["query"].split()),
                "bm25": ndcg(bm25_order, rel),
                "first": ndcg(first_order, rel),
            }
        )
    frame = pd.DataFrame(rows)
    candidates = {0: "always_first", 1: "fallback_1_or_less"}
    for threshold in (2, 3, 4):
        candidates[threshold] = f"fallback_below_{threshold}"
    scores = {}
    for threshold, name in candidates.items():
        if threshold == 0:
            values = frame["first"]
        else:
            values = frame.apply(
                lambda row, limit=threshold: row["bm25"]
                if row["words"] < limit
                else row["first"],
                axis=1,
            )
        scores[name] = {
            "threshold": threshold,
            "ndcg10": float(values.mean()),
            "fallback_queries": int((frame.words < threshold).sum()),
        }
    selected = max(
        scores.values(), key=lambda value: (value["ndcg10"], -value["threshold"])
    )
    output = {
        "protocol": {
            "split": "NFCorpus dev only",
            "candidate_thresholds": [1, 2, 3, 4],
            "rule": "words < threshold => BM25, otherwise FIRST",
            "tie_break": "smaller threshold",
            "test_accessed": False,
        },
        "queries": len(frame),
        "candidates": scores,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
