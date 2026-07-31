"""Evaluate R5.3 debiased FIRST scores on held-out TREC-DL qrels."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

VARIANTS = ("baseline", "reverse", "random_permutation", "identifier_remap")


def zscore(values: dict[str, float]) -> dict[str, float]:
    mean = sum(values.values()) / len(values)
    scale = math.sqrt(sum((v - mean) ** 2 for v in values.values()) / len(values))
    return {k: (v - mean) / scale if scale else 0.0 for k, v in values.items()}


def ndcg(ranking: list[str], relevance: dict[str, int], cutoff: int = 10) -> float:
    observed = [relevance.get(pid, 0) for pid in ranking[:cutoff]]
    ideal = sorted(relevance.values(), reverse=True)[:cutoff]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(observed))
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = [json.loads(line) for line in args.prompt_inputs.open()]
    mapping = {
        (p["query_id"], p["variant"]): {
            x["identifier"]: x["candidate_id"] for x in p["candidate_mapping"]
        }
        for p in prompts
    }
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for line in args.results.open():
        row = json.loads(line)["payload"]
        ids = mapping[(row["query_id"], row["variant"])]
        grouped[row["query_id"]][row["variant"]] = {
            ids[k]: v for k, v in row["identifier_logits"].items()
        }

    candidates = pd.read_parquet(args.candidates)
    qrels = pd.read_parquet(args.qrels)
    queries = pd.read_parquet(args.queries)
    request_for_query = dict(
        zip(queries["query_id"].astype(str), queries["request_id"].astype(str), strict=True)
    )
    relevance = {
        str(req): {str(r.passage_id): int(r.graded_relevance) for r in group.itertuples()}
        for req, group in qrels.groupby("request_id", sort=False)
    }
    scores = defaultdict(lambda: {"bm25": [], "first_baseline": [], "first_debiased": []})
    for query_id, variants in grouped.items():
        request_id = request_for_query[query_id]
        candidate_ids = list(variants["baseline"])
        baseline_rank = sorted(candidate_ids, key=lambda pid: (-variants["baseline"][pid], pid))
        normalized = {v: zscore(variants[v]) for v in VARIANTS}
        debiased = {pid: sum(normalized[v][pid] for v in VARIANTS) / 4 for pid in candidate_ids}
        debiased_rank = sorted(candidate_ids, key=lambda pid: (-debiased[pid], pid))
        bm25_rank = (
            candidates[candidates["request_id"].astype(str) == request_id]
            .sort_values("bm25_rank")["passage_id"]
            .astype(str)
            .tolist()
        )
        scores[request_id]["bm25"].append(ndcg(bm25_rank, relevance[request_id]))
        scores[request_id]["first_baseline"].append(ndcg(baseline_rank, relevance[request_id]))
        scores[request_id]["first_debiased"].append(ndcg(debiased_rank, relevance[request_id]))
    report = {
        "queries": len(scores),
        "metrics": {
            metric: sum(row[metric][0] for row in scores.values()) / len(scores)
            for metric in ("bm25", "first_baseline", "first_debiased")
        },
        "by_year": {},
    }
    ordered = list(scores.values())
    rng = random.Random(42)
    bootstrap = {
        "first_baseline_minus_bm25": [],
        "first_debiased_minus_bm25": [],
        "first_debiased_minus_baseline": [],
    }
    for _ in range(10000):
        sample = [ordered[rng.randrange(len(ordered))] for _ in ordered]
        bootstrap["first_baseline_minus_bm25"].append(
            sum(x["first_baseline"][0] - x["bm25"][0] for x in sample) / len(sample)
        )
        bootstrap["first_debiased_minus_bm25"].append(
            sum(x["first_debiased"][0] - x["bm25"][0] for x in sample) / len(sample)
        )
        bootstrap["first_debiased_minus_baseline"].append(
            sum(x["first_debiased"][0] - x["first_baseline"][0] for x in sample) / len(sample)
        )
    report["paired_bootstrap_95ci"] = {
        key: [sorted(values)[250], sorted(values)[9749]] for key, values in bootstrap.items()
    }
    request_year = dict(
        zip(queries["request_id"].astype(str), queries["year"].astype(int), strict=True)
    )
    for year in sorted(set(request_year.values())):
        subset = [scores[r] for r, y in request_year.items() if y == year]
        report["by_year"][str(year)] = {
            metric: sum(row[metric][0] for row in subset) / len(subset)
            for metric in ("bm25", "first_baseline", "first_debiased")
        }
    report["metric"] = "linear graded NDCG@10 with complete official qrels"
    report["qrels_accessed_after_inference_freeze"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
