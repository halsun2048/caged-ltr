"""Evaluate a deterministic candidate-identity debiasing post-process for R5.2."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

VARIANTS = ("baseline", "reverse", "random_permutation", "identifier_remap")


def tau(left: list[str], right: list[str]) -> float:
    pos = {x: i for i, x in enumerate(right)}
    seq = [pos[x] for x in left]
    concordant = discordant = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] < seq[j]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def rank(scores: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def zscore(scores: dict[str, float]) -> dict[str, float]:
    values = list(scores.values())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    return {key: (value - mean) / scale if scale else 0.0 for key, value in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-inputs", type=Path, required=True)
    parser.add_argument("--first-token", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = [json.loads(line) for line in args.prompt_inputs.open()]
    mappings = {
        (row["query_id"], row["variant"]): {
            x["identifier"]: x["candidate_id"] for x in row["candidate_mapping"]
        }
        for row in prompts
    }
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in args.first_token.open():
        payload = json.loads(line)["payload"]
        mapping = mappings[(payload["query_id"], payload["variant"])]
        grouped[payload["query_id"]][payload["variant"]] = {
            "scores": {mapping[key]: value for key, value in payload["identifier_logits"].items()},
            "raw_rank": [mapping[key] for key in payload["first_token_ranking"]],
        }

    per_variant: dict[str, list[float]] = defaultdict(list)
    loo: dict[str, list[float]] = defaultdict(list)
    aggregate_taus: list[float] = []
    for query in grouped.values():
        normalized = {variant: zscore(query[variant]["scores"]) for variant in VARIANTS}
        aggregate = {
            candidate: sum(normalized[v][candidate] for v in VARIANTS) / len(VARIANTS)
            for candidate in normalized[VARIANTS[0]]
        }
        aggregate_rank = rank(aggregate)
        aggregate_taus.append(
            sum(tau(aggregate_rank, query[v]["raw_rank"]) for v in VARIANTS) / len(VARIANTS)
        )
        for variant in VARIANTS:
            per_variant[variant].append(tau(aggregate_rank, query[variant]["raw_rank"]))
            held_out = {
                candidate: sum(normalized[v][candidate] for v in VARIANTS if v != variant) / 3
                for candidate in aggregate
            }
            loo[variant].append(tau(rank(held_out), query[variant]["raw_rank"]))

    report = {
        "query_count": len(grouped),
        "method": "candidate-identity alignment; per-variant z-score; mean ensemble",
        "mean_tau_to_debiased_ensemble": sum(aggregate_taus) / len(aggregate_taus),
        "per_variant_tau_to_ensemble": {
            v: sum(per_variant[v]) / len(per_variant[v]) for v in VARIANTS
        },
        "leave_one_variant_out_tau": {v: sum(loo[v]) / len(loo[v]) for v in VARIANTS},
        "acceptance": {
            "all_queries_have_four_variants": all(len(q) == 4 for q in grouped.values()),
            "no_model_training": True,
        },
    }
    report["all_acceptance_pass"] = all(report["acceptance"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
