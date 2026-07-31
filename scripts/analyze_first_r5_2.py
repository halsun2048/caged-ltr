"""Summarize FIRST R5.2 perturbation stability without model inference."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def kendall_tau(left: list[str], right: list[str]) -> float:
    pos = {value: i for i, value in enumerate(right)}
    seq = [pos[value] for value in left]
    concordant = discordant = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] < seq[j]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-inputs", type=Path, required=True)
    parser.add_argument("--first-token", type=Path, required=True)
    parser.add_argument("--full-generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt_rows = [json.loads(line) for line in args.prompt_inputs.open()]
    mapping = {row["fingerprint"]: {x["identifier"]: x["candidate_id"] for x in row["candidate_mapping"]} for row in prompt_rows}
    rows = [json.loads(line)["payload"] for line in args.first_token.open()]
    by_query: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        ids = mapping[next(x["fingerprint"] for x in prompt_rows if x["query_id"] == row["query_id"] and x["variant"] == row["variant"])]
        by_query[row["query_id"]][row["variant"]] = {**row, "candidate_ranking": [ids[x] for x in row["first_token_ranking"]]}

    variants = ["baseline", "reverse", "random_permutation", "identifier_remap"]
    summary = {"query_count": len(by_query), "variant_count": {v: sum(v in q for q in by_query.values()) for v in variants}, "variants": {}}
    for variant in variants:
        taus = []
        top1_positions = []
        entropies = []
        margins = []
        for query in by_query.values():
            base = query.get("baseline")
            cur = query.get(variant)
            if not base or not cur:
                continue
            taus.append(kendall_tau(cur["candidate_ranking"], base["candidate_ranking"]))
            entropies.append(cur["normalized_entropy"])
            margins.append(cur["top1_top2_margin"])
            top1_positions.append(cur["first_token_ranking"].index(next(iter(cur["first_token_ranking"]))) + 1)
        summary["variants"][variant] = {
            "paired_kendall_tau_vs_baseline": sum(taus) / len(taus) if taus else None,
            "mean_normalized_entropy": sum(entropies) / len(entropies) if entropies else None,
            "mean_top1_top2_margin": sum(margins) / len(margins) if margins else None,
            "records": len(taus),
        }
    full = [json.loads(line)["payload"] for line in args.full_generation.open()]
    agreements = [row["pair_agreement"] for row in full]
    summary["full_generation"] = {"records": len(full), "mean_pair_agreement": sum(agreements) / len(agreements), "min_pair_agreement": min(agreements), "max_pair_agreement": max(agreements)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
