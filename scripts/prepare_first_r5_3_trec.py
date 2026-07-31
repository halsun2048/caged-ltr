"""Prepare FIRST prompts for held-out TREC-DL top-10 candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from caged_ltr.teachers.first import (
    FIRST_MODEL,
    FIRST_MODEL_REVISION,
    FirstCandidate,
    build_prompt_entries,
    fit_first_prompt,
    stable_sha256,
)
from caged_ltr.teachers.prp_real import load_teacher_inputs

VARIANTS = ("baseline", "reverse", "random_permutation", "identifier_remap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=10)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(FIRST_MODEL, revision=FIRST_MODEL_REVISION, token=False)
    queries = load_teacher_inputs(args.teacher_inputs)
    if args.query_limit:
        queries = queries[: args.query_limit]
    records = []
    for query in queries:
        candidates = tuple(FirstCandidate(c.passage_id, c.passage, c.bm25_rank) for c in query.candidates[: args.candidate_limit])
        for variant in VARIANTS:
            entries = build_prompt_entries(candidates, query_id=query.query_id, variant=variant, seed=42)
            prompt, token_count, word_budget = fit_first_prompt(tokenizer, query.query, entries, context_size=4096, initial_max_passage_words=300)
            mapping = [{"candidate_id": e.candidate.candidate_id, "input_position": e.input_position, "identifier": e.identifier, "retrieval_rank": e.candidate.retrieval_rank} for e in entries]
            records.append({"schema": "first_prompt_input_v1", "query_id": query.query_id, "slate_id": f"{query.query_id}:{variant}", "variant": variant, "query": query.query, "candidate_mapping": mapping, "prompt": prompt, "first_token_prompt": prompt + "[", "prompt_token_count": token_count, "max_passage_words": word_budget, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "fingerprint": stable_sha256({"query_id": query.query_id, "variant": variant, "mapping": mapping, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"queries": len(queries), "records": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
