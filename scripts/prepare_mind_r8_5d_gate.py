"""Materialize frozen gate pools and FIRST prompts for R8.5d."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
from transformers import AutoTokenizer

from caged_ltr.teachers.first import (
    FIRST_MODEL,
    FIRST_MODEL_REVISION,
    FirstCandidate,
    build_prompt_entries,
    fit_first_prompt,
    stable_sha256,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sql_paths(paths: list[Path]) -> str:
    return "[" + ",".join("'" + str(path).replace("'", "''") + "'" for path in paths) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/external/mind/mteb_english"))
    parser.add_argument(
        "--split-ids", type=Path, default=Path("data/processed/mind_r8_5b/gate_split_ids.parquet")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/mind_r8_5d"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mind_r8_5d"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/mind_r8_5d_gate_package.json")
    )
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--splits", nargs="+", default=["gate_dev", "gate_confirm"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    query_file = next((args.raw_root / "queries").glob("*.parquet"))
    corpus_file = next((args.raw_root / "corpus").glob("*.parquet"))
    labels = sorted((args.raw_root / "data").glob("*.parquet"))
    candidates = sorted((args.raw_root / "top_ranked").glob("*.parquet"))
    if not labels or not candidates:
        raise FileNotFoundError("raw MTEB MIND label/candidate shards are missing")
    connection = duckdb.connect()
    connection.execute("SET memory_limit='12GB'")
    connection.execute("SET threads=8")
    outputs = {}
    for split in args.splits:
        output = args.output_dir / f"{split}.parquet"
        outputs[split] = output
        if output.exists() and args.resume:
            continue
        if args.progress:
            print(f"[pool] materializing {split} top-{args.candidates}", flush=True)
        connection.execute(
            f"""
            COPY (
              WITH selected AS (
                SELECT query_id FROM read_parquet('{args.split_ids}') WHERE split = '{split}'
              ), q AS (
                SELECT id AS query_id, text AS query FROM read_parquet('{query_file}')
              ), l AS (
                SELECT "query-id" AS query_id, "corpus-id" AS corpus_id, score AS relevance
                FROM read_parquet({sql_paths(labels)})
              ), candidate_selected AS (
                SELECT r."query-id" AS query_id, r."corpus-ids" AS corpus_ids
                FROM read_parquet({sql_paths(candidates)}) r
                INNER JOIN selected s ON r."query-id" = s.query_id
              ), raw_rank AS (
                SELECT c.query_id, list_extract(c.corpus_ids, u.rank) AS corpus_id,
                       u.rank AS source_rank
                FROM candidate_selected c,
                     UNNEST(range(1, least(len(c.corpus_ids), {args.candidates}) + 1)) u(rank)
              ), ranked AS (
                SELECT query_id, corpus_id, min(source_rank)::INTEGER AS source_rank
                FROM raw_rank GROUP BY query_id, corpus_id
              ), corpus AS (
                SELECT id AS corpus_id, text AS passage FROM read_parquet('{corpus_file}')
              )
              SELECT r.query_id, q.query, r.corpus_id, c.passage,
                     coalesce(l.relevance, 0)::INTEGER AS relevance, r.source_rank,
                     '{split}' AS split
              FROM ranked r INNER JOIN q USING(query_id)
              INNER JOIN corpus c USING(corpus_id)
              LEFT JOIN l USING(query_id, corpus_id)
              ORDER BY r.query_id, r.source_rank
            ) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    tokenizer = AutoTokenizer.from_pretrained(
        FIRST_MODEL, revision=FIRST_MODEL_REVISION, token=False
    )
    prompt_files = {}
    variants = ("baseline", "reverse", "random_permutation")
    for split, pool_path in outputs.items():
        prompt_path = args.run_dir / f"{split}_prompts.jsonl"
        prompt_files[split] = prompt_path
        if prompt_path.exists() and args.resume:
            continue
        pool = pd.read_parquet(pool_path)
        temporary = prompt_path.with_suffix(".jsonl.tmp")
        with temporary.open("w") as handle:
            total = pool.query_id.nunique()
            for number, (query_id, group) in enumerate(pool.groupby("query_id", sort=False), 1):
                query = str(group.iloc[0]["query"])
                slate = tuple(
                    FirstCandidate(
                        candidate_id=str(row.corpus_id),
                        text=str(row.passage),
                        retrieval_rank=int(row.source_rank),
                    )
                    for row in group.itertuples()
                )
                if len(slate) < 2:
                    continue
                for variant in variants:
                    entries = build_prompt_entries(
                        slate, query_id=str(query_id), variant=variant, seed=args.seed
                    )
                    prompt, token_count, word_budget = fit_first_prompt(
                        tokenizer,
                        query,
                        entries,
                        context_size=4096,
                        initial_max_passage_words=128,
                    )
                    mapping = [
                        {
                            "candidate_id": entry.candidate.candidate_id,
                            "input_position": entry.input_position,
                            "identifier": entry.identifier,
                            "retrieval_rank": entry.candidate.retrieval_rank,
                        }
                        for entry in entries
                    ]
                    record = {
                        "schema": "first_prompt_input_v1",
                        "query_id": str(query_id),
                        "slate_id": f"{split}:{query_id}",
                        "variant": variant,
                        "prompt": prompt,
                        "first_token_prompt": prompt + "[",
                        "candidate_mapping": mapping,
                        "token_count": token_count,
                        "word_budget": word_budget,
                    }
                    record["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
                    record["fingerprint"] = stable_sha256(record)
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                if args.progress and number % 1000 == 0:
                    print(f"[prompts:{split}] {number:,}/{total:,}", flush=True)
        temporary.replace(prompt_path)
    report = {
        "schema": "mind_r8_5d_gate_package_v1",
        "source": "mteb/MindSmallReranking English derivative",
        "top_k": args.candidates,
        "variants": list(variants),
        "model": FIRST_MODEL,
        "revision": FIRST_MODEL_REVISION,
        "pools": {
            split: {
                "path": str(path),
                "sha256": sha256(path),
                "queries": int(pd.read_parquet(path, columns=["query_id"]).query_id.nunique()),
            }
            for split, path in outputs.items()
        },
        "prompts": {
            split: {"path": str(path), "sha256": sha256(path)}
            for split, path in prompt_files.items()
        },
        "large_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "report": str(args.report)}))


if __name__ == "__main__":
    main()
