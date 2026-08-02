"""Freeze a 2,000-query train-only FIRST package for the R14.4 pilot."""

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
    parser.add_argument("--split-ids", type=Path, default=Path("data/processed/mind_r8_0/large_split_ids.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/mind_r14_4"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mind_r14_4"))
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.output_dir / "pilot_query_ids.parquet"
    pool_path = args.output_dir / "pilot_pool.parquet"
    prompts_path = args.run_dir / "pilot_prompts.jsonl"
    query_file = next((args.raw_root / "queries").glob("*.parquet"))
    corpus_file = next((args.raw_root / "corpus").glob("*.parquet"))
    labels = sorted((args.raw_root / "data").glob("*.parquet"))
    candidates = sorted((args.raw_root / "top_ranked").glob("*.parquet"))
    if not labels or not candidates:
        raise FileNotFoundError("MIND labels/candidate shards are missing")
    connection = duckdb.connect()
    connection.execute("SET memory_limit='12GB'")
    connection.execute("SET threads=8")
    connection.execute(
        f"""
        COPY (
          SELECT query_id, hash64, 'r14_4_pilot_train' AS split
          FROM read_parquet('{args.split_ids}')
          WHERE split = 'large_train'
          ORDER BY hash64, query_id LIMIT {args.queries}
        ) TO '{ids_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    if args.progress:
        print(f"[1/3] froze {args.queries:,} large_train query IDs", flush=True)
    connection.execute(
        f"""
        COPY (
          WITH selected AS (SELECT query_id FROM read_parquet('{ids_path}')),
          q AS (SELECT id AS query_id, text AS query FROM read_parquet('{query_file}')),
          l AS (
            SELECT "query-id" AS query_id, "corpus-id" AS corpus_id, score AS relevance
            FROM read_parquet({sql_paths(labels)})
          ), chosen AS (
            SELECT r."query-id" AS query_id, r."corpus-ids" AS corpus_ids
            FROM read_parquet({sql_paths(candidates)}) r INNER JOIN selected s ON r."query-id"=s.query_id
          ), raw_rank AS (
            SELECT c.query_id, list_extract(c.corpus_ids,u.rank) AS corpus_id, u.rank AS source_rank
            FROM chosen c, UNNEST(range(1,least(len(c.corpus_ids),{args.candidates})+1)) u(rank)
          ), ranked AS (
            SELECT query_id,corpus_id,min(source_rank)::INTEGER AS source_rank
            FROM raw_rank GROUP BY query_id,corpus_id
          ), corpus AS (SELECT id AS corpus_id,text AS passage FROM read_parquet('{corpus_file}'))
          SELECT r.query_id,q.query,r.corpus_id,c.passage,
                 coalesce(l.relevance,0)::INTEGER AS relevance,r.source_rank,
                 'r14_4_pilot_train' AS split
          FROM ranked r INNER JOIN q USING(query_id) INNER JOIN corpus c USING(corpus_id)
          LEFT JOIN l USING(query_id,corpus_id)
          ORDER BY r.query_id,r.source_rank
        ) TO '{pool_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    pool = pd.read_parquet(pool_path)
    if pool.query_id.nunique() != args.queries:
        raise RuntimeError("pilot pool query count mismatch")
    if args.progress:
        print(f"[2/3] materialized {len(pool):,} top-{args.candidates} rows", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(FIRST_MODEL, revision=FIRST_MODEL_REVISION, token=False)
    with prompts_path.open("w") as handle:
        for number, (query_id, group) in enumerate(pool.groupby("query_id", sort=False), 1):
            query = str(group.iloc[0].query)
            slate = tuple(
                FirstCandidate(str(row.corpus_id), str(row.passage), int(row.source_rank))
                for row in group.itertuples()
            )
            entries = build_prompt_entries(slate, query_id=str(query_id), variant="baseline", seed=args.seed)
            prompt, token_count, word_budget = fit_first_prompt(tokenizer, query, entries, context_size=4096, initial_max_passage_words=128)
            mapping = [
                {"candidate_id": e.candidate.candidate_id, "input_position": e.input_position, "identifier": e.identifier, "retrieval_rank": e.candidate.retrieval_rank}
                for e in entries
            ]
            record = {
                "schema": "first_prompt_input_v1", "query_id": str(query_id),
                "slate_id": f"r14_4_train:{query_id}", "variant": "baseline",
                "prompt": prompt, "first_token_prompt": prompt + "[",
                "candidate_mapping": mapping, "token_count": token_count, "word_budget": word_budget,
            }
            record["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            record["fingerprint"] = stable_sha256(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if args.progress and number % 200 == 0:
                print(f"[prompts] {number:,}/{args.queries:,}", flush=True)
    report = {
        "schema": "mind_r14_4_pilot_package_v1", "source_split": "large_train_only",
        "query_count": args.queries, "candidate_count": args.candidates, "seed": args.seed,
        "ids": {"path": str(ids_path), "sha256": sha256(ids_path)},
        "pool": {"path": str(pool_path), "sha256": sha256(pool_path)},
        "prompts": {"path": str(prompts_path), "sha256": sha256(prompts_path)},
        "model": FIRST_MODEL, "revision": FIRST_MODEL_REVISION,
        "large_dev_accessed": False, "confirm_accessed": False, "large_test_accessed": False,
    }
    out = Path("reports/data/mind_r14_4_pilot_package.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "report": str(out), "prompts": str(prompts_path)}))


if __name__ == "__main__":
    main()
