"""Build sharded English MIND large-train/dev packages without materializing test rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import duckdb


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sql_paths(paths: list[Path]) -> str:
    values = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"[{values}]"


def english_gate(text: str) -> dict[str, float | int | bool]:
    words = re.findall(r"[A-Za-z]+", text.lower())
    stopwords = {"the", "a", "an", "and", "of", "to", "in", "for", "on", "with"}
    alphabetic = sum(character.isalpha() for character in text)
    latin = sum(character.isascii() and character.isalpha() for character in text)
    latin_ratio = latin / max(alphabetic, 1)
    stopword_ratio = sum(word in stopwords for word in words) / max(len(words), 1)
    return {
        "characters": len(text),
        "tokens": len(words),
        "latin_alphabetic_ratio": latin_ratio,
        "english_stopword_ratio": stopword_ratio,
        "passed": latin_ratio >= 0.98 and stopword_ratio >= 0.02,
    }


def copy_query(connection, query: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(f"COPY ({query}) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/external/mind/mteb_english"))
    parser.add_argument(
        "--split-ids",
        type=Path,
        default=Path("data/processed/mind_r8_0/large_split_ids.parquet"),
    )
    parser.add_argument(
        "--test-guard",
        type=Path,
        default=Path("artifacts/mind_r8_0_large_test_guard.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/mind_r8_1"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/mind_r8_1_large_package.json")
    )
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=20260801)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    guard = json.loads(args.test_guard.read_text())
    if guard["status"] != "locked_unaccessed" or guard["evaluation_count"] != 0:
        raise RuntimeError("large-test guard is not in the required locked state")
    if args.report.exists() and not args.resume:
        raise RuntimeError("R8.1 report exists; use --resume instead of overwriting")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work = args.output_dir / "work"
    work.mkdir(exist_ok=True)
    queries_file = next((args.raw_root / "queries").glob("*.parquet"))
    corpus_file = next((args.raw_root / "corpus").glob("*.parquet"))
    label_files = sorted((args.raw_root / "data").glob("*.parquet"))
    candidate_files = sorted((args.raw_root / "top_ranked").glob("*.parquet"))
    connection = duckdb.connect()
    connection.execute("SET memory_limit='12GB'")
    connection.execute("SET threads=8")
    selected_queries = work / "selected_queries.parquet"
    selected_labels = work / "selected_labels.parquet"
    selected_candidates = work / "selected_candidates.parquet"
    steps = [
        (
            selected_queries,
            f"""
            SELECT q.id AS query_id, q.text AS query, s.split
            FROM read_parquet('{queries_file}') q
            INNER JOIN read_parquet('{args.split_ids}') s ON q.id = s.query_id
            WHERE s.split IN ('large_train', 'large_dev')
            ORDER BY query_id
            """,
            "query text",
        ),
        (
            selected_labels,
            f"""
            SELECT d."query-id" AS query_id, d."corpus-id" AS corpus_id,
                   d.score, s.split
            FROM read_parquet({sql_paths(label_files)}) d
            INNER JOIN read_parquet('{args.split_ids}') s ON d."query-id" = s.query_id
            WHERE s.split IN ('large_train', 'large_dev')
            ORDER BY query_id, corpus_id
            """,
            "relevance labels",
        ),
        (
            selected_candidates,
            f"""
            SELECT r."query-id" AS query_id, r."corpus-ids" AS corpus_ids, s.split
            FROM read_parquet({sql_paths(candidate_files)}) r
            INNER JOIN read_parquet('{args.split_ids}') s ON r."query-id" = s.query_id
            WHERE s.split IN ('large_train', 'large_dev')
            ORDER BY query_id
            """,
            "candidate lists",
        ),
    ]
    for index, (output, query, label) in enumerate(steps, 1):
        if output.exists() and args.resume:
            if args.progress:
                print(f"[{index}/6] cached {label}", flush=True)
            continue
        if args.progress:
            print(f"[{index}/6] materializing large-train/dev {label}", flush=True)
        copy_query(connection, query, output)
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW query_text AS
          SELECT * FROM read_parquet('{selected_queries}');
        CREATE OR REPLACE TEMP VIEW labels AS
          SELECT * FROM read_parquet('{selected_labels}');
        CREATE OR REPLACE TEMP VIEW candidate_lists AS
          SELECT * FROM read_parquet('{selected_candidates}');
        CREATE OR REPLACE TEMP VIEW corpus AS
          SELECT id AS corpus_id, text AS passage FROM read_parquet('{corpus_file}');
        CREATE OR REPLACE TEMP TABLE item_frequency AS
          SELECT corpus_id, count(DISTINCT query_id)::BIGINT AS train_item_frequency
          FROM labels WHERE split = 'large_train' AND score > 0 GROUP BY corpus_id;
        """
    )
    enriched = work / "enriched_rows.parquet"
    if not (enriched.exists() and args.resume):
        if args.progress:
            print(
                "[4/6] joining ranks, labels, texts, frequencies, and uncertainty fields",
                flush=True,
            )
        copy_query(
            connection,
            """
            WITH ranked_raw AS (
              SELECT c.query_id, c.split,
                     list_extract(c.corpus_ids, u.source_rank) AS corpus_id,
                     u.source_rank
              FROM candidate_lists c,
                   UNNEST(range(1, len(c.corpus_ids) + 1)) AS u(source_rank)
            ), ranked AS (
              SELECT query_id, split, corpus_id, min(source_rank) AS source_rank
              FROM ranked_raw GROUP BY query_id, split, corpus_id
            ), joined AS (
              SELECT r.query_id, r.split, q.query, r.corpus_id, p.passage,
                     l.score::INTEGER AS relevance, r.source_rank::INTEGER AS source_rank,
                     count(*) OVER (PARTITION BY r.query_id)::INTEGER AS candidate_count,
                     sum((l.score > 0)::INTEGER) OVER (PARTITION BY r.query_id)::INTEGER
                       AS positive_count,
                     coalesce(f.train_item_frequency, 0)::BIGINT AS train_item_frequency
              FROM ranked r
              INNER JOIN labels l USING (query_id, corpus_id, split)
              INNER JOIN query_text q USING (query_id, split)
              INNER JOIN corpus p USING (corpus_id)
              LEFT JOIN item_frequency f USING (corpus_id)
            )
            SELECT *, positive_count::DOUBLE / candidate_count AS positive_ratio,
                   length(query)::INTEGER AS query_characters,
                   length(passage)::INTEGER AS passage_characters,
                   (candidate_count - positive_count)::INTEGER AS negative_count
            FROM joined ORDER BY query_id, source_rank
            """,
            enriched,
        )
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW enriched AS SELECT * FROM read_parquet('{enriched}')"
    )
    pairs = work / "train_pairs.parquet"
    if not (pairs.exists() and args.resume):
        if args.progress:
            print("[5/6] constructing multi-positive rank-hard and random pairs", flush=True)
        copy_query(
            connection,
            f"""
            WITH positives AS (
              SELECT * FROM enriched WHERE split = 'large_train' AND relevance > 0
            ), hard_negative AS (
              SELECT query_id, arg_min(corpus_id, source_rank) AS negative_id,
                     min(source_rank) AS negative_rank
              FROM enriched WHERE split = 'large_train' AND relevance = 0 GROUP BY query_id
            ), random_negative AS (
              SELECT query_id,
                     arg_min(corpus_id, hash(query_id || ':' || corpus_id || ':{args.random_seed}'))
                       AS negative_id,
                     arg_min(
                       source_rank,
                       hash(query_id || ':' || corpus_id || ':{args.random_seed}')
                     ) AS negative_rank
              FROM enriched WHERE split = 'large_train' AND relevance = 0 GROUP BY query_id
            ), negatives AS (
              SELECT query_id, negative_id, negative_rank, 'rank_hard' AS negative_type
              FROM hard_negative
              UNION ALL
              SELECT query_id, negative_id, negative_rank, 'random' AS negative_type
              FROM random_negative
            )
            SELECT p.query_id, p.query, p.corpus_id AS positive_id,
                   p.passage AS positive_passage, p.source_rank AS positive_rank,
                   n.negative_id, c.passage AS negative_passage, n.negative_rank,
                   n.negative_type, p.train_item_frequency AS positive_item_frequency,
                   p.candidate_count, p.positive_count, p.negative_count, p.positive_ratio,
                   p.query_characters, p.passage_characters AS positive_characters,
                   length(c.passage)::INTEGER AS negative_characters
            FROM positives p
            INNER JOIN negatives n USING (query_id)
            INNER JOIN corpus c ON n.negative_id = c.corpus_id
            ORDER BY p.query_id, p.corpus_id, n.negative_type
            """,
            pairs,
        )
    if args.progress:
        print(f"[6/6] writing {args.shards} deterministic shards per package", flush=True)
    package_specs = {
        "train_pairs": (pairs, "TRUE"),
        "train_listwise": (enriched, "split = 'large_train'"),
        "dev_listwise": (enriched, "split = 'large_dev'"),
    }
    output_files: dict[str, list[Path]] = {}
    for package, (source, condition) in package_specs.items():
        package_dir = args.output_dir / package
        package_dir.mkdir(exist_ok=True)
        output_files[package] = []
        for shard in range(args.shards):
            output = package_dir / f"part-{shard:05d}-of-{args.shards:05d}.parquet"
            output_files[package].append(output)
            if output.exists() and args.resume:
                continue
            copy_query(
                connection,
                f"""
                SELECT * FROM read_parquet('{source}')
                WHERE {condition} AND hash(query_id) % {args.shards} = {shard}
                ORDER BY query_id
                """,
                output,
            )
    split_counts = dict(
        connection.execute(
            f"SELECT split, count(*) FROM read_parquet('{selected_queries}') GROUP BY split"
        ).fetchall()
    )
    label_rows = connection.execute(
        f"SELECT count(*) FROM read_parquet('{selected_labels}')"
    ).fetchone()[0]
    candidate_elements = connection.execute(
        f"SELECT sum(len(corpus_ids)) FROM read_parquet('{selected_candidates}')"
    ).fetchone()[0]
    unique_candidate_pairs = connection.execute(
        f"""
        SELECT count(*) FROM (
          SELECT DISTINCT c.query_id, u.corpus_id
          FROM read_parquet('{selected_candidates}') c,
               UNNEST(c.corpus_ids) AS u(corpus_id)
        )
        """
    ).fetchone()[0]
    enriched_rows = connection.execute(
        f"SELECT count(*) FROM read_parquet('{enriched}')"
    ).fetchone()[0]
    pair_rows = connection.execute(f"SELECT count(*) FROM read_parquet('{pairs}')").fetchone()[0]
    positive_rows = connection.execute(
        f"""
        SELECT count(*) FROM read_parquet('{enriched}')
        WHERE relevance > 0 AND split = 'large_train'
        """
    ).fetchone()[0]
    missing_test_rows = {}
    for package, files in output_files.items():
        missing_test_rows[package] = connection.execute(
            f"""
            SELECT count(*) FROM read_parquet({sql_paths(files)}) o
            INNER JOIN read_parquet('{args.split_ids}') s USING (query_id)
            WHERE s.split = 'large_test'
            """
        ).fetchone()[0]
    sample = connection.execute(
        f"SELECT query || ' ' || passage FROM read_parquet('{enriched}') LIMIT 20000"
    ).fetchall()
    language = english_gate(" ".join(row[0] for row in sample))
    package_files = {
        package: [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ]
        for package, files in output_files.items()
    }
    payload = {
        "schema": "mind_r8_1_large_package_v1",
        "source": "mteb/MindSmallReranking English derivative",
        "split_counts": split_counts,
        "counts": {
            "selected_label_rows": label_rows,
            "candidate_elements": candidate_elements,
            "unique_candidate_pairs": unique_candidate_pairs,
            "duplicate_candidate_occurrences_removed": candidate_elements
            - unique_candidate_pairs,
            "enriched_rows": enriched_rows,
            "large_train_positive_rows": positive_rows,
            "train_pair_rows": pair_rows,
            "expected_pair_rows": positive_rows * 2,
        },
        "features": [
            "rank-hard negative",
            "deterministic random negative",
            "all positive documents",
            "listwise candidate group",
            "train-only item frequency",
            "candidate/positive counts and ratio",
            "query/passage character lengths",
        ],
        "language": language,
        "packages": package_files,
        "test_boundary": {
            "guard_sha256": sha256(args.test_guard),
            "raw_mixed_parquet_physically_scanned": True,
            "large_test_rows_emitted_or_aggregated": False,
            "large_test_labels_or_candidates_materialized": False,
            "large_test_metric_computed": False,
            "output_test_row_counts": missing_test_rows,
        },
        "acceptance": {
            "exact_train_dev_query_counts": split_counts
            == {"large_train": 200_000, "large_dev": 20_000},
            "candidate_label_alignment_complete_after_deduplication": label_rows
            == unique_candidate_pairs
            == enriched_rows,
            "two_pairs_per_positive": pair_rows == positive_rows * 2,
            "english_gate_passed": language["passed"],
            "sixteen_shards_per_package": all(
                len(files) == args.shards for files in output_files.values()
            ),
            "no_large_test_rows_in_outputs": all(
                value == 0 for value in missing_test_rows.values()
            ),
            "large_test_remains_locked": True,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    args.report.with_suffix(".md").write_text(
        "# R8.1 MIND large-scale package\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    print(
        json.dumps(
            {"stage": "complete", "report": str(args.report), "acceptance": payload["acceptance"]}
        )
    )


if __name__ == "__main__":
    main()
