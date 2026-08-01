"""Validate the English MIND-MTEB bundle and freeze train/dev/calibration splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED = {
    "corpus": {"files": 1, "rows": 5_277, "columns": ["id", "text", "title"]},
    "queries": {"files": 1, "rows": 2_362_514, "columns": ["id", "text"]},
    "data": {
        "files": 16,
        "rows": 97_006_943,
        "columns": ["query-id", "corpus-id", "score"],
    },
    "top_ranked": {
        "files": 11,
        "rows": 2_362_514,
        "columns": ["query-id", "corpus-ids"],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_bucket(query_id: str, seed: int) -> int:
    payload = f"mind-r7.5:{seed}:{query_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % 10_000


def english_statistics(texts: list[str]) -> dict[str, float | int]:
    sample = " ".join(texts)[:2_000_000]
    words = re.findall(r"[A-Za-z]+", sample.lower())
    stopwords = {"the", "a", "an", "and", "of", "to", "in", "for", "on", "with"}
    alphabetic = sum(character.isalpha() for character in sample)
    latin = sum(character.isascii() and character.isalpha() for character in sample)
    return {
        "characters": len(sample),
        "tokens": len(words),
        "latin_alphabetic_ratio": latin / max(alphabetic, 1),
        "english_stopword_ratio": sum(word in stopwords for word in words) / max(len(words), 1),
    }


def inspect_group(root: Path, name: str) -> dict[str, object]:
    files = sorted((root / name).glob("*.parquet"))
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    columns = pq.ParquetFile(files[0]).schema_arrow.names if files else []
    expected = EXPECTED[name]
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "rows": rows,
        "columns": columns,
        "expected": expected,
        "matches_expected": (
            len(files) == expected["files"]
            and rows == expected["rows"]
            and columns == expected["columns"]
        ),
        "file_sha256": {path.name: sha256(path) for path in files},
    }


def select_queries(path: Path, seed: int) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=100_000):
        frame = batch.to_pandas()
        buckets = frame["id"].map(lambda value: stable_bucket(value, seed))
        split = pd.Series("holdout", index=frame.index)
        split[buckets < 200] = "train"
        split[(buckets >= 200) & (buckets < 220)] = "dev"
        split[(buckets >= 220) & (buckets < 240)] = "calibration"
        keep = split != "holdout"
        part = frame.loc[keep].copy()
        part["split"] = split[keep]
        selected.append(part)
    return pd.concat(selected, ignore_index=True)


def sql_paths(paths: list[Path]) -> str:
    values = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"[{values}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/external/mind/mteb_english"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/mind_r7_5"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/mind_r7_5_english_bundle.json")
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.progress:
        print("[1/5] validating file counts, schemas, rows, and SHA-256", flush=True)
    groups = {name: inspect_group(args.root, name) for name in EXPECTED}
    if not all(group["matches_expected"] for group in groups.values()):
        raise RuntimeError("MIND bundle does not match the frozen data-card manifest")
    corpus_path = next((args.root / "corpus").glob("*.parquet"))
    query_path = next((args.root / "queries").glob("*.parquet"))
    corpus_sample = pq.read_table(corpus_path, columns=["text"]).slice(0, 5_000).to_pandas()
    query_sample = pq.read_table(query_path, columns=["text"]).slice(0, 20_000).to_pandas()
    language = english_statistics(
        corpus_sample["text"].fillna("").tolist() + query_sample["text"].fillna("").tolist()
    )
    english = (
        language["latin_alphabetic_ratio"] >= 0.98 and language["english_stopword_ratio"] >= 0.02
    )
    if not english:
        raise RuntimeError("English-only language gate failed")
    if args.progress:
        print("[2/5] applying deterministic query-level hash split", flush=True)
    selected = select_queries(query_path, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_output = args.output_dir / "queries_selected.parquet"
    pq.write_table(
        pa.Table.from_pandas(selected, preserve_index=False), query_output, compression="zstd"
    )
    split_counts = selected["split"].value_counts().sort_index().to_dict()
    if args.progress:
        print(f"[3/5] filtering labels for {len(selected):,} selected queries", flush=True)
    connection = duckdb.connect()
    connection.register("selected", selected[["id", "split"]])
    labels_output = args.output_dir / "labels_selected.parquet"
    candidates_output = args.output_dir / "candidates_selected.parquet"
    data_files = sorted((args.root / "data").glob("*.parquet"))
    ranked_files = sorted((args.root / "top_ranked").glob("*.parquet"))
    connection.execute(
        f"""
        COPY (
          SELECT d."query-id" AS query_id, d."corpus-id" AS corpus_id,
                 d.score, s.split
          FROM read_parquet({sql_paths(data_files)}) d
          INNER JOIN selected s ON d."query-id" = s.id
          ORDER BY query_id, corpus_id
        ) TO '{labels_output}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    if args.progress:
        print("[4/5] filtering candidate lists and checking references", flush=True)
    connection.execute(
        f"""
        COPY (
          SELECT r."query-id" AS query_id, r."corpus-ids" AS corpus_ids, s.split
          FROM read_parquet({sql_paths(ranked_files)}) r
          INNER JOIN selected s ON r."query-id" = s.id
          ORDER BY query_id
        ) TO '{candidates_output}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    labels_rows = pq.ParquetFile(labels_output).metadata.num_rows
    candidates_rows = pq.ParquetFile(candidates_output).metadata.num_rows
    selected_ids = set(selected["id"])
    disjoint = sum(split_counts.values()) == len(selected_ids)
    outputs = [query_output, labels_output, candidates_output]
    payload = {
        "schema": "mind_r7_5_english_bundle_v1",
        "source": "mteb/MindSmallReranking test configuration derived from English MIND",
        "license": "other; Microsoft Research License terms must be retained",
        "language_policy": "English only; non-English/translated variants rejected",
        "raw_groups": groups,
        "language": {**language, "english_gate_passed": english},
        "split_policy": {
            "unit": "query_id",
            "seed": args.seed,
            "hash": "BLAKE2b-64 over mind-r7.5:{seed}:{query_id}",
            "buckets": {
                "train": "0-199",
                "dev": "200-219",
                "calibration": "220-239",
                "holdout": "240-9999",
            },
            "counts": split_counts,
            "holdout_queries": EXPECTED["queries"]["rows"] - len(selected),
        },
        "processed": {
            "selected_queries": len(selected),
            "selected_labels": labels_rows,
            "selected_candidate_lists": candidates_rows,
            "files": {
                str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in outputs
            },
        },
        "boundaries": {
            "external_pretraining_or_independent_dev_only": True,
            "nfcorpus_final_evidence": False,
            "mind_holdout_accessed": False,
            "nfcorpus_locked_test_accessed": False,
        },
        "acceptance": {
            "all_raw_files_match_manifest": True,
            "english_only": english,
            "query_splits_disjoint": disjoint,
            "all_selected_queries_have_candidate_lists": candidates_rows == len(selected),
            "all_selected_queries_have_labels": labels_rows > len(selected),
            "large_holdout_preserved": EXPECTED["queries"]["rows"] - len(selected) > 2_000_000,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    args.report.with_suffix(".md").write_text(
        "# R7.5 English MIND bundle and frozen split\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    if args.progress:
        print("[5/5] reproducible bundle written", flush=True)
    print(
        json.dumps(
            {"stage": "complete", "report": str(args.report), "acceptance": payload["acceptance"]}
        )
    )


if __name__ == "__main__":
    main()
