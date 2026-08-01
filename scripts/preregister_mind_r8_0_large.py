"""Freeze exact large-scale MIND query splits before accessing their labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash64(query_id: str, seed: int, namespace: str) -> int:
    payload = f"{namespace}:{seed}:{query_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/external/mind/mteb_english/queries/test-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--prior-selected",
        type=Path,
        default=Path("data/processed/mind_r7_5/queries_selected.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/mind_r8_0/large_split_ids.parquet")
    )
    parser.add_argument(
        "--test-guard", type=Path, default=Path("artifacts/mind_r8_0_large_test_guard.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/mind_r8_0_preregistration.json")
    )
    parser.add_argument("--train-queries", type=int, default=200_000)
    parser.add_argument("--dev-queries", type=int, default=20_000)
    parser.add_argument("--test-queries", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.test_guard.exists() or args.report.exists() or args.output.exists():
        raise RuntimeError("R8.0 preregistration already exists; refusing to silently replace it")
    if args.progress:
        print("[1/4] reading query IDs only; no candidate labels are accessed", flush=True)
    query_ids = pq.read_table(args.queries, columns=["id"]).column("id").to_pylist()
    prior_ids = set(pq.read_table(args.prior_selected, columns=["id"]).column("id").to_pylist())
    eligible = [value for value in query_ids if value not in prior_ids]
    required = args.train_queries + args.dev_queries + args.test_queries
    if len(eligible) < required:
        raise RuntimeError("not enough untouched query IDs")
    if args.progress:
        print(f"[2/4] hashing {len(eligible):,} eligible IDs with frozen seed", flush=True)
    hashes = np.fromiter(
        (hash64(value, args.seed, "mind-r8.0-large") for value in eligible),
        dtype=np.uint64,
        count=len(eligible),
    )
    selected_positions = np.argpartition(hashes, required - 1)[:required]
    selected_positions = selected_positions[np.argsort(hashes[selected_positions], kind="stable")]
    selected_ids = np.asarray(eligible, dtype=object)[selected_positions]
    selected_hashes = hashes[selected_positions]
    splits = np.empty(required, dtype=object)
    train_end = args.train_queries
    dev_end = train_end + args.dev_queries
    splits[:train_end] = "large_train"
    splits[train_end:dev_end] = "large_dev"
    splits[dev_end:] = "large_test"
    frame = pd.DataFrame(
        {"query_id": selected_ids, "split": splits, "hash64": selected_hashes.astype(str)}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    output_hash = sha256(args.output)
    test_ids = frame.loc[frame.split == "large_test", "query_id"]
    test_id_hash = hashlib.sha256("\n".join(test_ids).encode()).hexdigest()
    guard = {
        "schema": "mind_r8_0_large_test_guard_v1",
        "status": "locked_unaccessed",
        "query_count": args.test_queries,
        "query_id_sha256": test_id_hash,
        "allowed_evaluation_count": 1,
        "evaluation_count": 0,
        "labels_materialized": False,
        "predictions_exist": False,
        "policy": (
            "Only an explicit final R8 admission command may materialize labels and score "
            "this split."
        ),
    }
    args.test_guard.parent.mkdir(parents=True, exist_ok=True)
    args.test_guard.write_text(json.dumps(guard, ensure_ascii=False, indent=2) + "\n")
    counts = frame.split.value_counts().sort_index().to_dict()
    payload = {
        "schema": "mind_r8_0_large_preregistration_v1",
        "source": "mteb/MindSmallReranking English derivative",
        "seed": args.seed,
        "selection": (
            "Exact lowest BLAKE2b-64 ranks over IDs not selected by R7.5; fixed contiguous "
            "rank intervals allocate train/dev/test."
        ),
        "counts": counts,
        "eligible_queries": len(eligible),
        "continuing_reserve_queries": len(eligible) - required,
        "files": {
            "raw_query_ids": {"path": str(args.queries), "sha256": sha256(args.queries)},
            "prior_selected": {
                "path": str(args.prior_selected),
                "sha256": sha256(args.prior_selected),
            },
            "large_split_ids": {"path": str(args.output), "sha256": output_hash},
            "large_test_guard": {
                "path": str(args.test_guard),
                "sha256": sha256(args.test_guard),
            },
        },
        "boundary_disclosure": {
            "query_text_was_mechanically_streamed_during_r7_5_split_construction": True,
            "r8_0_reads_only_query_id_column": True,
            "large_candidate_labels_accessed_before_freeze": False,
            "large_test_candidates_or_labels_materialized": False,
            "large_test_metric_computed": False,
        },
        "acceptance": {
            "exact_requested_counts": counts
            == {
                "large_dev": args.dev_queries,
                "large_test": args.test_queries,
                "large_train": args.train_queries,
            },
            "no_overlap_with_r7_5": not bool(set(frame.query_id) & prior_ids),
            "splits_disjoint": len(frame.query_id) == frame.query_id.nunique(),
            "large_test_locked": True,
            "labels_not_accessed": True,
            "reserve_exceeds_two_million": len(eligible) - required > 2_000_000,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    args.report.with_suffix(".md").write_text(
        "# R8.0 MIND large-scale preregistration\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    if args.progress:
        print("[3/4] exact train/dev/test IDs written", flush=True)
        print("[4/4] large-test guard locked before label materialization", flush=True)
    print(
        json.dumps(
            {"stage": "complete", "report": str(args.report), "acceptance": payload["acceptance"]}
        )
    )


if __name__ == "__main__":
    main()
