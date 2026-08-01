"""Preregister independent gate-dev and gate-confirm IDs from untouched MIND reserve."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def rank(query_id: str, seed: int) -> int:
    payload = f"mind-r8.5b-gate:{seed}:{query_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/external/mind/mteb_english/queries/test-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--r7-ids", type=Path, default=Path("data/processed/mind_r7_5/queries_selected.parquet")
    )
    parser.add_argument(
        "--r8-ids",
        type=Path,
        default=Path("data/processed/mind_r8_0/large_split_ids.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/mind_r8_5b/gate_split_ids.parquet")
    )
    parser.add_argument(
        "--guard", type=Path, default=Path("artifacts/mind_r8_5b_gate_guard.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/mind_r8_5b_gate_preregistration.json")
    )
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    outputs = (args.output, args.guard, args.report)
    if any(path.exists() for path in outputs):
        if all(path.exists() for path in outputs):
            print(json.dumps({"stage": "cached", "report": str(args.report)}))
            return
        raise RuntimeError("partial R8.5b preregistration exists")
    ids = pq.read_table(args.queries, columns=["id"]).column("id").to_pylist()
    r7 = set(pq.read_table(args.r7_ids, columns=["id"]).column("id").to_pylist())
    r8 = set(pq.read_table(args.r8_ids, columns=["query_id"]).column("query_id").to_pylist())
    eligible = [value for value in ids if value not in r7 and value not in r8]
    required = 2 * args.count
    if len(eligible) < required:
        raise RuntimeError("insufficient untouched reserve")
    if args.progress:
        print(f"[1/3] hashing {len(eligible):,} untouched reserve IDs", flush=True)
    ranks = np.fromiter((rank(value, args.seed) for value in eligible), np.uint64)
    positions = np.argpartition(ranks, required - 1)[:required]
    positions = positions[np.argsort(ranks[positions], kind="stable")]
    selected = np.asarray(eligible, dtype=object)[positions]
    frame = pd.DataFrame(
        {
            "query_id": selected,
            "split": ["gate_dev"] * args.count + ["gate_confirm"] * args.count,
            "hash64": ranks[positions].astype(str),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    guard = {
        "schema": "mind_r8_5b_gate_guard_v1",
        "gate_dev_access_count": 0,
        "gate_confirm_access_count": 0,
        "gate_confirm_allowed_access_count": 1,
        "status": "preregistered",
    }
    args.guard.parent.mkdir(parents=True, exist_ok=True)
    args.guard.write_text(json.dumps(guard, indent=2) + "\n")
    report = {
        "schema": "mind_r8_5b_gate_preregistration_v1",
        "seed": args.seed,
        "counts": frame.split.value_counts().to_dict(),
        "eligible_reserve": len(eligible),
        "files": {
            "split_ids": {"path": str(args.output), "sha256": digest(args.output)},
            "queries": {"path": str(args.queries), "sha256": digest(args.queries)},
        },
        "acceptance": {
            "exact_counts": len(frame) == required,
            "disjoint": frame.query_id.nunique() == required,
            "no_r7_overlap": not bool(set(frame.query_id) & r7),
            "no_r8_overlap": not bool(set(frame.query_id) & r8),
            "large_test_accessed": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.progress:
        print("[2/3] froze 20k gate-dev + 20k gate-confirm", flush=True)
        print("[3/3] guards and hashes written; no labels accessed", flush=True)
    print(json.dumps({"stage": "complete", "report": str(args.report)}))


if __name__ == "__main__":
    main()
