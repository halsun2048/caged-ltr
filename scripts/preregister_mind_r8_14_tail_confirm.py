"""Preregister a fresh confirm for the frozen Tail-floor policy."""

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


def rank(query_id: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(
            f"mind-r8.14-tail-confirm:{seed}:{query_id}".encode(), digest_size=8
        ).digest(),
        "big",
    )


def ids(path: Path, column: str) -> set[str]:
    return set(pq.read_table(path, columns=[column]).column(column).to_pylist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/external/mind/mteb_english/queries/test-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--r7", type=Path, default=Path("data/processed/mind_r7_5/queries_selected.parquet")
    )
    parser.add_argument(
        "--r8", type=Path, default=Path("data/processed/mind_r8_0/large_split_ids.parquet")
    )
    parser.add_argument(
        "--old-gate", type=Path, default=Path("data/processed/mind_r8_5b/gate_split_ids.parquet")
    )
    parser.add_argument(
        "--fresh-gate",
        type=Path,
        default=Path("data/processed/mind_r8_8a/fresh_confirm_ids.parquet"),
    )
    parser.add_argument(
        "--gate-model", type=Path, default=Path("artifacts/mind_r8_6_gate_v2.joblib")
    )
    parser.add_argument(
        "--development", type=Path, default=Path("reports/experiments/mind_r8_11_tail_floor.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/mind_r8_14/fresh_confirm_ids.parquet")
    )
    parser.add_argument(
        "--guard", type=Path, default=Path("artifacts/mind_r8_14_confirm_guard.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/mind_r8_14_preregistration.json")
    )
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if any(path.exists() for path in (args.output, args.guard, args.report)):
        if all(path.exists() for path in (args.output, args.guard, args.report)):
            print(json.dumps({"stage": "cached", "report": str(args.report)}))
            return
        raise RuntimeError("partial R8.14 preregistration exists")
    development = json.loads(args.development.read_text())
    selected = development["selected"]
    if selected["budget"] != 0.4 or selected["tail_floor"] != 0.75:
        raise RuntimeError("development policy is not the frozen 40% / 75% Tail-floor policy")
    all_ids = pq.read_table(args.queries, columns=["id"]).column("id").to_pylist()
    excluded_sets = {
        "r7": ids(args.r7, "id"),
        "r8": ids(args.r8, "query_id"),
        "old_gate": ids(args.old_gate, "query_id"),
        "fresh_gate": ids(args.fresh_gate, "query_id"),
    }
    excluded = set().union(*excluded_sets.values())
    eligible = [value for value in all_ids if value not in excluded]
    ranks = np.fromiter((rank(value, args.seed) for value in eligible), np.uint64)
    positions = np.argpartition(ranks, args.count - 1)[: args.count]
    positions = positions[np.argsort(ranks[positions], kind="stable")]
    frame = pd.DataFrame(
        {
            "query_id": np.asarray(eligible, dtype=object)[positions],
            "split": "fresh_confirm_v2",
            "hash64": ranks[positions].astype(str),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    policy = {
        "model": "extra_trees_leaf10",
        "gate_model_sha256": sha256(args.gate_model),
        "budget": 0.4,
        "tail_floor": 0.75,
        "features": development["candidates"][0]["by_bucket"]
        and json.loads(Path("reports/experiments/mind_r8_6_gate_v2.json").read_text())["features"],
        "acceptance": {
            "overall_gap_at_most": 0.003,
            "tail_gap_at_most": 0.003,
            "first_call_rate_at_most": 0.55,
        },
    }
    guard = {
        "schema": "mind_r8_14_confirm_guard_v1",
        "status": "preregistered_unaccessed",
        "evaluation_count": 0,
        "allowed_evaluation_count": 1,
        "policy": policy,
    }
    args.guard.parent.mkdir(parents=True, exist_ok=True)
    args.guard.write_text(json.dumps(guard, indent=2) + "\n")
    report = {
        "schema": "mind_r8_14_preregistration_v1",
        "seed": args.seed,
        "count": len(frame),
        "eligible_reserve": len(eligible),
        "policy": policy,
        "query_ids_sha256": sha256(args.output),
        "source_queries_sha256": sha256(args.queries),
        "exclusions": {key: len(value) for key, value in excluded_sets.items()},
        "acceptance": {
            "exact_count": len(frame) == args.count,
            "unique": frame.query_id.nunique() == args.count,
            "disjoint": not bool(set(frame.query_id) & excluded),
            "labels_not_accessed": True,
            "large_test_not_accessed": True,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.progress:
        print(f"[1/3] eligible reserve={len(eligible):,}", flush=True)
        print(f"[2/3] froze {len(frame):,} fresh-confirm-v2 IDs", flush=True)
        print("[3/3] froze Tail-floor policy and guards; no labels accessed", flush=True)
    print(json.dumps({"stage": "complete", "report": str(args.report)}))


if __name__ == "__main__":
    main()
