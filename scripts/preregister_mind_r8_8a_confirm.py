"""Freeze a fresh confirmation split and the already-selected gain gate protocol."""

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


def hash_rank(query_id: str, seed: int) -> int:
    payload = f"mind-r8.8a-fresh-confirm:{seed}:{query_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def read_ids(path: Path, column: str) -> set[str]:
    return set(pq.read_table(path, columns=[column]).column(column).to_pylist())


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
        "--r8-ids", type=Path, default=Path("data/processed/mind_r8_0/large_split_ids.parquet")
    )
    parser.add_argument(
        "--old-gate-ids",
        type=Path,
        default=Path("data/processed/mind_r8_5b/gate_split_ids.parquet"),
    )
    parser.add_argument(
        "--gate-report", type=Path, default=Path("reports/experiments/mind_r8_6_gate_v2.json")
    )
    parser.add_argument(
        "--gate-model", type=Path, default=Path("artifacts/mind_r8_6_gate_v2.joblib")
    )
    parser.add_argument(
        "--gate-code", type=Path, default=Path("scripts/select_mind_r8_6_gate_v2.py")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mind_r8_8a/fresh_confirm_ids.parquet"),
    )
    parser.add_argument(
        "--guard", type=Path, default=Path("artifacts/mind_r8_8a_confirm_guard.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/mind_r8_8a_preregistration.json")
    )
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    outputs = (args.output, args.guard, args.report)
    if any(path.exists() for path in outputs):
        if all(path.exists() for path in outputs):
            print(json.dumps({"stage": "cached", "report": str(args.report)}))
            return
        raise RuntimeError("partial R8.8a preregistration exists")
    gate = json.loads(args.gate_report.read_text())
    selected = gate["selected"]
    features = gate["features"]
    forbidden = [value for value in features if value.startswith("first_") or "relevance" in value]
    if forbidden:
        raise RuntimeError(f"non-deployable pre-FIRST features: {forbidden}")
    all_ids = pq.read_table(args.queries, columns=["id"]).column("id").to_pylist()
    excluded_sets = {
        "r7": read_ids(args.r7_ids, "id"),
        "r8_train_dev_test": read_ids(args.r8_ids, "query_id"),
        "old_gate_dev_confirm": read_ids(args.old_gate_ids, "query_id"),
    }
    excluded = set().union(*excluded_sets.values())
    eligible = [value for value in all_ids if value not in excluded]
    if len(eligible) < args.count:
        raise RuntimeError("insufficient untouched reserve for fresh confirmation")
    if args.progress:
        print(f"[1/4] hashing {len(eligible):,} untouched reserve IDs", flush=True)
    ranks = np.fromiter((hash_rank(value, args.seed) for value in eligible), np.uint64)
    positions = np.argpartition(ranks, args.count - 1)[: args.count]
    positions = positions[np.argsort(ranks[positions], kind="stable")]
    frame = pd.DataFrame(
        {
            "query_id": np.asarray(eligible, dtype=object)[positions],
            "split": "fresh_confirm",
            "hash64": ranks[positions].astype(str),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    frozen_policy = {
        "model": selected["model"],
        "features": features,
        "threshold": selected["threshold"],
        "candidate_top_k": 20,
        "first_variants": ["baseline", "reverse", "random_permutation"],
        "acceptance": {
            "absolute_ndcg10_gap_vs_first_at_most": 0.003,
            "first_call_rate_at_most": 0.55,
            "gate_ndcg10_at_least_student": True,
            "tail_ndcg10_gap_vs_first_at_most": 0.003,
            "all_hash_and_boundary_checks": True,
        },
    }
    guard = {
        "schema": "mind_r8_8a_fresh_confirm_guard_v1",
        "status": "preregistered_unaccessed",
        "allowed_evaluation_count": 1,
        "evaluation_count": 0,
        "labels_materialized": False,
        "policy": frozen_policy,
    }
    args.guard.parent.mkdir(parents=True, exist_ok=True)
    args.guard.write_text(json.dumps(guard, indent=2) + "\n")
    report = {
        "schema": "mind_r8_8a_preregistration_v1",
        "seed": args.seed,
        "count": len(frame),
        "eligible_reserve": len(eligible),
        "frozen_policy": frozen_policy,
        "hashes": {
            "query_ids": sha256(args.output),
            "source_queries": sha256(args.queries),
            "gate_report": sha256(args.gate_report),
            "gate_model": sha256(args.gate_model),
            "gate_code": sha256(args.gate_code),
        },
        "exclusions": {name: len(values) for name, values in excluded_sets.items()},
        "acceptance": {
            "exact_count": len(frame) == args.count,
            "unique_ids": frame.query_id.nunique() == args.count,
            "no_prior_split_overlap": not bool(set(frame.query_id) & excluded),
            "features_available_before_first": not forbidden,
            "labels_not_accessed": True,
            "large_test_not_accessed": True,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.progress:
        print(f"[2/4] froze {len(frame):,} fresh-confirm query IDs", flush=True)
        print("[3/4] froze model, feature list, threshold, and acceptance gates", flush=True)
        print("[4/4] verified disjointness; labels and large-test remain untouched", flush=True)
    print(json.dumps({"stage": "complete", "report": str(args.report)}))


if __name__ == "__main__":
    main()
