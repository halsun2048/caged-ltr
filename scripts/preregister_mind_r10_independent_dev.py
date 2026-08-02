#!/usr/bin/env python3
"""Freeze a new R10 dev/confirm split without materializing labels or touching test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "data/external/mind/mteb_english/data/*.parquet"
OUT = ROOT / "data/processed/mind_r10_0"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=10000)
    ap.add_argument("--confirm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--progress", action="store_true", help="print phase progress")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    print("[R10 1/4] scanning query universe")
    con.execute(f"CREATE OR REPLACE TEMP TABLE all_q AS SELECT DISTINCT \"query-id\" AS query_id FROM read_parquet('{REL}')")
    con.execute("CREATE OR REPLACE TEMP TABLE excluded AS SELECT query_id FROM read_parquet(?) UNION SELECT query_id FROM read_parquet(?)", [str(ROOT/'data/processed/mind_r8_0/large_split_ids.parquet'), str(ROOT/'data/processed/mind_r8_14/fresh_confirm_ids.parquet')])
    print("[R10 2/4] deterministic exclusion and hash split")
    con.execute("CREATE OR REPLACE TEMP TABLE chosen AS SELECT query_id, CASE WHEN row_number() OVER (ORDER BY hash(md5(query_id || ?)), query_id) <= ? THEN 'dev' ELSE 'confirm' END AS split FROM (SELECT a.query_id FROM all_q a LEFT JOIN excluded e USING(query_id) WHERE e.query_id IS NULL) t LIMIT ?", [str(args.seed), args.dev, args.dev + args.confirm])
    con.execute(f"COPY chosen TO '{OUT/'query_ids.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print("[R10 3/4] writing guard and manifest")
    qhash = sha(OUT / "query_ids.parquet")
    counts = con.execute("SELECT split, count(*) FROM chosen GROUP BY split ORDER BY split").fetchall()
    guard = {"schema": "mind_r10_0_independent_guard_v1", "status": "preregistered_locked", "labels_materialized": False, "evaluation_count": 0, "allowed_evaluation_count": 1, "query_count": args.dev + args.confirm, "query_id_sha256": qhash, "seed": args.seed, "source": "MTEB MIND-derived relation; excludes R8.0 large and R8.14 confirm", "policy": "dev may be used for routing selection; confirm is one-time fixed confirmation; no large-test access"}
    (ROOT/'artifacts/mind_r10_0_independent_guard.json').write_text(json.dumps(guard, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {"schema": "mind_r10_0_split_manifest_v1", "counts": dict(counts), "query_id_sha256": qhash, "seed": args.seed, "excluded": ["mind_r8_0_large_split_ids", "mind_r8_14_fresh_confirm_ids"], "labels_read": False}
    (ROOT/'reports/data/mind_r10_0_split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({"stage": "complete", "counts": dict(counts), "guard": str(ROOT/'artifacts/mind_r10_0_independent_guard.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
