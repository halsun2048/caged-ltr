#!/usr/bin/env python3
"""Freeze an R11 dev/confirm split excluding every prior guarded split."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "data/external/mind/mteb_english/data/*.parquet"
OUT = ROOT / "data/processed/mind_r11_0"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=10000)
    ap.add_argument("--confirm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(); con.execute("PRAGMA threads=4")
    if args.progress: print("[R11 1/3] scanning query universe", flush=True)
    con.execute(f"CREATE OR REPLACE TEMP TABLE all_q AS SELECT DISTINCT \"query-id\" AS query_id FROM read_parquet('{REL}')")
    excluded = [ROOT/'data/processed/mind_r8_0/large_split_ids.parquet', ROOT/'data/processed/mind_r8_14/fresh_confirm_ids.parquet', ROOT/'data/processed/mind_r10_0/query_ids.parquet']
    con.execute("CREATE OR REPLACE TEMP TABLE excluded AS SELECT query_id FROM read_parquet(?) UNION SELECT query_id FROM read_parquet(?) UNION SELECT query_id FROM read_parquet(?)", [str(p) for p in excluded])
    if args.progress: print("[R11 2/3] deterministic split and overlap guard", flush=True)
    total = args.dev + args.confirm
    con.execute("CREATE OR REPLACE TEMP TABLE chosen AS SELECT query_id, CASE WHEN row_number() OVER (ORDER BY hash(md5(query_id || ?)), query_id) <= ? THEN 'dev' ELSE 'confirm' END AS split FROM (SELECT a.query_id FROM all_q a LEFT JOIN excluded e USING(query_id) WHERE e.query_id IS NULL) t LIMIT ?", [str(args.seed), args.dev, total])
    con.execute(f"COPY chosen TO '{OUT/'query_ids.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    qhash = sha(OUT/'query_ids.parquet')
    counts = con.execute("SELECT split, count(*) FROM chosen GROUP BY split ORDER BY split").fetchall()
    guard = {"schema":"mind_r11_0_independent_guard_v1","status":"preregistered_locked","labels_materialized":False,"evaluation_count":0,"allowed_evaluation_count":1,"query_count":total,"query_id_sha256":qhash,"seed":args.seed,"excluded_splits":["mind_r8_0_large","mind_r8_14_confirm","mind_r10_0_dev_confirm"],"policy":"select on dev only; confirm one-time fixed evaluation; no large-test access"}
    (ROOT/'artifacts/mind_r11_0_independent_guard.json').write_text(json.dumps(guard,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest = {"schema":"mind_r11_0_split_manifest_v1","counts":dict(counts),"query_id_sha256":qhash,"seed":args.seed,"excluded": [str(p) for p in excluded],"labels_read":False}
    (ROOT/'reports/data/mind_r11_0_split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.progress: print("[R11 3/3] guard locked", flush=True)
    print(json.dumps({"stage":"complete","counts":dict(counts),"guard":str(ROOT/'artifacts/mind_r11_0_independent_guard.json')},ensure_ascii=False))

if __name__ == '__main__': main()
