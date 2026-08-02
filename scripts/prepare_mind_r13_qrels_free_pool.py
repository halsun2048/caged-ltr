#!/usr/bin/env python3
"""Materialize candidate pools without relevance/qrels columns."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import duckdb

def paths(ps): return "["+",".join("'"+str(p).replace("'","''")+"'" for p in ps)+"]"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-root',type=Path,default=Path('data/external/mind/mteb_english')); ap.add_argument('--split-ids',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--candidates',type=int,default=20); ap.add_argument('--progress',action='store_true'); a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    q=next((a.raw_root/'queries').glob('*.parquet')); corpus=next((a.raw_root/'corpus').glob('*.parquet')); ranks=sorted((a.raw_root/'top_ranked').glob('*.parquet')); con=duckdb.connect(); con.execute("PRAGMA threads=8")
    for split in ('dev','confirm'):
      out=a.output_dir/f'{split}.parquet'
      if a.progress: print(f'[pool:{split}] building qrels-free pool',flush=True)
      con.execute(f"COPY (WITH selected AS (SELECT query_id FROM read_parquet('{a.split_ids}') WHERE split='{split}'), q AS (SELECT id AS query_id,text AS query FROM read_parquet('{q}')), r AS (SELECT \"query-id\" AS query_id,\"corpus-ids\" AS corpus_ids FROM read_parquet({paths(ranks)})), raw AS (SELECT r.query_id,list_extract(r.corpus_ids,u.rank) AS corpus_id,u.rank AS source_rank FROM r INNER JOIN selected s ON r.query_id=s.query_id,UNNEST(range(1,least(len(r.corpus_ids),{a.candidates})+1)) u(rank)), ranked AS (SELECT query_id,corpus_id,min(source_rank)::INTEGER source_rank FROM raw GROUP BY query_id,corpus_id), c AS (SELECT id AS corpus_id,text AS passage FROM read_parquet('{corpus}')) SELECT ranked.query_id,q.query,ranked.corpus_id,c.passage,ranked.source_rank,'{split}' AS split FROM ranked INNER JOIN q USING(query_id) INNER JOIN c USING(corpus_id) ORDER BY ranked.query_id,ranked.source_rank) TO '{out}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    print(json.dumps({'stage':'complete','output_dir':str(a.output_dir),'qrels_free':True},ensure_ascii=False))
if __name__=='__main__': main()
