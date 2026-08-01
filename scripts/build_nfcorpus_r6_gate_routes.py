"""Materialize the frozen v2 gate routes without accessing test data."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prompt-inputs',type=Path,required=True); ap.add_argument('--results',type=Path,required=True); ap.add_argument('--dev-queries',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    dev=set(args.dev_queries.read_text().splitlines()); prompts={}
    with args.prompt_inputs.open() as handle:
        for line in handle:
            row=json.loads(line); prompts[row['fingerprint']]={item['identifier']:item['candidate_id'] for item in row['candidate_mapping']}
    variants={}; baseline={}
    with args.results.open() as handle:
        for line in handle:
            row=json.loads(line); payload=row['payload']; query_id=payload['query_id']
            if query_id not in dev: continue
            mapping=prompts[row['key']]; ranking=[mapping[value] for value in payload['first_token_ranking']]
            variants.setdefault(query_id,{})[payload['variant']]=ranking
            if payload['variant']=='baseline': baseline[query_id]=payload
    rows=[]
    for query_id in sorted(dev):
        ranks=variants[query_id]; base=ranks['baseline']; agreements=[]
        for name in ('reverse','random_permutation'):
            positions={value:index for index,value in enumerate(ranks[name])}; agreements.append(np.mean([abs(index-positions[value])<=2 for index,value in enumerate(base)]))
        stability=float(np.mean(agreements)); payload=baseline[query_id]; entropy=float(payload['normalized_entropy']); rows.append({'query_id':query_id,'entropy':entropy,'stability':stability,'use_first':bool(entropy<=0.7 and stability>=0.2),'first_model_latency_ms':float(payload['prefill_seconds']+payload['decoding_seconds'])*1000})
    frame=pd.DataFrame(rows); args.output.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(args.output,index=False); print(json.dumps({'queries':len(frame),'first_routes':int(frame.use_first.sum()),'bm25_routes':int((~frame.use_first).sum()),'test_accessed':False}))
if __name__=='__main__': main()
