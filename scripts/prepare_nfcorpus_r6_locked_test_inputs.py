"""Prepare label-free locked-test inputs after the R6 gate is frozen."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--queries',type=Path,required=True); ap.add_argument('--candidates',type=Path,required=True); ap.add_argument('--prompt-inputs',type=Path,required=True); ap.add_argument('--results',type=Path,required=True); ap.add_argument('--item-buckets',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); prompts={}
    with args.prompt_inputs.open() as handle:
        for line in handle:
            row=json.loads(line); prompts[row['fingerprint']]={item['identifier']:item['candidate_id'] for item in row['candidate_mapping']}
    variants={}; baseline={}; logits={}
    with args.results.open() as handle:
        for line in handle:
            row=json.loads(line); payload=row['payload']; query_id=payload['query_id']; mapping=prompts[row['key']]; variants.setdefault(query_id,{})[payload['variant']]=[mapping[x] for x in payload['first_token_ranking']]
            if payload['variant']=='baseline': baseline[query_id]=payload; logits.update({(query_id,mapping[key]):float(value) for key,value in payload['identifier_logits'].items()})
    feature_rows=[]
    for query_id,ranks in variants.items():
        base=ranks['baseline']; agreement=[]
        for name in ('reverse','random_permutation'):
            position={value:index for index,value in enumerate(ranks[name])}; agreement.append(np.mean([abs(index-position[value])<=2 for index,value in enumerate(base)]))
        payload=baseline[query_id]; feature_rows.append({'query_id':query_id,'entropy':float(payload['normalized_entropy']),'stability':float(np.mean(agreement)),'first_model_latency_ms':float(payload['prefill_seconds']+payload['decoding_seconds'])*1000})
    candidates=pd.read_parquet(args.candidates); queries=pd.read_parquet(args.queries); buckets=pd.read_parquet(args.item_buckets); frame=candidates[candidates.query_id.isin(baseline)].merge(queries[['query_id','query']],on='query_id',how='left').merge(buckets,on='passage_id',how='left').merge(pd.DataFrame(feature_rows),on='query_id',how='left'); frame['train_item_frequency']=frame.train_item_frequency.fillna(0).astype(int); frame['frequency_bucket']=frame.frequency_bucket.fillna('tail'); frame['first']=np.array([logits[(q,p)] for q,p in zip(frame.query_id,frame.passage_id)]); frame['bm25']=-frame.bm25_rank.astype(float)
    if frame[['query','first','entropy','stability']].isna().any().any(): raise RuntimeError('incomplete label-free locked-test inputs')
    args.output.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(args.output,index=False); print(json.dumps({'stage':'complete','queries':int(frame.query_id.nunique()),'rows':len(frame),'qrels_accessed':False}))
if __name__=='__main__': main()
