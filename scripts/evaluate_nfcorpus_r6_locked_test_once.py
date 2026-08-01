"""Apply the frozen R6 gate and access locked-test qrels exactly once."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,pandas as pd
MODELS=('bm25','distilled_minilm','first')
FEATURES=['query_chars','query_tokens','mean_passage_chars','mean_item_frequency','max_item_frequency','entropy','stability']+[f'{model}_{field}' for model in MODELS for field in ('margin','score_entropy','score_std')]
def sha(path):
    digest=hashlib.sha256();
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1048576),b''): digest.update(chunk)
    return digest.hexdigest()
def ndcg(values):
    values=list(values); top=values[:10]; dcg=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(top)); ideal=sorted(values,reverse=True)[:10]; idcg=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(ideal)); return float(dcg/idcg) if idcg else 0.
def score_features(values):
    values=np.asarray(values,float); ordered=np.sort(values)[::-1]; centered=np.clip(values-values.max(),-50,50); probability=np.exp(centered); probability/=probability.sum(); entropy=-sum(x*math.log(max(x,1e-12)) for x in probability)/math.log(len(probability)); return float(ordered[0]-ordered[1]),float(entropy),float(values.std())
def query_table(frame):
    rows=[]
    for query_id,group in frame.groupby('query_id'):
        row={'query_id':query_id,'query_chars':len(group.iloc[0]['query']),'query_tokens':len(str(group.iloc[0]['query']).split()),'mean_passage_chars':float(group.passage.str.len().mean()),'mean_item_frequency':float(group.train_item_frequency.mean()),'max_item_frequency':float(group.train_item_frequency.max()),'entropy':float(group.iloc[0].entropy),'stability':float(group.iloc[0].stability),'first_model_latency_ms':float(group.iloc[0].first_model_latency_ms)}
        for model in MODELS:
            margin,entropy,std=score_features(group[model]); row[f'{model}_margin']=margin; row[f'{model}_score_entropy']=entropy; row[f'{model}_score_std']=std; ranked=group.sort_values(model,ascending=False); rel=ranked.graded_relevance.to_numpy(); relevant=np.flatnonzero(rel>0); row[f'{model}_ndcg10']=ndcg(rel); row[f'{model}_hit10']=float(np.any(rel[:10]>0)); row[f'{model}_mrr']=float(1/(relevant[0]+1)) if len(relevant) else 0.
        rows.append(row)
    return pd.DataFrame(rows)
def summary(table,route):
    values={metric:np.array([table.iloc[i][f'{name}_{metric}'] for i,name in enumerate(route)]) for metric in ('ndcg10','hit10','mrr')}; return {metric:float(value.mean()) for metric,value in values.items()},values
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scored',type=Path,required=True); ap.add_argument('--qrels',type=Path,required=True); ap.add_argument('--gate-artifact',type=Path,required=True); ap.add_argument('--scoring-metadata',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    if args.output.exists(): raise RuntimeError('locked-test output already exists; refusing a second evaluation')
    scored=pd.read_parquet(args.scored); qrels=pd.read_parquet(args.qrels)[['query_id','passage_id','graded_relevance']]; frame=scored.merge(qrels,on=['query_id','passage_id'],how='left'); frame['graded_relevance']=frame.graded_relevance.fillna(0.); table=query_table(frame); artifact=json.loads(args.gate_artifact.read_text()); metadata=json.loads(args.scoring_metadata.read_text()); mean=np.array(artifact['scaler']['mean']); scale=np.array(artifact['scaler']['scale']); x=(table[FEATURES].to_numpy()-mean)/scale; predicted=[]
    for model in MODELS:
        spec=artifact['ridge'][model]; predicted.append(x@np.array(spec['coef'])+float(spec['intercept'])-artifact['cost_lambda']*artifact['latency_ms'][model])
    route=np.array(MODELS)[np.column_stack(predicted).argmax(1)]; gate_metrics,gate_values=summary(table,route); baselines={}
    for model in MODELS: baselines[model]={metric:float(table[f'{model}_{metric}'].mean()) for metric in ('ndcg10','hit10','mrr')}
    frozen=np.where((table.entropy<=.7)&(table.stability>=.2),'first','bm25'); baselines['previous_frozen_gate']=summary(table,frozen)[0]
    latency={'bm25':0.,'distilled_minilm':float(metadata['latency_ms_per_query'])}; gate_latency=float(np.mean([table.iloc[i].first_model_latency_ms if name=='first' else latency[name] for i,name in enumerate(route)])); first_latency=float(table.first_model_latency_ms.mean()); frozen_latency=float(np.mean(np.where(frozen=='first',table.first_model_latency_ms,0.)))
    rng=np.random.default_rng(20240801); bootstrap={'gate_minus_first':[],'gate_minus_bm25':[]}
    for _ in range(10000):
        idx=rng.integers(0,len(table),len(table)); bootstrap['gate_minus_first'].append(float((gate_values['ndcg10'][idx]-table.first_ndcg10.to_numpy()[idx]).mean())); bootstrap['gate_minus_bm25'].append(float((gate_values['ndcg10'][idx]-table.bm25_ndcg10.to_numpy()[idx]).mean()))
    ci={key:[float(np.quantile(value,.025)),float(np.quantile(value,.975))] for key,value in bootstrap.items()}; payload={'schema':'nfcorpus_r6_locked_test_v1','locked_commit':'365ac7c','queries':len(table),'gate':{**gate_metrics,'routes':{name:int(np.sum(route==name)) for name in MODELS},'first_call_rate':float(np.mean(route=='first')),'latency_ms_per_query':gate_latency},'baselines':baselines,'efficiency':{'first_latency_ms_per_query':first_latency,'previous_frozen_gate_latency_ms_per_query':frozen_latency,'distilled_minilm_latency_ms_per_query':latency['distilled_minilm']},'paired_bootstrap_95ci':ci,'acceptance':{'near_first_with_50pct_or_less_calls':gate_metrics['ndcg10']>=baselines['first']['ndcg10']-.01 and np.mean(route=='first')<=.5},'source_sha256':{'scored':sha(args.scored),'qrels':sha(args.qrels),'gate_artifact':sha(args.gate_artifact),'scoring_metadata':sha(args.scoring_metadata)},'test_accessed_once':True,'further_tuning_prohibited':True}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'stage':'locked_complete','gate':payload['gate'],'baselines':baselines,'no_further_tuning':True}))
if __name__=='__main__': main()
