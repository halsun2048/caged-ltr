"""Fit, select and independently validate a BM25/MiniLM/FIRST cost-aware gate."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

MODELS=('bm25','distilled_minilm','first'); LATENCY={'bm25':0.0,'distilled_minilm':11.654340584288907,'first':427.9928405587209}
def ndcg(values):
    values=list(values); top=values[:10]; dcg=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(top)); ideal=sorted(values,reverse=True)[:10]; idcg=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(ideal)); return float(dcg/idcg) if idcg else 0.
def sha(path):
    digest=hashlib.sha256();
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()
def soft_features(values):
    values=np.asarray(values,float); ordered=np.sort(values)[::-1]; centered=np.clip(values-values.max(),-50,50); prob=np.exp(centered); prob/=prob.sum(); entropy=-sum(x*math.log(max(x,1e-12)) for x in prob)/math.log(len(prob)); return float(ordered[0]-ordered[1]),float(entropy),float(values.std())
def build(predictions,routes):
    rows=[]; route=routes.set_index('query_id')
    for query_id,group in predictions.groupby('query_id'):
        row={'query_id':query_id,'query_chars':len(group.iloc[0]['query']),'query_tokens':len(str(group.iloc[0]['query']).split()),'mean_passage_chars':float(group.passage.str.len().mean()),'mean_item_frequency':float(group.train_item_frequency.mean()),'max_item_frequency':float(group.train_item_frequency.max()),'entropy':float(route.loc[query_id,'entropy']),'stability':float(route.loc[query_id,'stability'])}
        for model in MODELS:
            margin,entropy,std=soft_features(group[model]); row[f'{model}_margin']=margin; row[f'{model}_score_entropy']=entropy; row[f'{model}_score_std']=std; ranked=group.sort_values(model,ascending=False); row[f'{model}_ndcg10']=ndcg(ranked.graded_relevance)
        use_first=bool(route.loc[query_id,'use_first']); frozen='first' if use_first else 'bm25'; row['frozen_gate_ndcg10']=row[f'{frozen}_ndcg10']; row['frozen_gate_route']=frozen
        rows.append(row)
    return pd.DataFrame(rows)
def route_frame(frame,models,scaler,lam):
    x=scaler.transform(frame[FEATURES]); predicted=np.column_stack([models[name].predict(x) for name in MODELS]); adjusted=predicted-np.array([LATENCY[name]*lam for name in MODELS]); chosen=np.array(MODELS)[adjusted.argmax(1)]; actual=np.array([frame.iloc[i][f'{name}_ndcg10'] for i,name in enumerate(chosen)]); return chosen,actual
FEATURES=['query_chars','query_tokens','mean_passage_chars','mean_item_frequency','max_item_frequency','entropy','stability']+[f'{model}_{field}' for model in MODELS for field in ('margin','score_entropy','score_std')]
def summarize(frame,chosen,actual): return {'queries':len(frame),'ndcg10':float(actual.mean()),'latency_ms_per_query':float(np.mean([LATENCY[x] for x in chosen])),'first_call_rate':float(np.mean(chosen=='first')),'minilm_call_rate':float(np.mean(chosen=='distilled_minilm')),'bm25_call_rate':float(np.mean(chosen=='bm25')),'routes':{name:int(np.sum(chosen==name)) for name in MODELS}}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',type=Path,required=True); ap.add_argument('--routes',type=Path,required=True); ap.add_argument('--student-checkpoint',type=Path,required=True); ap.add_argument('--formal-report',type=Path,required=True); ap.add_argument('--search-report',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--artifact',type=Path,required=True); args=ap.parse_args(); frame=build(pd.read_parquet(args.predictions),pd.read_parquet(args.routes)); ordered=sorted(frame.query_id,key=lambda value:hashlib.sha256(value.encode()).hexdigest()); fit_ids=set(ordered[:238]); select_ids=set(ordered[238:357]); valid_ids=set(ordered[357:]); fit=frame[frame.query_id.isin(fit_ids)].reset_index(drop=True); select=frame[frame.query_id.isin(select_ids)].reset_index(drop=True); valid=frame[frame.query_id.isin(valid_ids)].reset_index(drop=True)
    scaler=StandardScaler().fit(fit[FEATURES]); x=scaler.transform(fit[FEATURES]); regressors={name:Ridge(alpha=10.0).fit(x,fit[f'{name}_ndcg10']) for name in MODELS}; lambdas=(0.,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3); selection=[]
    for lam in lambdas:
        chosen,actual=route_frame(select,regressors,scaler,lam); selection.append({'lambda':lam,**summarize(select,chosen,actual)})
    eligible=[row for row in selection if row['first_call_rate']<=.5]; selected=max(eligible,key=lambda row:row['ndcg10']); chosen,actual=route_frame(valid,regressors,scaler,selected['lambda']); validation=summarize(valid,chosen,actual)
    baselines={name:{'ndcg10':float(valid[f'{name}_ndcg10'].mean()),'latency_ms_per_query':LATENCY[name]} for name in MODELS}; baselines['frozen_gate']={'ndcg10':float(valid.frozen_gate_ndcg10.mean()),'latency_ms_per_query':float(np.mean([LATENCY[x] for x in valid.frozen_gate_route])),'first_call_rate':float(np.mean(valid.frozen_gate_route=='first'))}
    artifact={'schema':'nfcorpus_r6_three_way_gate_v1','features':FEATURES,'scaler':{'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()},'ridge':{name:{'alpha':10.0,'coef':regressors[name].coef_.tolist(),'intercept':float(regressors[name].intercept_)} for name in MODELS},'cost_lambda':selected['lambda'],'latency_ms':LATENCY,'student_checkpoint_sha256':sha(args.student_checkpoint),'split_hashes':{'fit':hashlib.sha256('\n'.join(sorted(fit_ids)).encode()).hexdigest(),'select':hashlib.sha256('\n'.join(sorted(select_ids)).encode()).hexdigest(),'validation':hashlib.sha256('\n'.join(sorted(set(ordered)-fit_ids-select_ids)).encode()).hexdigest()},'test_accessed':False}; args.artifact.parent.mkdir(parents=True,exist_ok=True); args.artifact.write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
    formal=json.loads(args.formal_report.read_text()); search=json.loads(args.search_report.read_text()); models=formal['models']; confirmations=search['confirmation_seeds']; mind=models['mind_minilm']['ndcg10']; acceptance={'minilm_beats_tfidf_overall':models['distilled_minilm']['ndcg10']>models['tfidf']['ndcg10'],'minilm_tail_not_below_tfidf':models['distilled_minilm']['buckets']['tail']['ndcg10']>=models['tfidf']['buckets']['tail']['ndcg10'],'three_seed_direction_stable':all(row['best_dev_ndcg10']>mind for row in confirmations),'latency_within_budget':formal['efficiency']['distilled_minilm']['latency_ms_per_query']<20,'gate_near_first_with_50pct_or_less_calls':validation['ndcg10']>=baselines['first']['ndcg10']-.01 and validation['first_call_rate']<=.5}; payload={'schema':'nfcorpus_r6_gate_selection_v1','split':{'fit':len(fit),'select':len(select),'validation':len(valid)},'selection_pareto':selection,'selected':selected,'validation':validation,'validation_baselines':baselines,'acceptance':acceptance,'all_acceptance_passed':all(acceptance.values()),'artifact':str(args.artifact),'source_sha256':{'formal_report':sha(args.formal_report),'search_report':sha(args.search_report),'predictions':sha(args.predictions),'routes':sha(args.routes)},'test_accessed':False}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'stage':'complete','selected_lambda':selected['lambda'],'validation':validation,'acceptance':acceptance}))
if __name__=='__main__': main()
