"""Nested-CV comparison of Ridge, logistic and MLP three-way gates on dev only."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
MODELS=np.array(['bm25','distilled_minilm','first']); LAT=np.array([0.,11.654340584288907,427.9928405587209]); LAMBDAS=(0.,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4)
def ndcg(values):
    values=list(values); dcg=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(values[:10])); ideal=sorted(values,reverse=True)[:10]; idcg=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(ideal)); return dcg/idcg if idcg else 0.
def sf(values):
    values=np.asarray(values,float); ordered=np.sort(values)[::-1]; centered=np.clip(values-values.max(),-50,50); p=np.exp(centered);p/=p.sum();return float(ordered[0]-ordered[1]),float(-sum(x*math.log(max(x,1e-12)) for x in p)/math.log(len(p))),float(values.std())
def build(predictions,routes):
    route=routes.set_index('query_id');rows=[]
    for query_id,g in predictions.groupby('query_id'):
        row={'query_id':query_id,'query_chars':len(g.iloc[0]['query']),'query_tokens':len(str(g.iloc[0]['query']).split()),'mean_passage_chars':float(g.passage.str.len().mean()),'short_passage_rate':float((g.passage.str.len()<300).mean()),'mean_item_frequency':float(g.train_item_frequency.mean()),'max_item_frequency':float(g.train_item_frequency.max()),'entropy':float(route.loc[query_id,'entropy']),'stability':float(route.loc[query_id,'stability'])}
        for model in MODELS:
            margin,entropy,std=sf(g[model]);row[f'{model}_margin']=margin;row[f'{model}_entropy']=entropy;row[f'{model}_std']=std;row[f'{model}_ndcg']=ndcg(g.sort_values(model,ascending=False).graded_relevance)
        row['margin_diff_minilm_bm25']=row['distilled_minilm_margin']-row['bm25_margin'];row['entropy_diff_minilm_bm25']=row['distilled_minilm_entropy']-row['bm25_entropy'];row['frozen_route']='first' if row['entropy']<=.7 and row['stability']>=.2 else 'bm25';rows.append(row)
    return pd.DataFrame(rows)
FEATURES=['query_chars','query_tokens','mean_passage_chars','short_passage_rate','mean_item_frequency','max_item_frequency','entropy','stability']+[f'{m}_{f}' for m in MODELS for f in ('margin','entropy','std')]+['margin_diff_minilm_bm25','entropy_diff_minilm_bm25']
CONFIGS=[('ridge',1.),('ridge',10.),('ridge',100.),('logistic',.1),('logistic',1.),('mlp',.0001),('mlp',.001)]
def fit_predict(kind,param,train,test):
    xtrain=train[FEATURES];xtest=test[FEATURES];targets=train[[f'{m}_ndcg' for m in MODELS]].to_numpy()
    if kind=='logistic':
        label=targets.argmax(1);model=make_pipeline(StandardScaler(),LogisticRegression(C=param,max_iter=2000,class_weight='balanced',random_state=42)).fit(xtrain,label);prob=model.predict_proba(xtest);out=np.zeros((len(test),3));out[:,model[-1].classes_]=prob;return out
    predictions=[]
    for index in range(3):
        estimator=Ridge(alpha=param) if kind=='ridge' else MLPRegressor(hidden_layer_sizes=(16,),alpha=param,max_iter=1000,early_stopping=True,random_state=42)
        model=make_pipeline(StandardScaler(),estimator).fit(xtrain,targets[:,index]);predictions.append(model.predict(xtest))
    return np.column_stack(predictions)
def actual(frame,route): return np.array([frame.iloc[i][f'{name}_ndcg'] for i,name in enumerate(route)])
def choose(pred,lam): return MODELS[(pred-LAT*lam).argmax(1)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--routes',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();frame=build(pd.read_parquet(args.predictions),pd.read_parquet(args.routes)).sort_values('query_id').reset_index(drop=True);outer=KFold(5,shuffle=True,random_state=42);oof_route=np.empty(len(frame),dtype=object);folds=[]
    for fold,(train_idx,test_idx) in enumerate(outer.split(frame)):
        outer_train=frame.iloc[train_idx].reset_index(drop=True);outer_test=frame.iloc[test_idx].reset_index(drop=True);inner=KFold(4,shuffle=True,random_state=100+fold);candidates=[]
        for kind,param in CONFIGS:
            for lam in LAMBDAS:
                values=[];calls=[]
                for inner_train,inner_valid in inner.split(outer_train):
                    valid=outer_train.iloc[inner_valid].reset_index(drop=True);pred=fit_predict(kind,param,outer_train.iloc[inner_train],valid);route=choose(pred,lam);values.extend(actual(valid,route));calls.extend(route=='first')
                candidates.append({'kind':kind,'parameter':param,'lambda':lam,'ndcg':float(np.mean(values)),'first_call_rate':float(np.mean(calls))})
        eligible=[x for x in candidates if x['first_call_rate']<=.5];selected=max(eligible,key=lambda x:x['ndcg']);pred=fit_predict(selected['kind'],selected['parameter'],outer_train,outer_test);route=choose(pred,selected['lambda']);oof_route[test_idx]=route;folds.append({'fold':fold,'selected':selected,'test_ndcg':float(actual(outer_test,route).mean()),'test_first_call_rate':float(np.mean(route=='first'))})
    values=actual(frame,oof_route);frozen=frame.frozen_route.to_numpy();frozen_values=actual(frame,frozen);first=frame.first_ndcg.to_numpy();bm25=frame.bm25_ndcg.to_numpy();rng=np.random.default_rng(20240803);boot={'gate_minus_first':[],'gate_minus_bm25':[],'gate_minus_frozen':[]}
    for _ in range(10000):
        idx=rng.integers(0,len(frame),len(frame));boot['gate_minus_first'].append(float((values[idx]-first[idx]).mean()));boot['gate_minus_bm25'].append(float((values[idx]-bm25[idx]).mean()));boot['gate_minus_frozen'].append(float((values[idx]-frozen_values[idx]).mean()))
    payload={'schema':'nfcorpus_r7_nested_gate_v1','protocol':'5-fold outer / 4-fold inner nested CV on independent dev only','queries':len(frame),'features':FEATURES,'folds':folds,'oof':{'ndcg10':float(values.mean()),'first_call_rate':float(np.mean(oof_route=='first')),'minilm_call_rate':float(np.mean(oof_route=='distilled_minilm')),'bm25_call_rate':float(np.mean(oof_route=='bm25')),'latency_ms_per_query':float(np.mean([LAT[list(MODELS).index(x)] for x in oof_route]))},'baselines':{'bm25_ndcg10':float(bm25.mean()),'first_ndcg10':float(first.mean()),'frozen_gate_ndcg10':float(frozen_values.mean())},'paired_bootstrap_95ci':{key:[float(np.quantile(value,.025)),float(np.quantile(value,.975))] for key,value in boot.items()},'test_accessed':False};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');args.output.with_suffix('.md').write_text('# R7.2 nested-CV gate\n\n```json\n'+json.dumps(payload,ensure_ascii=False,indent=2)+'\n```\n');print(json.dumps({'stage':'complete','oof':payload['oof'],'ci':payload['paired_bootstrap_95ci']}))
if __name__=='__main__':main()
