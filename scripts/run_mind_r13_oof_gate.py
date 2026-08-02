#!/usr/bin/env python3
"""Five-fold OOF hard-Tail gate on R12 dev only."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from run_mind_r11_gate_search import FEATURES, merge
from run_mind_r12_hard_tail_gate import route, bucket_metrics
from select_mind_r8_6_gate import metrics

def fold_id(q): return int(hashlib.sha256(str(q).encode()).hexdigest()[:8],16)%5
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--pool-root',type=Path,default=Path('data/processed/mind_r12_0')); ap.add_argument('--first-root',type=Path,default=Path('runs/mind_r12_0')); ap.add_argument('--metrics',type=Path,default=Path('runs/mind_r12_0/r12_dev_query_metrics.parquet')); ap.add_argument('--output',type=Path,default=Path('reports/experiments/mind_r13_oof_gate.json')); ap.add_argument('--progress',action='store_true'); a=ap.parse_args()
 frame=merge('dev',a.metrics,a.pool_root,a.first_root).reset_index(drop=True); folds=np.array([fold_id(x) for x in frame.query_id]); pred=np.zeros(len(frame))
 for f in range(5):
  tr=folds!=f; va=~tr; model=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=10,random_state=20260805+f,n_jobs=-1).fit(frame.loc[tr,FEATURES].fillna(0),frame.loc[tr,'first_ndcg10']-frame.loc[tr,'ndcg10']); pred[va]=model.predict(frame.loc[va,FEATURES].fillna(0));
  if a.progress: print(f'[OOF] fold {f+1}/5 train={tr.sum()} valid={va.sum()}',flush=True)
 candidates=[]
 for budget in (.45,.50,.55,.60):
  for torso in (.35,.50,.65):
   fold_rows=[]; feasible=True
   for f in range(5):
    g=frame.loc[folds==f].reset_index(drop=True); r=route(g,pred[folds==f],budget,torso)
    if r is None: feasible=False; break
    b=bucket_metrics(g,r); m=metrics(g,r); fold_rows.append({'fold':f,'overall':m,'buckets':b,'tail_gap':b['tail']['first']['ndcg10']-b['tail']['gate']['ndcg10'],'torso_gap':b['torso']['first']['ndcg10']-b['torso']['gate']['ndcg10']})
   if feasible: candidates.append({'budget':budget,'tail_floor':1.0,'torso_floor':torso,'folds':fold_rows,'worst_tail_gap':max(x['tail_gap'] for x in fold_rows),'worst_torso_gap':max(x['torso_gap'] for x in fold_rows),'mean_ndcg10':float(np.mean([x['overall']['ndcg10'] for x in fold_rows])),'mean_call_rate':float(np.mean([x['overall']['first_call_rate'] for x in fold_rows]))})
 eligible=[c for c in candidates if c['worst_tail_gap']<=.003 and c['mean_call_rate']<=.60]; selected=min(eligible,key=lambda c:(c['worst_torso_gap'],c['mean_call_rate']))
 payload={'schema':'mind_r13_five_fold_oof_gate_v1','source_split':'r12_dev_only','historical_confirm_accessed':False,'large_test_accessed':False,'candidates':candidates,'selected':selected,'acceptance':{'all_folds_tail_pass':selected['worst_tail_gap']<=.003,'call_rate_at_most_0p60':selected['mean_call_rate']<=.60}}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'stage':'complete','selected':{k:selected[k] for k in ('budget','tail_floor','torso_floor','worst_torso_gap','mean_ndcg10','mean_call_rate')}},ensure_ascii=False))
if __name__=='__main__': main()
