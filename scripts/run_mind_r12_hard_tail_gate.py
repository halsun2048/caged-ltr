#!/usr/bin/env python3
"""R12 hard-Tail=100% gate search; selection uses dev only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from run_mind_r11_gate_search import FEATURES, merge
from select_mind_r8_6_gate import metrics

def route(frame, pred, budget, torso_floor):
    n=len(frame); route=np.zeros(n,dtype=bool); total=round(n*budget)
    tail=np.flatnonzero(frame.frequency_bucket.to_numpy()=="tail")
    torso=np.flatnonzero(frame.frequency_bucket.to_numpy()=="torso")
    route[tail]=True
    torso_n=min(len(torso),round(len(torso)*torso_floor))
    if torso_n: route[torso[np.argsort(-pred[torso],kind="stable")[:torso_n]]]=True
    remaining=total-int(route.sum())
    if remaining<0: return None
    other=np.flatnonzero(~route)
    if remaining: route[other[np.argsort(-pred[other],kind="stable")[:remaining]]]=True
    return route

def bucket_metrics(frame, route):
    out={}
    for k,g in frame.groupby("frequency_bucket",sort=True):
        idx=g.index.to_numpy(); out[k]={"route_rate":float(route[idx].mean()),"student":metrics(g,np.zeros(len(g),bool)),"first":metrics(g,np.ones(len(g),bool)),"gate":metrics(g,route[idx])}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pool-root",type=Path,default=Path("data/processed/mind_r12_0")); ap.add_argument("--first-root",type=Path,default=Path("runs/mind_r12_0")); ap.add_argument("--dev-metrics",type=Path,default=Path("runs/mind_r12_0/r12_dev_query_metrics.parquet")); ap.add_argument("--confirm-metrics",type=Path,default=Path("runs/mind_r12_0/r12_confirm_query_metrics.parquet")); ap.add_argument("--output",type=Path,default=Path("reports/experiments/mind_r12_gate_search.json")); ap.add_argument("--progress",action="store_true"); a=ap.parse_args()
    if a.progress: print("[R12 1/4] loading metrics",flush=True)
    dev=merge("dev",a.dev_metrics,a.pool_root,a.first_root); confirm=merge("confirm",a.confirm_metrics,a.pool_root,a.first_root)
    model=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=10,random_state=20260804,n_jobs=-1).fit(dev[FEATURES].fillna(0),dev.first_ndcg10-dev.ndcg10)
    pd=model.predict(dev[FEATURES].fillna(0)); pc=model.predict(confirm[FEATURES].fillna(0))
    first_d=metrics(dev,np.ones(len(dev),bool)); first_c=metrics(confirm,np.ones(len(confirm),bool)); first_db={k:metrics(g,np.ones(len(g),bool)) for k,g in dev.groupby("frequency_bucket",sort=True)}
    candidates=[]
    if a.progress: print("[R12 2/4] hard Tail search on dev",flush=True)
    for budget in (.45,.50,.55,.60):
      for torso_floor in (.35,.50,.65):
        r=route(dev,pd,budget,torso_floor)
        if r is None: continue
        bm=bucket_metrics(dev,r); m=metrics(dev,r)
        candidates.append({"budget":budget,"tail_floor":1.0,"torso_floor":torso_floor,"overall":m,"by_bucket":bm,"dev_tail_gap":first_db["tail"]["ndcg10"]-bm["tail"]["gate"]["ndcg10"],"dev_torso_gap":first_db["torso"]["ndcg10"]-bm["torso"]["gate"]["ndcg10"]})
    eligible=[c for c in candidates if c["dev_tail_gap"]<=.003 and c["overall"]["first_call_rate"]<=.60]
    selected=min(eligible,key=lambda c:(c["dev_torso_gap"],c["overall"]["first_call_rate"])) if eligible else None
    result={"schema":"mind_r12_hard_tail_gate_v1","selection_split":"dev","confirm_split":"confirm","search_space":{"budget":[.45,.50,.55,.60],"tail_floor":[1.0],"torso_floor":[.35,.50,.65]},"candidates":candidates,"selected":selected,"dev_first":first_d,"confirm_first":first_c,"large_test_accessed":False}
    if selected:
      r=route(confirm,pc,selected["budget"],selected["torso_floor"]); cb=bucket_metrics(confirm,r); cm=metrics(confirm,r); first_cb=cb["tail"]["first"]; result["confirm"]={"gate":cm,"buckets":cb}; result["acceptance"]={"overall_gap_at_most_0p003":first_c["ndcg10"]-cm["ndcg10"]<=.003,"tail_gap_at_most_0p003":first_cb["ndcg10"]-cb["tail"]["gate"]["ndcg10"]<=.003,"first_call_rate_at_most_0p60":cm["first_call_rate"]<=.60,"gate_at_least_student":cm["ndcg10"]>=metrics(confirm,np.zeros(len(confirm),bool))["ndcg10"]}
      result["acceptance"]["all_passed"]=all(result["acceptance"].values())
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    if a.progress: print("[R12 3/4] confirm fixed evaluation computed",flush=True); print("[R12 4/4] complete",flush=True)
    print(json.dumps({"stage":"complete","selected":selected,"acceptance":result.get("acceptance")},ensure_ascii=False))

if __name__=="__main__": main()
