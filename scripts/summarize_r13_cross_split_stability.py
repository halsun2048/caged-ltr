#!/usr/bin/env python3
"""Summarize fixed-policy behavior across R10/R11/R12 without new evaluation."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT/'reports/experiments'
def load(n): return json.loads((EXP/n).read_text())
def main():
 rows=[]
 for name in ('mind_r10_gate_search.json','mind_r11_gate_search.json','mind_r12_gate_search.json'):
  d=load(name); c=d.get('confirm',{}); g=c.get('gate',{}); b=c.get('buckets',{})
  rows.append({'split':name.split('_')[1],'ndcg10':g.get('ndcg10'),'call_rate':g.get('first_call_rate'),'tail_ndcg10':b.get('tail',{}).get('gate',{}).get('ndcg10'),'torso_ndcg10':b.get('torso',{}).get('gate',{}).get('ndcg10'),'accepted':d.get('acceptance',{}).get('all_passed',False)})
 payload={'schema':'mind_r13_cross_split_stability_v1','historical_confirm_reopened':False,'rows':rows,'interpretation':'R10/R11 gain gate fail Tail; R12 hard-Tail passes Tail and overall. This is cross-split descriptive evidence, not multi-seed OOF.','next_required':'true train-fold OOF on qrels-free data'}
 out=ROOT/'reports/experiments/mind_r13_cross_split_stability.json'; out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'stage':'complete','report':str(out)},ensure_ascii=False))
if __name__=='__main__': main()
