#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 r9=json.loads((ROOT/'reports/experiments/mind_r9_0_closeout.json').read_text()); r12=json.loads((ROOT/'reports/experiments/mind_r12_gate_search.json').read_text())
 rows=[]
 for name,x in r9['final_metrics'].items(): rows.append({'experiment':'R9 large-test','method':name,'ndcg10':x['ndcg10'],'call_rate':x.get('first_call_rate',0),'latency_ms':x.get('latency_ms')})
 rows.append({'experiment':'R12 independent confirm','method':'hard-tail gate','ndcg10':r12['confirm']['gate']['ndcg10'],'call_rate':r12['confirm']['gate']['first_call_rate'],'latency_ms':r12['confirm']['gate']['latency_ms']})
 out=ROOT/'reports/experiments/mind_r13_efficiency_summary.json'; out.write_text(json.dumps({'schema':'mind_r13_efficiency_summary_v1','rows':rows,'p50_p95_p99':'not measured; requires service benchmark','large_test_reopened':False},ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'stage':'complete','report':str(out)},ensure_ascii=False))
if __name__=='__main__': main()
