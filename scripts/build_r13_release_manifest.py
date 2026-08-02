#!/usr/bin/env python3
"""Create a release manifest for the locked R9/R12 evidence and protocol status."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def digest(p: Path) -> str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main() -> None:
 files=[ROOT/'reports/experiments/mind_r9_0_closeout.json',ROOT/'reports/experiments/mind_r12_gate_search.json',ROOT/'artifacts/mind_r8_0_large_test_guard.json',ROOT/'artifacts/mind_r12_0_independent_guard.json']
 manifest={"schema":"caged_ltr_r13_release_manifest_v1","git_commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),"evidence":{},"large_test_policy":"locked; no additional evaluation","r12_confirm_policy":"consumed_closed; evaluation_count=1","next_work":"qrels-free R13 protocol and train-fold OOF only"}
 for p in files:
  if p.exists(): manifest['evidence'][str(p.relative_to(ROOT))]={"sha256":digest(p),"bytes":p.stat().st_size}
 out=ROOT/'reports/data/mind_r13_release_manifest.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding='utf-8'); print(json.dumps({"stage":"complete","manifest":str(out)},ensure_ascii=False))

if __name__=='__main__': main()
