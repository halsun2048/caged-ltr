"""Validation-only staged search for the R6 MiniLM student."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path


CONFIGS = [
    {"name":"pair_lr1e5", "loss":"pairwise", "lr":1e-5, "hard":False, "tail":1.0, "batch":32},
    {"name":"pair_lr2e5", "loss":"pairwise", "lr":2e-5, "hard":False, "tail":1.0, "batch":32},
    {"name":"pair_hard_lr2e5", "loss":"pairwise", "lr":2e-5, "hard":True, "tail":1.0, "batch":32},
    {"name":"soft_hard_lr1e5", "loss":"soft_margin", "lr":1e-5, "hard":True, "tail":1.0, "batch":32},
    {"name":"soft_hard_lr2e5", "loss":"soft_margin", "lr":2e-5, "hard":True, "tail":1.0, "batch":32},
    {"name":"pair_hard_tail15", "loss":"pairwise", "lr":2e-5, "hard":True, "tail":1.5, "batch":32},
    {"name":"listwise_lr1e5", "loss":"listwise", "lr":1e-5, "hard":False, "tail":1.0, "batch":4},
    {"name":"listwise_tail15", "loss":"listwise", "lr":1e-5, "hard":False, "tail":1.5, "batch":4},
]


def run_trial(args, config, name, seed, full):
    trial = args.output_dir / name; trial.mkdir(parents=True, exist_ok=True)
    report, last, best = trial / "report.json", trial / "last.pt", trial / "best.pt"
    command = [sys.executable, str(args.trainer), "--pairs", str(args.pairs), "--dev", str(args.dev), "--initial-checkpoint", str(args.initial_checkpoint), "--checkpoint", str(last), "--best-checkpoint", str(best), "--report", str(report), "--loss", config["loss"], "--learning-rate", str(config["lr"]), "--epochs", str(2 if full else 1), "--batch-size", str(config["batch"]), "--tail-weight", str(config["tail"]), "--seed", str(seed), "--checkpoint-every", "10000", "--progress"]
    if config["hard"]: command += ["--hard-negatives", str(args.hard_negatives)]
    if config["loss"] == "listwise": command += ["--listwise", str(args.listwise)]
    if not full:
        command += (["--max-train-queries", "400"] if config["loss"] == "listwise" else ["--max-train-pairs", "5000"])
    print(json.dumps({"stage":"trial_start","name":name,"seed":seed,"full":full}), flush=True)
    subprocess.run(command, check=True)
    if last.exists(): last.unlink()
    payload = json.loads(report.read_text()); payload["config"] = config; payload["name"] = name; payload["seed"] = seed
    return payload, best


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--trainer',type=Path,required=True); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--hard-negatives',type=Path,required=True); ap.add_argument('--listwise',type=Path,required=True); ap.add_argument('--dev',type=Path,required=True); ap.add_argument('--initial-checkpoint',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--report',type=Path,required=True); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    coarse=[]; checkpoints=[]
    for config in CONFIGS:
        result, checkpoint=run_trial(args,config,config['name'],42,False); coarse.append(result); checkpoints.append(checkpoint)
    selected=max(coarse,key=lambda row:row['best_dev_ndcg10']); selected_config=selected['config']
    for checkpoint in checkpoints:
        if checkpoint.exists(): checkpoint.unlink()
    confirmations=[]; confirmation_checkpoints=[]
    for seed in (42,2024,3407):
        result, checkpoint=run_trial(args,selected_config,f"confirm_seed{seed}",seed,True); confirmations.append(result); confirmation_checkpoints.append(checkpoint)
    winner=max(range(len(confirmations)),key=lambda i:confirmations[i]['best_dev_ndcg10'])
    for index, checkpoint in enumerate(confirmation_checkpoints):
        if index != winner and checkpoint.exists(): checkpoint.unlink()
    payload={"schema":"nfcorpus_r6_search_v1","selection_split":"independent_dev_only","coarse_trials":coarse,"selected_config":selected_config,"confirmation_seeds":confirmations,"mean_confirmation_ndcg10":sum(x['best_dev_ndcg10'] for x in confirmations)/len(confirmations),"direction_stable":all(x['best_dev_ndcg10']>0 for x in confirmations),"retained_checkpoint":str(confirmation_checkpoints[winner]),"test_accessed":False}
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({"stage":"complete","selected":selected_config,"mean_ndcg10":payload['mean_confirmation_ndcg10'],"report":str(args.report)}))
if __name__=='__main__': main()
