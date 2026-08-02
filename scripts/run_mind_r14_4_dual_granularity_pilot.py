"""Train a bounded objective-level Pairwise/Listwise MiniLM pilot on train-only FIRST logits."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class BiEncoder(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def embed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return nn.functional.normalize(pooled, dim=-1)


def tokenize(tokenizer, texts, device, max_length):
    batch = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ndcg(relevance: np.ndarray) -> float:
    ranked = relevance[:10]
    ideal = np.sort(relevance)[::-1][:10]
    discounts = np.log2(np.arange(2, len(ranked) + 2))
    dcg = np.sum((2**ranked - 1) / discounts)
    idcg = np.sum((2**ideal - 1) / discounts)
    return float(dcg / idcg) if idcg else 0.0


def load_teacher(prompts_path: Path, results_path: Path) -> dict[str, dict[str, float]]:
    mapping = {}
    with prompts_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            mapping[str(row["query_id"])] = {x["identifier"]: str(x["candidate_id"]) for x in row["candidate_mapping"]}
    output = {}
    with results_path.open() as handle:
        for line in handle:
            payload = json.loads(line)["payload"]
            query_id = str(payload["query_id"])
            ids = mapping[query_id]
            output[query_id] = {ids[key]: float(value) for key, value in payload["identifier_logits"].items()}
    return output


def groups_from_frame(frame: pd.DataFrame, teacher=None):
    groups = []
    for query_id, group in frame.groupby("query_id", sort=False):
        group = group.sort_values("source_rank")
        row = {
            "query_id": str(query_id), "query": str(group.iloc[0].query),
            "passages": group.passage.astype(str).tolist(),
            "relevance": group.relevance.to_numpy(np.float32),
        }
        if teacher is not None:
            scores = teacher.get(str(query_id), {})
            if len(scores) != len(group):
                continue
            row["teacher"] = np.asarray([scores[str(x)] for x in group.corpus_id], np.float32)
        else:
            positive = group.loc[group.relevance > 0, "train_item_frequency"]
            frequency = float(positive.max()) if len(positive) else 0.0
            row["bucket"] = "tail" if frequency <= 883 else ("torso" if frequency <= 4029 else "head")
        groups.append(row)
    return groups


def load_dev(pattern: str, limit: int):
    paths = sorted(glob.glob(pattern))
    ids = []
    for path in paths:
        ids.extend(pd.read_parquet(path, columns=["query_id"]).query_id.unique())
    keep = set(sorted(set(ids), key=lambda q: hashlib.sha256(str(q).encode()).hexdigest())[:limit])
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frames.append(frame[frame.query_id.isin(keep)])
    return pd.concat(frames, ignore_index=True)


def pairwise_per_query(student, teacher):
    teacher_delta = teacher[:, :, None] - teacher[:, None, :]
    student_delta = student[:, :, None] - student[:, None, :]
    upper = torch.triu(torch.ones_like(teacher_delta, dtype=torch.bool), diagonal=1)
    mask = upper & (teacher_delta.abs() >= 0.25)
    loss = nn.functional.softplus(-teacher_delta.sign() * student_delta) * teacher_delta.abs().clamp(max=2.0)
    return (loss * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)


@torch.inference_mode()
def evaluate(model, tokenizer, groups, device, max_length, batch_size):
    model.eval()
    rows = []
    for offset in range(0, len(groups), batch_size):
        batch_groups = groups[offset : offset + batch_size]
        count = len(batch_groups[0]["passages"])
        query = tokenize(tokenizer, [x["query"] for x in batch_groups], device, max_length)
        passages = tokenize(tokenizer, [p for x in batch_groups for p in x["passages"]], device, max_length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = model.embed(query)
            p = model.embed(passages).reshape(len(batch_groups), count, -1)
            score = torch.einsum("bd,bnd->bn", q, p).float().cpu().numpy()
        for group, values in zip(batch_groups, score, strict=True):
            order = np.argsort(-values, kind="stable")
            rows.append((group["bucket"], ndcg(group["relevance"][order])))
    frame = pd.DataFrame(rows, columns=["bucket", "ndcg10"])
    return {
        "overall_ndcg10": float(frame.ndcg10.mean()),
        "buckets": {key: {"ndcg10": float(group.ndcg10.mean()), "queries": len(group)} for key, group in frame.groupby("bucket")},
    }


def train_variant(name, seed, train_groups, dev_groups, tokenizer, args, device, route_threshold):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = BiEncoder(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    started = time.time()
    last_loss = None
    for epoch in range(args.epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        by_count: dict[int, list[int]] = {}
        for index, group in enumerate(train_groups):
            by_count.setdefault(len(group["passages"]), []).append(index)
        chosen_batches = []
        for indices in by_count.values():
            order = rng.permutation(indices)
            chosen_batches.extend(
                order[offset : offset + args.batch_size]
                for offset in range(0, len(order), args.batch_size)
            )
        rng.shuffle(chosen_batches)
        batches = len(chosen_batches)
        for batch_number, chosen in enumerate(chosen_batches):
            groups = [train_groups[i] for i in chosen]
            count = len(groups[0]["passages"])
            query = tokenize(tokenizer, [x["query"] for x in groups], device, args.max_length)
            passages = tokenize(tokenizer, [p for x in groups for p in x["passages"]], device, args.max_length)
            teacher = torch.as_tensor(np.stack([x["teacher"] for x in groups]), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = model.embed(query)
                p = model.embed(passages).reshape(len(groups), count, -1)
                score = torch.einsum("bd,bnd->bn", q, p)
                list_loss = -(torch.softmax(teacher, 1) * torch.log_softmax(score / 0.05, 1)).sum(1).mean()
                pair_query = pairwise_per_query(score, teacher)
                if name == "listwise_only": loss = list_loss
                elif name == "pairwise_only": loss = pair_query.mean()
                elif name == "joint": loss = list_loss + 0.5 * pair_query.mean()
                else:
                    entropy = -(torch.softmax(teacher, 1) * torch.log_softmax(teacher, 1)).sum(1)
                    routed = entropy >= route_threshold
                    pair = pair_query[routed].mean() if routed.any() else pair_query.mean() * 0
                    loss = list_loss + 0.5 * pair
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            last_loss = float(loss.detach())
            if args.progress and (batch_number % 20 == 0 or batch_number + 1 == batches):
                total_done = epoch * batches + batch_number + 1; total = args.epochs * batches
                elapsed = time.time() - started; eta = elapsed / total_done * (total - total_done)
                print(f"[{name} seed={seed}] epoch={epoch+1}/{args.epochs} batch={batch_number+1}/{batches} loss={last_loss:.4f} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
    metrics = evaluate(model, tokenizer, dev_groups, device, args.max_length, args.eval_batch_size)
    metrics.update({"variant": name, "seed": seed, "last_loss": last_loss, "elapsed_seconds": round(time.time()-started,2), "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()/2**30, "pairwise_route_rate": 0.0 if name=="listwise_only" else (args.route_rate if name=="routed_joint" else 1.0)})
    del model; torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("data/processed/mind_r14_4/pilot_pool.parquet"))
    parser.add_argument("--prompts", type=Path, default=Path("runs/mind_r14_4/pilot_prompts.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("runs/mind_r14_4/first/results.jsonl"))
    parser.add_argument("--dev", default="data/processed/mind_r8_1_v2/dev_listwise/*.parquet")
    parser.add_argument("--model", default="/root/caged-ltr/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=3); parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1); parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5); parser.add_argument("--dev-queries", type=int, default=2000)
    parser.add_argument("--route-rate", type=float, default=0.4); parser.add_argument("--seeds", nargs="+", type=int, default=[42,2024])
    parser.add_argument("--output", type=Path, default=Path("reports/experiments/mind_r14_4_dual_granularity_pilot.json")); parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    pool = pd.read_parquet(args.pool); teacher = load_teacher(args.prompts,args.results); train_groups=groups_from_frame(pool,teacher)
    if len(train_groups)!=2000: raise RuntimeError(f"expected 2000 aligned train queries, got {len(train_groups)}")
    dev_groups=groups_from_frame(load_dev(args.dev,args.dev_queries))
    entropies=[]
    for x in train_groups:
        prob=torch.softmax(torch.as_tensor(x["teacher"]),0); entropies.append(float(-(prob*prob.log()).sum()))
    route_threshold=float(np.quantile(entropies,1-args.route_rate))
    device=torch.device("cuda"); tokenizer=AutoTokenizer.from_pretrained(args.model)
    rows=[]
    for seed in args.seeds:
        for variant in ("listwise_only","pairwise_only","joint","routed_joint"):
            rows.append(train_variant(variant,seed,train_groups,dev_groups,tokenizer,args,device,route_threshold))
    by={}
    for variant in ("listwise_only","pairwise_only","joint","routed_joint"):
        subset=[x for x in rows if x["variant"]==variant]
        by[variant]={"mean_ndcg10":float(np.mean([x["overall_ndcg10"] for x in subset])),"seeds":subset}
    base=by["listwise_only"]["mean_ndcg10"]; joint=by["joint"]["mean_ndcg10"]; routed=by["routed_joint"]["mean_ndcg10"]
    joint_gain=joint-base; routed_gain=routed-base
    tail_ok=all(x["buckets"].get("tail",{}).get("ndcg10",0)>=next(y for y in rows if y["variant"]=="listwise_only" and y["seed"]==x["seed"])["buckets"].get("tail",{}).get("ndcg10",0) for x in rows if x["variant"]=="joint")
    acceptance={"joint_gain_at_least_0p005":joint_gain>=.005,"tail_not_reverse_all_seeds":tail_ok,"routed_retains_95pct_joint_gain":joint_gain>0 and routed_gain>=.95*joint_gain,"pairwise_call_reduction_at_least_40pct":args.route_rate<=.6,"two_seed_direction_consistent":all(next(x for x in rows if x["variant"]=="joint" and x["seed"]==s)["overall_ndcg10"]>next(x for x in rows if x["variant"]=="listwise_only" and x["seed"]==s)["overall_ndcg10"] for s in args.seeds)}
    payload={"schema":"mind_r14_4_dual_granularity_pilot_v1","scope":"objective-level pairwise/listwise losses derived from the same FIRST logits; not an independent PRP teacher","train_queries":len(train_groups),"dev_queries":len(dev_groups),"seeds":args.seeds,"route_rate":args.route_rate,"route_entropy_threshold":route_threshold,"variants":by,"joint_gain":joint_gain,"routed_gain":routed_gain,"acceptance":acceptance,"go":all(acceptance.values()),"boundaries":{"large_train_only_for_training":True,"large_dev_evaluation_only":True,"confirm_accessed":False,"large_test_accessed":False},"source_sha256":{str(p):sha256(p) for p in (args.pool,args.prompts,args.results)}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"complete","go":payload["go"],"joint_gain":joint_gain,"routed_gain":routed_gain,"acceptance":acceptance,"report":str(args.output)}))


if __name__ == "__main__": main()
