"""Train/resume an English NFCorpus MiniLM bi-encoder from the MIND checkpoint."""
from __future__ import annotations

import argparse
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
    def __init__(self, name: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)

    def embed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return nn.functional.normalize(pooled, dim=-1)


def encode(tokenizer, texts, device, max_length):
    return {
        key: value.to(device)
        for key, value in tokenizer(
            list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).items()
    }


def ndcg_at_10(relevance: list[float]) -> float:
    values = relevance[:10]
    dcg = sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values))
    ideal = sorted(relevance, reverse=True)[:10]
    idcg = sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def identity(args) -> str:
    payload = {key: value for key, value in vars(args).items() if key not in {"resume", "stop_after_batches", "progress"}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def atomic_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(model, tokenizer, dev, device, max_length, batch_size):
    model.eval()
    query_text = dev.groupby("query_id")["query"].first().to_dict()
    query_ids = list(query_text)
    query_vectors = {}
    start = time.perf_counter()
    for offset in range(0, len(query_ids), batch_size):
        ids = query_ids[offset : offset + batch_size]
        vectors = model.embed(encode(tokenizer, [query_text[q] for q in ids], device, max_length)).cpu()
        query_vectors.update(zip(ids, vectors))
    passage_scores = []
    passages = dev["passage"].tolist()
    for offset in range(0, len(passages), batch_size):
        chunk = dev.iloc[offset : offset + batch_size]
        passage_vectors = model.embed(encode(tokenizer, chunk["passage"], device, max_length)).cpu()
        query_batch = torch.stack([query_vectors[q] for q in chunk["query_id"]])
        passage_scores.extend((query_batch * passage_vectors).sum(1).tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000 / len(query_ids)
    scored = dev.copy()
    scored["score"] = passage_scores
    overall, buckets = [], {name: [] for name in ("head", "torso", "tail")}
    for _, group in scored.groupby("query_id"):
        ranked = group.sort_values("score", ascending=False)
        overall.append(ndcg_at_10(ranked["graded_relevance"].tolist()))
        for name in buckets:
            filtered = ranked["frequency_bucket"] == name
            if (ranked.loc[filtered, "graded_relevance"] > 0).any():
                relevance = np.where(filtered, ranked["graded_relevance"], 0.0).tolist()
                buckets[name].append(ndcg_at_10(relevance))
    return {
        "ndcg10": float(np.mean(overall)),
        "head_ndcg10": float(np.mean(buckets["head"])) if buckets["head"] else None,
        "torso_ndcg10": float(np.mean(buckets["torso"])) if buckets["torso"] else None,
        "tail_ndcg10": float(np.mean(buckets["tail"])) if buckets["tail"] else None,
        "bucket_query_counts": {key: len(value) for key, value in buckets.items()},
        "latency_ms_per_query": latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=Path)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--loss", choices=("pairwise", "soft_margin"), default="pairwise")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    parser.add_argument("--teacher-temperature", type=float, default=4.0)
    parser.add_argument("--max-train-pairs", type=int)
    parser.add_argument("--max-dev-queries", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--stop-after-batches", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = pd.read_parquet(args.pairs)
    if args.hard_negatives:
        pairs = pd.concat([pairs, pd.read_parquet(args.hard_negatives)], ignore_index=True)
    if args.max_train_pairs:
        pairs = pairs.sample(min(args.max_train_pairs, len(pairs)), random_state=args.seed).reset_index(drop=True)
    dev = pd.read_parquet(args.dev)
    if args.max_dev_queries:
        keep = sorted(dev["query_id"].unique())[: args.max_dev_queries]
        dev = dev[dev["query_id"].isin(keep)].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BiEncoder(args.model)
    initial = torch.load(args.initial_checkpoint, map_location="cpu")
    model.load_state_dict(initial["model"], strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    run_identity = identity(args)
    epoch, batch_offset, global_batches = 0, 0, 0
    best_ndcg, best_epoch, history = -1.0, -1, []
    if args.resume and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        if state["identity"] != run_identity:
            raise RuntimeError("checkpoint identity mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        epoch, batch_offset, global_batches = state["epoch"], state["batch_offset"], state["global_batches"]
        best_ndcg, best_epoch, history = state["best_ndcg"], state["best_epoch"], state["history"]
    first_loss, last_loss, max_gradient = None, None, 0.0
    started = time.time()
    while epoch < args.epochs:
        model.train()
        indices = list(range(len(pairs))); random.Random(args.seed + epoch).shuffle(indices)
        total_batches = math.ceil(len(indices) / args.batch_size)
        for batch_number in range(batch_offset, total_batches):
            selected = indices[batch_number * args.batch_size : (batch_number + 1) * args.batch_size]
            batch = pairs.iloc[selected]
            query = model.embed(encode(tokenizer, batch["query"], device, args.max_length))
            positive = model.embed(encode(tokenizer, batch["positive_passage"], device, args.max_length))
            negative = model.embed(encode(tokenizer, batch["negative_passage"], device, args.max_length))
            score_delta = (query * positive).sum(1) - (query * negative).sum(1)
            if args.loss == "soft_margin":
                target = torch.sigmoid(torch.tensor(batch["teacher_margin"].to_numpy(), device=device, dtype=torch.float32) / args.teacher_temperature)
                loss_row = nn.functional.binary_cross_entropy_with_logits(score_delta, target, reduction="none")
            else:
                loss_row = nn.functional.softplus(-score_delta)
            weights = torch.ones_like(loss_row)
            if args.tail_weight != 1.0:
                tail = (batch["positive_frequency_bucket"].to_numpy() == "tail")
                weights[torch.tensor(tail, device=device)] = args.tail_weight
            loss = (loss_row * weights).sum() / weights.sum()
            optimizer.zero_grad(); loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); max_gradient = max(max_gradient, grad)
            optimizer.step(); value = float(loss.detach().cpu()); first_loss = value if first_loss is None else first_loss; last_loss = value
            global_batches += 1; next_offset = batch_number + 1
            payload = {"identity": run_identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "batch_offset": next_offset, "global_batches": global_batches, "best_ndcg": best_ndcg, "best_epoch": best_epoch, "history": history}
            if global_batches % args.checkpoint_every == 0:
                atomic_save(payload, args.checkpoint)
            if args.progress and (batch_number % 10 == 0 or next_offset == total_batches):
                elapsed = time.time() - started; rate = global_batches / max(elapsed, 1e-6); remaining = (total_batches-next_offset + (args.epochs-epoch-1)*total_batches)/max(rate,1e-6)
                print(f"epoch={epoch+1}/{args.epochs} batch={next_offset}/{total_batches} loss={value:.4f} elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m", flush=True)
            if args.stop_after_batches and global_batches >= args.stop_after_batches:
                atomic_save(payload, args.checkpoint)
                print(json.dumps({"stage":"intentional_stop","global_batches":global_batches,"checkpoint":str(args.checkpoint)})); return
        metrics = evaluate(model, tokenizer, dev, device, args.max_length, args.batch_size)
        history.append({"epoch": epoch + 1, **metrics})
        if metrics["ndcg10"] > best_ndcg:
            best_ndcg, best_epoch = metrics["ndcg10"], epoch + 1
            atomic_save({"identity":run_identity,"model":model.state_dict(),"metrics":metrics,"epoch":best_epoch}, args.best_checkpoint)
        epoch += 1; batch_offset = 0
        atomic_save({"identity":run_identity,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"batch_offset":0,"global_batches":global_batches,"best_ndcg":best_ndcg,"best_epoch":best_epoch,"history":history}, args.checkpoint)
        print(f"[epoch done] epoch={epoch} dev_ndcg10={metrics['ndcg10']:.6f} tail={metrics['tail_ndcg10']}", flush=True)
    report = {"schema":"nfcorpus_r6_student_v1","identity":run_identity,"device":str(device),"train_pairs":len(pairs),"dev_queries":int(dev.query_id.nunique()),"first_loss":first_loss,"last_loss":last_loss,"loss_decreased":bool(last_loss < first_loss),"max_gradient_norm":max_gradient,"finite_gradient":bool(np.isfinite(max_gradient)),"best_epoch":best_epoch,"best_dev_ndcg10":best_ndcg,"history":history,"resume_supported":True,"test_accessed":False,"elapsed_seconds":round(time.time()-started,2)}
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n"); print(json.dumps({"stage":"complete","report":str(args.report),"best_ndcg10":best_ndcg}))


if __name__ == "__main__":
    main()
