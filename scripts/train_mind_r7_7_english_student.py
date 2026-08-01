"""Train an English MiniLM bi-encoder on the frozen MIND R7.6 package."""

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
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def embed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return nn.functional.normalize(pooled, dim=-1)


def identity(args: argparse.Namespace) -> str:
    excluded = {"resume", "progress", "checkpoint", "best_checkpoint", "report"}
    payload = {key: value for key, value in vars(args).items() if key not in excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def tokenize(tokenizer, texts, device, max_length):
    batch = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def ndcg_at_10(relevance: np.ndarray) -> float:
    ranked = relevance[:10]
    ideal = np.sort(relevance)[::-1][:10]
    discounts = np.log2(np.arange(2, len(ranked) + 2))
    dcg = np.sum((2**ranked - 1) / discounts)
    idcg = np.sum((2**ideal - 1) / discounts)
    return float(dcg / idcg) if idcg else 0.0


@torch.inference_mode()
def embed_texts(model, tokenizer, texts, device, max_length, batch_size, dtype):
    model.eval()
    vectors = []
    for offset in range(0, len(texts), batch_size):
        batch = tokenize(tokenizer, texts[offset : offset + batch_size], device, max_length)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
            vectors.append(model.embed(batch).float().cpu().numpy())
    return np.concatenate(vectors)


@torch.inference_mode()
def evaluate(model, tokenizer, dev, device, max_length, batch_size, dtype):
    query_table = dev[["query_id", "query"]].drop_duplicates("query_id")
    corpus_table = dev[["corpus_id", "passage"]].drop_duplicates("corpus_id")
    started = time.perf_counter()
    query_vectors = embed_texts(
        model,
        tokenizer,
        query_table["query"].tolist(),
        device,
        max_length,
        batch_size,
        dtype,
    )
    corpus_vectors = embed_texts(
        model,
        tokenizer,
        corpus_table["passage"].tolist(),
        device,
        max_length,
        batch_size,
        dtype,
    )
    query_map = dict(zip(query_table.query_id, query_vectors, strict=True))
    corpus_map = dict(zip(corpus_table.corpus_id, corpus_vectors, strict=True))
    ndcgs, hits, reciprocal = [], [], []
    for _, group in dev.groupby("query_id", sort=False):
        query = query_map[group.iloc[0].query_id]
        passages = np.stack([corpus_map[value] for value in group.corpus_id])
        scores = passages @ query
        order = np.argsort(-scores, kind="stable")
        relevance = group.relevance.to_numpy()[order]
        ndcgs.append(ndcg_at_10(relevance))
        relevant = np.flatnonzero(relevance > 0)
        hits.append(float(np.any(relevance[:10] > 0)))
        reciprocal.append(1 / (int(relevant[0]) + 1) if len(relevant) else 0.0)
    elapsed = time.perf_counter() - started
    return {
        "ndcg10": float(np.mean(ndcgs)),
        "hit10": float(np.mean(hits)),
        "mrr": float(np.mean(reciprocal)),
        "latency_ms_per_query": 1_000 * elapsed / len(ndcgs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=Path, default=Path("data/processed/mind_r7_6/train_pairs.parquet")
    )
    parser.add_argument(
        "--dev", type=Path, default=Path("data/processed/mind_r7_6/dev_listwise.parquet")
    )
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mind_r7_7/latest.pt"))
    parser.add_argument(
        "--best-checkpoint", type=Path, default=Path("artifacts/mind_r7_7_english_student.pt")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/experiments/mind_r7_7_english_student.json")
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-pairs", type=int)
    parser.add_argument("--max-dev-queries", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--evaluate-before-training", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this formal run must not silently fall back to CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    train = pd.read_parquet(args.train)
    dev = pd.read_parquet(args.dev)
    if args.max_train_pairs:
        train = train.sample(min(args.max_train_pairs, len(train)), random_state=args.seed)
    if args.max_dev_queries:
        keep = sorted(dev.query_id.unique())[: args.max_dev_queries]
        dev = dev[dev.query_id.isin(keep)].reset_index(drop=True)
    if set(train.query_id) & set(dev.query_id):
        raise RuntimeError("train/dev query overlap")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BiEncoder(args.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": None,
    }[args.precision]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    run_identity = identity(args)
    epoch = batch_offset = global_step = stale = 0
    best_ndcg = -1.0
    history: list[dict[str, object]] = []
    if args.resume and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if state["identity"] != run_identity:
            raise RuntimeError("checkpoint identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        epoch = state["epoch"]
        batch_offset = state["batch_offset"]
        global_step = state["global_step"]
        stale = state["stale"]
        best_ndcg = state["best_ndcg"]
        history = state["history"]
    started = time.time()
    first_loss = last_loss = None
    baseline_metrics = None
    if history and history[0].get("phase") == "pretrained_baseline":
        baseline_metrics = {key: value for key, value in history[0].items() if key != "phase"}
    elif args.evaluate_before_training and epoch == 0:
        if args.progress:
            print("[baseline] evaluating pretrained English MiniLM on full dev", flush=True)
        baseline_metrics = evaluate(
            model,
            tokenizer,
            dev,
            device,
            args.max_length,
            args.eval_batch_size,
            dtype,
        )
        history.append({"phase": "pretrained_baseline", "epoch": 0, **baseline_metrics})
        print(
            f"[baseline done] dev_ndcg10={baseline_metrics['ndcg10']:.6f} "
            f"hit10={baseline_metrics['hit10']:.6f}",
            flush=True,
        )
    while epoch < args.epochs and stale < args.patience:
        model.train()
        indices = np.random.default_rng(args.seed + epoch).permutation(len(train))
        total_batches = math.ceil(len(indices) / args.batch_size)
        for batch_number in range(batch_offset, total_batches):
            chosen = indices[batch_number * args.batch_size : (batch_number + 1) * args.batch_size]
            batch = train.iloc[chosen]
            query = tokenize(tokenizer, batch["query"], device, args.max_length)
            positive = tokenize(tokenizer, batch["positive_passage"], device, args.max_length)
            negative = tokenize(tokenizer, batch["negative_passage"], device, args.max_length)
            with torch.autocast(
                device_type="cuda", dtype=dtype, enabled=device.type == "cuda" and dtype is not None
            ):
                query_vector = model.embed(query)
                positive_vector = model.embed(positive)
                negative_vector = model.embed(negative)
                delta = (query_vector * positive_vector).sum(1) - (
                    query_vector * negative_vector
                ).sum(1)
                loss = nn.functional.softplus(args.margin - delta).mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            value = float(loss.detach())
            first_loss = value if first_loss is None else first_loss
            last_loss = value
            global_step += 1
            next_offset = batch_number + 1
            state = {
                "identity": run_identity,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "batch_offset": next_offset,
                "global_step": global_step,
                "stale": stale,
                "best_ndcg": best_ndcg,
                "history": history,
            }
            if global_step % args.checkpoint_every == 0:
                atomic_save(state, args.checkpoint)
            if args.progress and (batch_number % 10 == 0 or next_offset == total_batches):
                elapsed = time.time() - started
                rate = global_step / max(elapsed, 1e-6)
                remaining_batches = (
                    total_batches - next_offset + (args.epochs - epoch - 1) * total_batches
                )
                eta = remaining_batches / max(rate, 1e-6)
                memory = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
                print(
                    f"epoch={epoch + 1}/{args.epochs} batch={next_offset}/{total_batches} "
                    f"loss={value:.4f} gpu={memory:.1f}GiB elapsed={elapsed / 60:.1f}m "
                    f"eta~{eta / 60:.1f}m",
                    flush=True,
                )
        metrics = evaluate(
            model,
            tokenizer,
            dev,
            device,
            args.max_length,
            args.eval_batch_size,
            dtype,
        )
        history.append({"epoch": epoch + 1, **metrics})
        improved = metrics["ndcg10"] > best_ndcg
        if improved:
            best_ndcg = metrics["ndcg10"]
            stale = 0
            atomic_save(
                {
                    "identity": run_identity,
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "max_length": args.max_length,
                    "epoch": epoch + 1,
                    "metrics": metrics,
                },
                args.best_checkpoint,
            )
        else:
            stale += 1
        epoch += 1
        batch_offset = 0
        state.update(
            {
                "epoch": epoch,
                "batch_offset": 0,
                "stale": stale,
                "best_ndcg": best_ndcg,
                "history": history,
            }
        )
        atomic_save(state, args.checkpoint)
        print(
            f"[epoch done] epoch={epoch} dev_ndcg10={metrics['ndcg10']:.6f} "
            f"hit10={metrics['hit10']:.6f} stale={stale}/{args.patience}",
            flush=True,
        )
    payload = {
        "schema": "mind_r7_7_english_student_v1",
        "identity": run_identity,
        "language": "English",
        "model": args.model,
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "precision": args.precision,
        "train_pairs": len(train),
        "dev_queries": int(dev.query_id.nunique()),
        "epochs_completed": epoch,
        "early_stopped": stale >= args.patience,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_decreased": bool(
            last_loss is not None and first_loss is not None and last_loss < first_loss
        ),
        "best_dev_ndcg10": best_ndcg,
        "pretrained_baseline": baseline_metrics,
        "history": history,
        "elapsed_seconds": round(time.time() - started, 2),
        "checkpoint": str(args.best_checkpoint),
        "checkpoint_sha256": (
            sha256(args.best_checkpoint) if args.best_checkpoint.exists() else None
        ),
        "boundaries": {
            "dev_used_for_early_stopping_only": True,
            "calibration_accessed": False,
            "mind_holdout_accessed": False,
            "nfcorpus_locked_test_accessed": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    args.report.with_suffix(".md").write_text(
        "# R7.7 English MIND MiniLM student\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    print(json.dumps({"stage": "complete", "report": str(args.report), "best_ndcg10": best_ndcg}))


if __name__ == "__main__":
    main()
