"""Evaluate one frozen MiniLM checkpoint on R8 large-dev without touching large-test."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_shards(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def tokenize(tokenizer, texts, device, max_length):
    batch = tokenizer(
        list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def embed(model, tokenizer, texts, device, max_length, batch_size):
    values = []
    for offset in range(0, len(texts), batch_size):
        batch = tokenize(tokenizer, texts[offset : offset + batch_size], device, max_length)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            values.append(model.embed(batch).float().cpu().numpy())
    return np.concatenate(values)


def ndcg(relevance: np.ndarray) -> float:
    ranked = relevance[:10]
    ideal = np.sort(relevance)[::-1][:10]
    discounts = np.log2(np.arange(2, len(ranked) + 2))
    dcg = np.sum((2**ranked - 1) / discounts)
    idcg = np.sum((2**ideal - 1) / discounts)
    return float(dcg / idcg) if idcg else 0.0


def summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "queries": len(frame),
        "ndcg10": float(frame.ndcg10.mean()),
        "hit10": float(frame.hit10.mean()),
        "mrr": float(frame.mrr.mean()),
        "top1_accuracy": float(frame.top1_correct.mean()),
        "mean_margin": float(frame.margin.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="data/processed/mind_r8_1_v2/dev_listwise/*.parquet")
    parser.add_argument("--model", default="/root/caged-ltr/all-MiniLM-L6-v2")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mind_r8_3"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R8 formal evaluation requires CUDA")
    device = torch.device("cuda")
    dev = read_shards(args.dev)
    if set(dev.split) != {"large_dev"}:
        raise RuntimeError("only the frozen large_dev split may be evaluated")
    model = BiEncoder(args.model)
    checkpoint_hash = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        checkpoint_hash = sha256(args.checkpoint)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    queries = dev[["query_id", "query"]].drop_duplicates("query_id")
    passages = dev[["passage"]].drop_duplicates("passage")
    started = time.perf_counter()
    if args.progress:
        print(f"[encode] queries={len(queries):,} unique_passages={len(passages):,}", flush=True)
    qvec = embed(model, tokenizer, queries.query.tolist(), device, args.max_length, args.batch_size)
    pvec = embed(
        model, tokenizer, passages.passage.tolist(), device, args.max_length, args.batch_size
    )
    qmap = dict(zip(queries.query_id, qvec, strict=True))
    pmap = dict(zip(passages.passage, pvec, strict=True))
    rows = []
    for number, (query_id, group) in enumerate(dev.groupby("query_id", sort=False), 1):
        scores = np.stack([pmap[value] for value in group.passage]) @ qmap[query_id]
        order = np.argsort(-scores, kind="stable")
        rel = group.relevance.to_numpy()[order]
        relevant = np.flatnonzero(rel > 0)
        positive_frequency = group.loc[group.relevance > 0, "train_item_frequency"].mean()
        rows.append(
            {
                "query_id": query_id,
                "ndcg10": ndcg(rel),
                "hit10": float(np.any(rel[:10] > 0)),
                "mrr": 1 / (int(relevant[0]) + 1) if len(relevant) else 0.0,
                "top1_correct": float(rel[0] > 0),
                "margin": float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 1.0,
                "positive_frequency": float(positive_frequency),
                "query_characters": int(group.query_characters.iloc[0]),
                "candidate_count": len(group),
            }
        )
        if args.progress and number % 2000 == 0:
            print(f"[score] {number:,}/{len(queries):,}", flush=True)
    query_metrics = pd.DataFrame(rows)
    query_metrics["frequency_bucket"] = pd.qcut(
        query_metrics.positive_frequency.rank(method="first"),
        3,
        labels=["tail", "torso", "head"],
    ).astype(str)
    query_metrics["length_bucket"] = pd.qcut(
        query_metrics.query_characters.rank(method="first"),
        3,
        labels=["short", "medium", "long"],
    ).astype(str)
    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / f"{args.name}_query_metrics.parquet"
    query_metrics.to_parquet(predictions, index=False)
    report_path = args.report or Path(f"reports/experiments/mind_r8_3_{args.name}.json")
    payload = {
        "schema": "mind_r8_3_large_dev_evaluation_v1",
        "name": args.name,
        "split": "large_dev",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_hash,
        "overall": summarize(query_metrics),
        "frequency_buckets": {
            key: summarize(value) for key, value in query_metrics.groupby("frequency_bucket")
        },
        "length_buckets": {
            key: summarize(value) for key, value in query_metrics.groupby("length_bucket")
        },
        "efficiency": {
            "elapsed_seconds": elapsed,
            "latency_ms_per_query_amortized": elapsed * 1000 / len(query_metrics),
            "throughput_queries_per_second": len(query_metrics) / elapsed,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "checkpoint_bytes": args.checkpoint.stat().st_size if args.checkpoint else None,
            "unique_passages_encoded": len(passages),
        },
        "query_metrics": str(predictions),
        "boundaries": {"large_dev_accessed": True, "large_test_accessed": False},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    result = {"stage": "complete", "report": str(report_path), "overall": payload["overall"]}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
