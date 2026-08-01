"""Offline FP32/INT8 compression audit for the frozen R6 MiniLM student."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.ao.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig
from transformers import BertConfig, BertModel


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Encoder(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        config = BertConfig(
            vocab_size=state["encoder.embeddings.word_embeddings.weight"].shape[0],
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=1536,
            max_position_embeddings=512,
            type_vocab_size=2,
        )
        self.encoder = BertModel(config)
        self.load_state_dict(state, strict=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_type_ids = torch.zeros_like(input_ids).contiguous()
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return nn.functional.normalize(pooled, dim=-1)


def hashed_tokens(
    texts: list[str], vocab_size: int, max_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    for text in texts:
        words = re.findall(r"[A-Za-z0-9]+", text.lower())[: max_length - 2]
        ids = [0]
        for word in words:
            value = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
            ids.append(1 + value % (vocab_size - 2))
        ids.append(2)
        rows.append(ids)
    width = min(max(len(row) for row in rows), max_length)
    input_ids = torch.zeros((len(rows), width), dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        row = row[:width]
        input_ids[index, : len(row)] = torch.tensor(row)
        attention[index, : len(row)] = 1
    return input_ids, attention


@torch.inference_mode()
def encode(
    model: nn.Module, texts: list[str], vocab: int, length: int, batch_size: int
) -> np.ndarray:
    output = []
    for offset in range(0, len(texts), batch_size):
        ids, mask = hashed_tokens(texts[offset : offset + batch_size], vocab, length)
        output.append(model(ids, mask).numpy())
    return np.concatenate(output)


@torch.inference_mode()
def latency(
    model: nn.Module, texts: list[str], vocab: int, length: int, batch_size: int
) -> dict[str, float]:
    batches = []
    for offset in range(0, len(texts), batch_size):
        batches.append(hashed_tokens(texts[offset : offset + batch_size], vocab, length))
    for ids, mask in batches[:2]:
        model(ids, mask)
    timings = []
    for ids, mask in batches:
        started = time.perf_counter()
        model(ids, mask)
        timings.append(time.perf_counter() - started)
    total = sum(timings)
    return {
        "batch_size": batch_size,
        "texts": len(texts),
        "milliseconds_per_text": 1_000 * total / len(texts),
        "texts_per_second": len(texts) / total,
        "p50_batch_ms": 1_000 * float(np.median(timings)),
        "p95_batch_ms": 1_000 * float(np.quantile(timings, 0.95)),
    }


def ndcg(relevance: np.ndarray, order: np.ndarray) -> float:
    ranked = relevance[order][:10]
    ideal = np.sort(relevance)[::-1][:10]
    discounts = np.log2(np.arange(2, len(ranked) + 2))
    dcg = np.sum((2**ranked - 1) / discounts)
    idcg = np.sum((2**ideal - 1) / discounts)
    return float(dcg / idcg) if idcg else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/nfcorpus_r6_student_best.pt")
    )
    parser.add_argument(
        "--dev", type=Path, default=Path("data/processed/nfcorpus_r6/dev_listwise.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/experiments/nfcorpus_r7_4_compression.json")
    )
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    if args.progress:
        print("[1/5] reconstructing frozen MiniLM encoder", flush=True)
    fp32 = Encoder(state).eval()
    if args.progress:
        print("[2/5] applying dynamic INT8 quantization", flush=True)
    int8 = torch.ao.quantization.quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8).eval()
    int8_full = torch.ao.quantization.quantize_dynamic(
        fp32,
        {
            nn.Linear: default_dynamic_qconfig,
            nn.Embedding: float_qparams_weight_only_qconfig,
        },
        dtype=torch.qint8,
    ).eval()
    dev = pd.read_parquet(args.dev)
    query_ids = sorted(dev.query_id.astype(str).unique())[: args.queries]
    sample = dev[dev.query_id.astype(str).isin(query_ids)].copy()
    query_texts = sample.groupby("query_id", sort=False)["query"].first().tolist()
    passage_texts = sample["passage"].astype(str).tolist()
    texts = (query_texts + passage_texts)[:512]
    vocab = state["encoder.embeddings.word_embeddings.weight"].shape[0]
    if args.progress:
        print(
            f"[3/5] scoring numerical/ranking preservation on {len(query_ids)} dev queries",
            flush=True,
        )
    fp_query = encode(fp32, query_texts, vocab, args.max_length, 16)
    q8_query = encode(int8, query_texts, vocab, args.max_length, 16)
    q8_full_query = encode(int8_full, query_texts, vocab, args.max_length, 16)
    fp_passage = encode(fp32, passage_texts, vocab, args.max_length, 16)
    q8_passage = encode(int8, passage_texts, vocab, args.max_length, 16)
    q8_full_passage = encode(int8_full, passage_texts, vocab, args.max_length, 16)
    score_rows = []
    position = 0
    for index, (_, group) in enumerate(sample.groupby("query_id", sort=False)):
        count = len(group)
        relevance = group["graded_relevance"].to_numpy(dtype=float)
        fp_scores = fp_passage[position : position + count] @ fp_query[index]
        q8_scores = q8_passage[position : position + count] @ q8_query[index]
        q8_full_scores = q8_full_passage[position : position + count] @ q8_full_query[index]
        fp_order = np.argsort(-fp_scores, kind="stable")
        q8_order = np.argsort(-q8_scores, kind="stable")
        q8_full_order = np.argsort(-q8_full_scores, kind="stable")
        score_rows.append(
            {
                "fp32_ndcg10": ndcg(relevance, fp_order),
                "int8_ndcg10": ndcg(relevance, q8_order),
                "top10_overlap": len(set(fp_order[:10]) & set(q8_order[:10])) / 10,
                "score_mae": float(np.mean(np.abs(fp_scores - q8_scores))),
                "full_int8_ndcg10": ndcg(relevance, q8_full_order),
                "full_int8_top10_overlap": len(set(fp_order[:10]) & set(q8_full_order[:10])) / 10,
                "full_int8_score_mae": float(np.mean(np.abs(fp_scores - q8_full_scores))),
            }
        )
        position += count
    if args.progress:
        print("[4/5] benchmarking CPU batch=1/8", flush=True)
    benchmark_texts = texts[:128]
    latencies = {
        "fp32_batch1": latency(fp32, benchmark_texts, vocab, args.max_length, 1),
        "fp32_batch8": latency(fp32, benchmark_texts, vocab, args.max_length, 8),
        "int8_batch1": latency(int8, benchmark_texts, vocab, args.max_length, 1),
        "int8_batch8": latency(int8, benchmark_texts, vocab, args.max_length, 8),
        "full_int8_batch1": latency(int8_full, benchmark_texts, vocab, args.max_length, 1),
        "full_int8_batch8": latency(int8_full, benchmark_texts, vocab, args.max_length, 8),
    }
    with tempfile.TemporaryDirectory(prefix="caged-r7-4-") as directory:
        fp_path = Path(directory) / "fp32.pt"
        int8_path = Path(directory) / "int8.pt"
        int8_full_path = Path(directory) / "int8_full.pt"
        torch.save(fp32.state_dict(), fp_path)
        torch.save(int8.state_dict(), int8_path)
        torch.save(int8_full.state_dict(), int8_full_path)
        sizes = {
            "fp32_bytes": fp_path.stat().st_size,
            "int8_bytes": int8_path.stat().st_size,
            "full_int8_bytes": int8_full_path.stat().st_size,
        }
    frame = pd.DataFrame(score_rows)
    payload = {
        "schema": "nfcorpus_r7_4_compression_v1",
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
        "scope": "independent dev only; untouched test was not accessed",
        "language": "English input texts only",
        "device": "cpu",
        "threads": args.threads,
        "quantization": "PyTorch eager dynamic INT8 on Linear layers",
        "quality_method": (
            "Deterministic hash-token proxy is used because tokenizer assets are unavailable "
            "locally; "
            "this validates numerical/ranking preservation, not absolute retrieval quality."
        ),
        "dev_queries": len(score_rows),
        "preservation": {key: float(value) for key, value in frame.mean().items()},
        "latency": latencies,
        "storage": {
            **sizes,
            "compression_ratio": sizes["fp32_bytes"] / sizes["int8_bytes"],
            "reduction_percent": 100 * (1 - sizes["int8_bytes"] / sizes["fp32_bytes"]),
            "full_int8_compression_ratio": sizes["fp32_bytes"] / sizes["full_int8_bytes"],
            "full_int8_reduction_percent": 100
            * (1 - sizes["full_int8_bytes"] / sizes["fp32_bytes"]),
        },
        "acceptance": {
            "test_not_accessed": True,
            "deployment_admitted": False,
            "int8_size_reduction_at_least_40_percent": sizes["int8_bytes"]
            <= 0.6 * sizes["fp32_bytes"],
            "mean_top10_overlap_at_least_0p95": float(frame.top10_overlap.mean()) >= 0.95,
            "mean_absolute_ndcg_drift_at_most_0p005": abs(
                float((frame.int8_ndcg10 - frame.fp32_ndcg10).mean())
            )
            <= 0.005,
            "int8_batch1_faster": latencies["int8_batch1"]["milliseconds_per_text"]
            < latencies["fp32_batch1"]["milliseconds_per_text"],
            "full_int8_size_reduction_at_least_40_percent": sizes["full_int8_bytes"]
            <= 0.6 * sizes["fp32_bytes"],
            "full_int8_mean_top10_overlap_at_least_0p95": float(
                frame.full_int8_top10_overlap.mean()
            )
            >= 0.95,
            "full_int8_mean_absolute_ndcg_drift_at_most_0p005": abs(
                float((frame.full_int8_ndcg10 - frame.fp32_ndcg10).mean())
            )
            <= 0.005,
        },
        "onnx_status": (
            "not_run: onnx/onnxruntime are not project dependencies; dynamic INT8 is the "
            "admitted no-new-dependency path"
        ),
        "decision": (
            "Reject both INT8 variants for deployment: speed/storage improve, but the frozen "
            "0.95 top-10 overlap requirement is not met."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(
        "# R7.4 MiniLM compression and CPU efficiency\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    if args.progress:
        print("[5/5] report written", flush=True)
    print(
        json.dumps(
            {"stage": "complete", "report": str(args.output), "acceptance": payload["acceptance"]}
        )
    )


if __name__ == "__main__":
    main()
