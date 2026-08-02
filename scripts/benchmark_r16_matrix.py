"""Run the R16 MiniLM serving matrix on CUDA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from train_mind_r8_2_large_student import BiEncoder, tokenize
from transformers import AutoTokenizer


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(array.mean()),
        "qps": float(1000 / array.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/experiments/r16_gpu_matrix.json")
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BiEncoder(str(args.model)).to(device).eval()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    )
    torch.cuda.synchronize()
    cold_ms = 1000 * (time.perf_counter() - started)
    groups = list(pd.read_parquet(args.pool).groupby("query_id", sort=False))[: args.queries]
    results = []
    for candidate_count in (10, 20, 50):
        for batch_size in (1, 4, 8, 16):
            values = []
            for index, (_, group) in enumerate(groups):
                texts = [
                    str(group.iloc[0].query),
                    *group.passage.astype(str).tolist()[:candidate_count],
                ]
                torch.cuda.synchronize()
                tic = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for offset in range(0, len(texts), batch_size):
                        model.embed(
                            tokenize(tokenizer, texts[offset : offset + batch_size], device, 96)
                        )
                torch.cuda.synchronize()
                elapsed = 1000 * (time.perf_counter() - tic)
                if index >= args.warmup:
                    values.append(elapsed)
                if args.progress and (index + 1) % max(args.queries // 2, 1) == 0:
                    print(
                        f"[R16 matrix] candidates={candidate_count} batch={batch_size} "
                        f"{index + 1}/{len(groups)}",
                        flush=True,
                    )
            results.append(
                {
                    "candidate_count": candidate_count,
                    "batch_size": batch_size,
                    **summary(values),
                    "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                }
            )
    payload = {
        "schema": "caged_ltr_r16_gpu_matrix_v1",
        "device": torch.cuda.get_device_name(),
        "cold_start_ms": cold_ms,
        "queries": len(groups),
        "warmup": args.warmup,
        "results": results,
        "large_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "report": str(args.report), "cold_start_ms": cold_ms}))


if __name__ == "__main__":
    main()
