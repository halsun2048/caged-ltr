"""Measure request-level MiniLM latency and replay recorded FIRST service latency."""

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

from train_mind_r8_2_large_student import BiEncoder, tokenize


def percentile(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(array.mean()),
        "throughput_qps": float(1000 / array.mean()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_latencies(pattern: str) -> list[float]:
    values = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            for line in handle:
                payload = json.loads(line)["payload"]
                if payload.get("status") == "complete" and payload.get("model_inference"):
                    values.append(1000 * (payload["prefill_seconds"] + payload["decoding_seconds"]))
    if not values:
        raise RuntimeError(f"no recorded FIRST inference timings matched {pattern}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/processed/mind_r13_qrels_free_r12/dev.parquet")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="/root/caged-ltr/all-MiniLM-L6-v2")
    parser.add_argument("--first-results", default="runs/mind_r12_0/dev_first/results.jsonl")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--gate-call-rate", type=float, default=0.55)
    parser.add_argument("--report", type=Path, default=Path("reports/experiments/mind_r13_service_latency.json"))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal service benchmark")

    frame = pd.read_parquet(args.pool)
    groups = list(frame.groupby("query_id", sort=False))[: args.queries]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = BiEncoder(args.model).cuda().eval()
    model.load_state_dict(state["model"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    latencies: list[float] = []
    for index, (_, group) in enumerate(groups):
        texts = [str(group.iloc[0]["query"]), *group["passage"].astype(str).tolist()]
        batch = tokenize(tokenizer, texts, torch.device("cuda"), 96)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.embed(batch)
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - started)
        if index >= args.warmup:
            latencies.append(elapsed)
        if args.progress and (index + 1) % 100 == 0:
            print(f"[student] {index + 1}/{len(groups)} latest={elapsed:.2f}ms", flush=True)

    first = first_latencies(args.first_results)
    count = min(len(latencies), len(first))
    routed = int(round(count * args.gate_call_rate))
    # Deterministic replay: every request pays student scoring; the frozen gate's
    # FIRST-routed requests additionally pay the recorded teacher inference.
    gate = np.asarray(latencies[:count])
    gate[:routed] += np.asarray(first[:routed])
    payload = {
        "schema": "mind_r13_service_latency_v1",
        "device": torch.cuda.get_device_name(),
        "requests": len(latencies),
        "warmup_requests": args.warmup,
        "candidate_count_per_request": int(frame.groupby("query_id").size().median()),
        "student": percentile(latencies),
        "first_recorded_model_inference": percentile(first),
        "hard_tail_gate_replay": {**percentile(gate.tolist()), "first_call_rate": args.gate_call_rate},
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "methodology": {
            "student": "synchronous request-level query plus candidate encoding on CUDA",
            "first": "recorded synchronous model prefill plus decoding, cache reads excluded",
            "gate": "student measurement plus deterministic replay of recorded FIRST latency",
        },
        "large_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "report": str(args.report), "student": payload["student"]}))


if __name__ == "__main__":
    main()
