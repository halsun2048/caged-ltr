"""Measure real MiniLM reranking latency on a configured CPU/GPU host."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from caged_ltr.r16_service import Candidate, MiniLMBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend = MiniLMBackend(args.model_path, args.checkpoint, device=args.device)
    candidates = [
        Candidate(str(i), f"restaurant recommendation item {i}")
        for i in range(args.candidates)
    ]
    durations = []
    for index in range(args.requests):
        started = time.perf_counter()
        backend.rerank("find a good restaurant downtown", candidates)
        durations.append((time.perf_counter() - started) * 1000)
        if (index + 1) % max(1, args.requests // 20) == 0:
            print(f"[MiniLM] {index + 1}/{args.requests}", flush=True)
    durations.sort()
    def percentile(p: float) -> float:
        return durations[min(len(durations) - 1, int(len(durations) * p) - 1)]
    report = {
        "schema": "caged_ltr_r46_minilm_benchmark_v1",
        "device": str(backend._device),
        "requests": args.requests,
        "candidates": args.candidates,
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "mean_ms": statistics.mean(durations),
        "throughput_qps": 1000.0 / statistics.mean(durations),
    }
    try:
        import torch

        report["peak_memory_mb"] = (
            torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        )
    except Exception:  # pragma: no cover - optional GPU runtime
        report["peak_memory_mb"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": "complete", **report}))


if __name__ == "__main__":
    main()
