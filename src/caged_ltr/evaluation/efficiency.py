"""Local inference latency and model-size measurements."""

from __future__ import annotations

import resource
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class EfficiencyResult:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    peak_python_memory_bytes: int
    process_max_rss_bytes: int


def count_parameters(model: torch.nn.Module, *, trainable_only: bool = False) -> int:
    parameters = (
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )
    return sum(parameter.numel() for parameter in parameters)


def benchmark_inference(
    predict: Callable[[], Any],
    *,
    examples_per_call: int,
    warmup: int = 5,
    repeats: int = 30,
) -> EfficiencyResult:
    """Measure synchronous inference; deployment hardware must be benchmarked separately."""
    if examples_per_call <= 0 or warmup < 0 or repeats < 2:
        raise ValueError("invalid benchmark arguments")
    for _ in range(warmup):
        predict()

    latencies: list[float] = []
    tracemalloc.start()
    for _ in range(repeats):
        started = time.perf_counter_ns()
        predict()
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    values = np.asarray(latencies)
    total_seconds = float(values.sum() / 1_000.0)
    return EfficiencyResult(
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        qps=float(examples_per_call * repeats / total_seconds),
        peak_python_memory_bytes=peak_memory,
        process_max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    )
