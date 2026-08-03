"""R46 lightweight API benchmark for cached or real deployments."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


def request(base_url: str, candidate_count: int) -> float:
    candidates = [
        {"item_id": str(index), "text": f"restaurant guide item {index}"}
        for index in range(candidate_count)
    ]
    payload = {"query": "restaurant guide", "backend": "gate", "candidates": candidates}
    started = time.perf_counter()
    req = Request(
        base_url.rstrip("/") + "/search",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urlopen(req, timeout=30):
        pass
    return (time.perf_counter() - started) * 1000


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * quantile))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--output", default="reports/experiments/r46_api_benchmark.json")
    args = parser.parse_args()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        latencies = list(
            executor.map(lambda _: request(args.base_url, args.candidates), range(args.requests))
        )
    elapsed = time.perf_counter() - started
    result = {
        "schema": "caged_ltr_r46_api_benchmark_v1",
        "base_url": args.base_url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "candidates": args.candidates,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "mean_ms": statistics.mean(latencies),
        "throughput_qps": args.requests / elapsed,
        "mode": "deployment-dependent; do not compare cached and real as equivalent",
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"stage": "complete", **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
