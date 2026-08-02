"""Small HTTP smoke/stress test for the R16 API."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


def call(url: str, payload: dict[str, object]) -> float:
    body = json.dumps(payload).encode()
    started = time.perf_counter()
    with urlopen(
        Request(url, data=body, headers={"content-type": "application/json"}), timeout=10
    ) as response:
        if response.status != 200:
            raise RuntimeError(response.status)
        response.read()
    return 1000 * (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/search")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    payload = {
        "query": "best restaurants downtown",
        "backend": "gate",
        "candidates": [
            {"item_id": "A", "text": "best restaurants downtown"},
            {"item_id": "B", "text": "home cooking guide"},
        ],
    }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        values = list(pool.map(lambda _: call(args.url, payload), range(args.requests)))
    print(
        json.dumps(
            {
                "requests": len(values),
                "workers": args.workers,
                "mean_ms": statistics.mean(values),
                "p50_ms": statistics.median(values),
                "p99_ms": sorted(values)[int(0.99 * len(values)) - 1],
                "qps": 1000 * len(values) / sum(values),
            }
        )
    )


if __name__ == "__main__":
    main()
