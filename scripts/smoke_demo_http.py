"""Dependency-light end-to-end smoke test for the local CAGED-LTR API."""

from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


def get(base_url: str, path: str) -> dict:
    with urlopen(base_url.rstrip("/") + path, timeout=5) as response:
        return json.loads(response.read())


def post(base_url: str, path: str, payload: dict) -> dict:
    request = Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    payload = {
        "query": "best restaurants near downtown",
        "user_id": "portfolio-smoke-user",
        "backend": "gate",
        "candidates": [
            {"item_id": "A", "text": "Best restaurants near downtown"},
            {"item_id": "B", "text": "A guide to home cooking"},
            {"item_id": "C", "text": "Downtown travel and dining"},
        ],
    }
    health = get(args.base_url, "/health")
    search = post(args.base_url, "/search", payload)
    first_ab = post(args.base_url, "/ab/search", payload)
    second_ab = post(args.base_url, "/ab/search", payload)
    metrics = get(args.base_url, "/metrics")
    checks = {
        "health_ok": health.get("status") == "ok",
        "ranked_all_candidates": len(search.get("results", [])) == 3,
        "route_exposed": "route" in search,
        "stable_ab_assignment": first_ab.get("arm") == second_ab.get("arm"),
        "metrics_incremented": metrics.get("requests", 0) >= 3,
    }
    if not all(checks.values()):
        raise SystemExit(json.dumps({"stage": "failed", "checks": checks}))
    print(json.dumps({"stage": "complete", "checks": checks, "health": health}))


if __name__ == "__main__":
    main()
