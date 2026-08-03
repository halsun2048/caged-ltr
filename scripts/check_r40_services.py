"""Check real PostgreSQL/Redis/Qdrant endpoints without mutating data."""

from __future__ import annotations

import os
import socket
import urllib.request


def tcp(url: str, default_port: int) -> dict[str, object]:
    host, _, port = url.partition("://")[-1].partition(":")
    host = host.split("/")[0]
    try:
        with socket.create_connection((host, int(port or default_port)), timeout=2):
            return {"status": "up", "endpoint": url}
    except OSError as error:
        return {"status": "down", "endpoint": url, "error": str(error)}


def main() -> None:
    qdrant = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    try:
        with urllib.request.urlopen(qdrant + "/healthz", timeout=2) as response:
            qdrant_result = {"status": "up", "http": response.status, "endpoint": qdrant}
    except Exception as error:  # pragma: no cover - depends on external services
        qdrant_result = {"status": "down", "endpoint": qdrant, "error": str(error)}
    report = {
        "postgres": tcp(os.getenv("POSTGRES_HOST", "127.0.0.1:5432"), 5432),
        "redis": tcp(os.getenv("REDIS_URL", "redis://127.0.0.1:6379"), 6379),
        "qdrant": qdrant_result,
    }
    print(report)
    if any(value["status"] != "up" for value in report.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
