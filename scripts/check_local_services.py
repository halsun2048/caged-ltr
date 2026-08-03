"""Validate CPU-only embedded substitutes when Docker daemons are unavailable."""

from __future__ import annotations

import json
from pathlib import Path

from caged_ltr.runtime_state import MemoryState
from caged_ltr.storage import EventStore


def main() -> None:
    root = Path("runs/local_services")
    root.mkdir(parents=True, exist_ok=True)
    store = EventStore(root / "postgres_compatible.sqlite3")
    event = store.record_search({"query": "local", "route": {"backend": "student"}})
    store.record_feedback(event, "local-user", "item-1", "click")
    postgres_compatible = store.summary()
    store.close()
    state = MemoryState()
    state.set("health", "ok")
    redis_compatible = {"value": state.get("health"), "counter": state.incr("requests")}
    qdrant_local = {"status": "unavailable", "reason": "qdrant-client not installed"}
    try:
        from qdrant_client import QdrantClient, models

        client = QdrantClient(path=str(root / "qdrant"))
        if not client.collection_exists("caged-local"):
            client.create_collection(
                collection_name="caged-local",
                vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
            )
        client.upsert(
            collection_name="caged-local",
            points=[models.PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"item_id": "item-1"})],
        )
        result = client.query_points(collection_name="caged-local", query=[1.0, 0.0, 0.0], limit=1)
        qdrant_local = {"status": "up", "top_item": result.points[0].payload["item_id"]}
        client.close()
    except ImportError:
        pass
    report = {
        "schema": "caged_ltr_r48_local_embedded_v1",
        "postgres_compatible_sqlite": postgres_compatible,
        "redis_compatible_memory": redis_compatible,
        "qdrant_local": qdrant_local,
        "production_services_started": False,
        "limitation": "SQLite/MemoryState/Qdrant local are CPU fallbacks, not PostgreSQL/Redis/Qdrant daemons.",
    }
    path = Path("reports/experiments/r48_local_embedded.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "report": str(path), **report}))


if __name__ == "__main__":
    main()
