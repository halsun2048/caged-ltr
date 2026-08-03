"""Small event-store abstraction with SQLite default and PostgreSQL-ready schema."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS search_events (
  event_id TEXT PRIMARY KEY, user_id TEXT, query TEXT NOT NULL,
  backend TEXT NOT NULL, route_mode TEXT, latency_ms REAL,
  first_called INTEGER NOT NULL DEFAULT 0, payload_json TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback_events (
  event_id TEXT PRIMARY KEY, search_event_id TEXT, user_id TEXT,
  item_id TEXT, feedback TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_search_events_created ON search_events(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_events_search ON feedback_events(search_event_id);
"""


class EventStore:
    """SQLite development store; the schema is intentionally portable to PostgreSQL."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or os.getenv("CAGED_EVENT_DB", "runs/caged_events.sqlite3")
        self.path = Path(configured)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path, check_same_thread=False, timeout=30.0
        )
        self._connection.execute("PRAGMA journal_mode=WAL")  # pragma: no cover - sqlite setup
        self._connection.execute("PRAGMA busy_timeout=30000")  # pragma: no cover - sqlite setup
        self._connection.executescript(SCHEMA)
        self._connection.commit()
        self._lock = threading.RLock()

    def record_search(self, payload: dict[str, Any]) -> str:
        event_id = str(uuid.uuid4())
        route = payload.get("route", {})
        with self._lock:
            self._connection.execute(
                "INSERT INTO search_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    payload.get("user_id"),
                    payload["query"],
                    payload.get("backend", "unknown"),
                    route.get("mode"),
                    payload.get("latency_ms"),
                    int(route.get("backend") == "first"),
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                ),
            )
            self._connection.commit()
        return event_id

    def record_feedback(
        self, search_event_id: str, user_id: str | None, item_id: str, feedback: str
    ) -> str:
        if feedback not in {"click", "long_click", "dismiss", "like", "dislike"}:
            raise ValueError("unsupported feedback type")
        event_id = str(uuid.uuid4())
        with self._lock:
            self._connection.execute(
                "INSERT INTO feedback_events VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, search_event_id, user_id, item_id, feedback, time.time()),
            )
            self._connection.commit()
        return event_id

    def summary(self) -> dict[str, int]:
        with self._lock:
            searches = self._connection.execute("SELECT COUNT(*) FROM search_events").fetchone()[0]
            feedback = self._connection.execute(
                "SELECT COUNT(*) FROM feedback_events"
            ).fetchone()[0]
            first_calls = self._connection.execute(
                "SELECT COALESCE(SUM(first_called), 0) FROM search_events"
            ).fetchone()[0]
        return {"search_events": searches, "feedback_events": feedback, "first_calls": first_calls}

    def close(self) -> None:
        self._connection.close()
