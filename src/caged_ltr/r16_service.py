"""Small, demo-ready serving layer for the CAGED-LTR pipeline.

The service keeps model adapters separate from HTTP concerns.  The cached adapter
is intentionally deterministic so the demo remains usable after the GPU is
released; a real Student/FIRST adapter can be plugged into the same interface.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Candidate:
    item_id: str
    text: str


@dataclass(frozen=True)
class RankedCandidate:
    item_id: str
    text: str
    score: float
    rank: int


class RerankBackend(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[Candidate]) -> list[RankedCandidate]: ...


def lexical_score(query: str, text: str) -> float:
    query_terms = set(query.lower().split())
    text_terms = set(text.lower().split())
    return len(query_terms & text_terms) / max(len(query_terms | text_terms), 1)


class CachedBackend:
    """Deterministic local backend used for offline demos and API tests."""

    name = "cached-student"

    def rerank(self, query: str, candidates: list[Candidate]) -> list[RankedCandidate]:
        scored = sorted(candidates, key=lambda item: (-lexical_score(query, item.text), item.item_id))
        return [RankedCandidate(item.item_id, item.text, lexical_score(query, item.text), rank) for rank, item in enumerate(scored, 1)]


class SearchService:
    def __init__(self, student: RerankBackend | None = None, first: RerankBackend | None = None, first_budget: float = 0.4) -> None:
        self.student = student or CachedBackend()
        self.first = first or CachedBackend()
        self.first_budget = first_budget
        self.requests = 0
        self.first_calls = 0
        self.total_ms = 0.0

    def route(self, query: str, candidates: list[Candidate]) -> dict[str, object]:
        digest = hashlib.sha256(query.encode()).digest()
        use_first = int.from_bytes(digest[:8], "big") / 2**64 < self.first_budget
        return {"backend": "first" if use_first else "student", "reason": "stable-budget-route", "budget": self.first_budget}

    def search(self, query: str, candidates: list[Candidate], backend: str = "gate") -> dict[str, object]:
        started = time.perf_counter()
        decision = self.route(query, candidates) if backend == "gate" else {"backend": backend, "reason": "explicit-backend"}
        selected = decision["backend"]
        if selected == "first":
            result = self.first.rerank(query, candidates)
            self.first_calls += 1
        elif selected == "student":
            result = self.student.rerank(query, candidates)
        else:
            raise ValueError(f"unknown backend: {selected}")
        elapsed = 1000 * (time.perf_counter() - started)
        self.requests += 1
        self.total_ms += elapsed
        return {"query": query, "backend": selected, "route": decision, "results": [item.__dict__ for item in result], "latency_ms": elapsed}

    def metrics(self) -> dict[str, object]:
        return {"requests": self.requests, "first_calls": self.first_calls, "first_call_rate": self.first_calls / max(self.requests, 1), "mean_latency_ms": self.total_ms / max(self.requests, 1)}


def create_app(service: SearchService | None = None):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as error:  # pragma: no cover - exercised when app extra is absent
        raise RuntimeError("install the 'app' extra to run the HTTP demo: uv sync --extra app") from error

    class CandidateInput(BaseModel):
        item_id: str
        text: str

    class SearchInput(BaseModel):
        query: str = Field(min_length=1)
        candidates: list[CandidateInput] = Field(min_length=1, max_length=100)
        backend: str = "gate"

    app = FastAPI(title="CAGED-LTR R16 Demo", version="0.1.0")
    runtime = service or SearchService()

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "student": runtime.student.name, "first": runtime.first.name}

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        return runtime.metrics()

    @app.post("/route")
    def route(payload: SearchInput) -> dict[str, object]:
        candidates = [Candidate(item.item_id, item.text) for item in payload.candidates]
        return runtime.route(payload.query, candidates)

    @app.post("/search")
    def search(payload: SearchInput) -> dict[str, object]:
        candidates = [Candidate(item.item_id, item.text) for item in payload.candidates]
        return runtime.search(payload.query, candidates, payload.backend)

    return app


try:
    app = create_app()
except RuntimeError:
    # Core ranking utilities remain importable without installing the demo extra.
    app = None
