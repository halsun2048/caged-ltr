# ruff: noqa: B008
"""Small, demo-ready serving layer for the CAGED-LTR pipeline.

The service keeps model adapters separate from HTTP concerns.  The cached adapter
is intentionally deterministic so the demo remains usable after the GPU is
released; a real Student/FIRST adapter can be plugged into the same interface.
"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
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
        scored = sorted(
            candidates, key=lambda item: (-lexical_score(query, item.text), item.item_id)
        )
        return [
            RankedCandidate(item.item_id, item.text, lexical_score(query, item.text), rank)
            for rank, item in enumerate(scored, 1)
        ]


class MiniLMBackend:
    """CUDA/CPU MiniLM adapter backed by the frozen R13 BiEncoder checkpoint."""

    name = "minilm"

    def __init__(self, model_path: str, checkpoint: str, device: str = "cuda") -> None:
        import torch
        from train_mind_r8_2_large_student import BiEncoder, tokenize
        from transformers import AutoTokenizer

        self._torch, self._tokenize = torch, tokenize
        self._device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = BiEncoder(model_path).to(self._device).eval()
        self._model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        )

    def rerank(self, query: str, candidates: list[Candidate]) -> list[RankedCandidate]:
        texts = [query, *[item.text for item in candidates]]
        batch = self._tokenize(self._tokenizer, texts, self._device, 96)
        with self._torch.inference_mode():
            embeddings = self._model.embed(batch)
            scores = (embeddings[0:1] @ embeddings[1:].T).flatten().float().cpu().tolist()
        order = sorted(
            range(len(candidates)), key=lambda index: (-scores[index], candidates[index].item_id)
        )
        return [
            RankedCandidate(candidates[index].item_id, candidates[index].text, scores[index], rank)
            for rank, index in enumerate(order, 1)
        ]


class ReplayFirstBackend:
    """Replay frozen FIRST identifier rankings for offline, deterministic demos."""

    name = "first-replay"

    def __init__(self, results_path: str | Path) -> None:
        self._ranks: list[list[str]] = []
        for line in Path(results_path).read_text().splitlines():
            payload = json.loads(line).get("payload", {})
            if payload.get("status") == "complete" and payload.get("first_token_ranking"):
                self._ranks.append(payload["first_token_ranking"])
        if not self._ranks:
            raise ValueError(f"no complete FIRST results in {results_path}")
        self._calls = 0

    def rerank(self, query: str, candidates: list[Candidate]) -> list[RankedCandidate]:
        ranking = self._ranks[self._calls % len(self._ranks)]
        self._calls += 1
        positions = {letter: index for index, letter in enumerate(ranking)}
        order = sorted(
            range(len(candidates)), key=lambda index: positions.get(chr(65 + index), len(ranking))
        )
        return [
            RankedCandidate(
                candidates[index].item_id,
                candidates[index].text,
                float(len(candidates) - rank),
                rank,
            )
            for rank, index in enumerate(order, 1)
        ]


class SearchService:
    def __init__(
        self,
        student: RerankBackend | None = None,
        first: RerankBackend | None = None,
        first_budget: float = 0.4,
        first_timeout_ms: float = 2000.0,
        max_retries: int = 1,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 10.0,
    ) -> None:
        self.student = student or CachedBackend()
        self.first = first or CachedBackend()
        self.first_budget = first_budget
        self.requests = 0
        self.first_calls = 0
        self.total_ms = 0.0
        self.ab_counts = {"control_first": 0, "treatment_gate": 0}
        self.first_timeout_ms = first_timeout_ms
        self.max_retries = max_retries
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_cooldown_seconds = circuit_cooldown_seconds
        self.first_failures = 0
        self.degradations = 0
        self._circuit_open_until = 0.0
        self._first_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="r16-first")

    def route(self, query: str, candidates: list[Candidate]) -> dict[str, object]:
        digest = hashlib.sha256(query.encode()).digest()
        use_first = int.from_bytes(digest[:8], "big") / 2**64 < self.first_budget
        return {
            "backend": "first" if use_first else "student",
            "reason": "stable-budget-route",
            "budget": self.first_budget,
        }

    def search(
        self, query: str, candidates: list[Candidate], backend: str = "gate"
    ) -> dict[str, object]:
        started = time.perf_counter()
        decision = (
            self.route(query, candidates)
            if backend == "gate"
            else {"backend": backend, "reason": "explicit-backend"}
        )
        selected = decision["backend"]
        if selected == "first":
            result, selected, failure = self._first_with_resilience(query, candidates)
            if failure:
                decision = {**decision, "fallback": "student", "failure": failure}
            else:
                self.first_calls += 1
        elif selected == "student":
            result = self.student.rerank(query, candidates)
        else:
            raise ValueError(f"unknown backend: {selected}")
        elapsed = 1000 * (time.perf_counter() - started)
        self.requests += 1
        self.total_ms += elapsed
        return {
            "query": query,
            "backend": selected,
            "route": decision,
            "results": [item.__dict__ for item in result],
            "latency_ms": elapsed,
        }

    def _first_with_resilience(
        self, query: str, candidates: list[Candidate]
    ) -> tuple[list[RankedCandidate], str, str | None]:
        if time.monotonic() < self._circuit_open_until:
            self.degradations += 1
            return self.student.rerank(query, candidates), "student", "circuit-open"
        failure = "provider-error"
        for attempt in range(self.max_retries + 1):
            future = self._first_executor.submit(self.first.rerank, query, candidates)
            try:
                result = future.result(timeout=self.first_timeout_ms / 1000)
                self.first_failures = 0
                return result, "first", None
            except TimeoutError:
                future.cancel()
                failure = "timeout"
            except Exception as error:
                failure = type(error).__name__
            self.first_failures += 1
            if attempt < self.max_retries:
                continue
        if self.first_failures >= self.circuit_failure_threshold:
            self._circuit_open_until = time.monotonic() + self.circuit_cooldown_seconds
        self.degradations += 1
        return self.student.rerank(query, candidates), "student", failure

    def metrics(self) -> dict[str, object]:
        return {
            "requests": self.requests,
            "first_calls": self.first_calls,
            "first_call_rate": self.first_calls / max(self.requests, 1),
            "mean_latency_ms": self.total_ms / max(self.requests, 1),
            "ab_counts": self.ab_counts,
            "first_failures": self.first_failures,
            "degradations": self.degradations,
            "circuit_open": time.monotonic() < self._circuit_open_until,
        }

    def ab_search(self, query: str, candidates: list[Candidate]) -> dict[str, object]:
        digest = hashlib.sha256(("r16-ab:" + query).encode()).digest()
        arm = "treatment_gate" if digest[0] < 128 else "control_first"
        self.ab_counts[arm] += 1
        return {
            "arm": arm,
            **self.search(query, candidates, "gate" if arm == "treatment_gate" else "first"),
        }


def create_app(service: SearchService | None = None):
    try:
        from fastapi import Body, FastAPI
        from fastapi.responses import PlainTextResponse
        from pydantic import BaseModel, Field
    except ImportError as error:  # pragma: no cover - exercised when app extra is absent
        raise RuntimeError(
            "install the 'app' extra to run the HTTP demo: uv sync --extra app"
        ) from error

    class CandidateInput(BaseModel):
        item_id: str
        text: str

    class SearchInput(BaseModel):
        query: str = Field(min_length=1)
        candidates: list[CandidateInput] = Field(min_length=1, max_length=100)
        backend: str = "gate"

    app = FastAPI(title="CAGED-LTR R16 Demo", version="0.1.0")
    runtime = service or SearchService()
    from .r16_llm_app import as_json, explain_result

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "student": runtime.student.name, "first": runtime.first.name}

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        return runtime.metrics()

    @app.get("/metrics/prometheus", response_class=PlainTextResponse)
    def prometheus_metrics() -> str:
        metrics = runtime.metrics()
        return (
            "\n".join(
                [
                    f"caged_ltr_requests_total {metrics['requests']}",
                    f"caged_ltr_first_calls_total {metrics['first_calls']}",
                    f"caged_ltr_first_call_rate {metrics['first_call_rate']}",
                    f"caged_ltr_mean_latency_ms {metrics['mean_latency_ms']}",
                    f"caged_ltr_first_failures_total {metrics['first_failures']}",
                    f"caged_ltr_degradations_total {metrics['degradations']}",
                ]
            )
            + "\n"
        )

    @app.post("/route")
    def route(payload: SearchInput = Body(...)) -> dict[str, object]:
        candidates = [Candidate(item.item_id, item.text) for item in payload.candidates]
        return runtime.route(payload.query, candidates)

    @app.post("/search")
    def search(payload: SearchInput = Body(...)) -> dict[str, object]:
        candidates = [Candidate(item.item_id, item.text) for item in payload.candidates]
        return runtime.search(payload.query, candidates, payload.backend)

    @app.post("/ab/search")
    def ab_search(payload: SearchInput = Body(...)) -> dict[str, object]:
        candidates = [Candidate(item.item_id, item.text) for item in payload.candidates]
        return runtime.ab_search(payload.query, candidates)

    @app.post("/understand")
    def understand(payload: SearchInput = Body(...)) -> dict[str, object]:
        return as_json(payload.query)

    @app.post("/explain")
    def explain(payload: SearchInput = Body(...)) -> dict[str, object]:
        result = runtime.search(
            payload.query,
            [Candidate(item.item_id, item.text) for item in payload.candidates],
            payload.backend,
        )
        return {
            "query": payload.query,
            "explanations": [
                explain_result(
                    payload.query, item["item_id"], item["text"], item["score"], result["backend"]
                )
                for item in result["results"]
            ],
        }

    return app


try:
    app = create_app()
except RuntimeError:
    # Core ranking utilities remain importable without installing the demo extra.
    app = None
