"""Hybrid retrieval interfaces with deterministic lexical fallback."""

from __future__ import annotations

from dataclasses import dataclass

from .r16_service import Candidate, lexical_score


@dataclass(frozen=True)
class RetrievedCandidate:
    candidate: Candidate
    lexical_score: float
    dense_score: float = 0.0


class HybridRetriever:
    """BM25-compatible lexical fallback and optional vector-provider hook."""

    def __init__(self, vector_provider=None) -> None:
        self.vector_provider = vector_provider

    def retrieve(self, query: str, candidates: list[Candidate], limit: int = 20) -> list[Candidate]:
        lexical = [RetrievedCandidate(item, lexical_score(query, item.text)) for item in candidates]
        if self.vector_provider is not None:
            dense_scores = self.vector_provider.score(query, candidates)
            lexical = [
                RetrievedCandidate(item.candidate, item.lexical_score, float(dense_scores[index]))
                for index, item in enumerate(lexical)
            ]
        ranked = sorted(
            lexical,
            key=lambda item: (-(item.lexical_score + item.dense_score), item.candidate.item_id),
        )
        return [item.candidate for item in ranked[:limit]]

    def retrieve_rrf(
        self, query: str, candidates: list[Candidate], limit: int = 20, k: int = 60
    ) -> list[Candidate]:
        """Fuse lexical and optional dense rankings with reciprocal rank fusion."""
        lexical = sorted(
            candidates, key=lambda item: (-lexical_score(query, item.text), item.item_id)
        )
        dense = lexical
        if self.vector_provider is not None:
            scores = self.vector_provider.score(query, candidates)
            dense = [
                item
                for _, item in sorted(
                    zip(scores, candidates, strict=True),
                    key=lambda pair: (-pair[0], pair[1].item_id),
                )
            ]
        lexical_rank = {item.item_id: index for index, item in enumerate(lexical, 1)}
        dense_rank = {item.item_id: index for index, item in enumerate(dense, 1)}
        fused = sorted(
            candidates,
            key=lambda item: (
                -(1 / (k + lexical_rank[item.item_id]) + 1 / (k + dense_rank[item.item_id])),
                item.item_id,
            ),
        )
        return fused[:limit]


class TokenOverlapVectorProvider:
    """Deterministic local dense-like hook for smoke tests; not a neural embedding."""

    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        return [lexical_score(query, item.text) for item in candidates]


class MiniLMVectorProvider:  # pragma: no cover - requires model checkpoint/GPU
    """Real semantic provider backed by the project's frozen MiniLM checkpoint.

    The backend is loaded lazily and exposes cosine/dot-product scores through the
    same provider interface used by RRF.  This keeps CPU demos lightweight while
    making the production path explicit when a checkpoint is configured.
    """

    def __init__(self, model_path: str, checkpoint: str, device: str = "cuda") -> None:
        from .r16_service import MiniLMBackend

        self.backend = MiniLMBackend(model_path, checkpoint, device=device)

    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        ranked = self.backend.rerank(query, candidates)
        return [
            next(item.score for item in ranked if item.item_id == candidate.item_id)
            for candidate in candidates
        ]


class HttpEmbeddingProvider:  # pragma: no cover - requires external provider
    """OpenAI-compatible embedding endpoint adapter for remote GPU serving."""

    def __init__(
        self, endpoint: str, model: str, api_key: str | None = None, timeout: float = 10.0
    ) -> None:
        self.endpoint, self.model, self.api_key, self.timeout = endpoint, model, api_key, timeout

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import json
        from urllib.request import Request, urlopen

        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.endpoint,
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        return [row["embedding"] for row in sorted(payload["data"], key=lambda row: row["index"])]

    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        vectors = self._embed([query, *[item.text for item in candidates]])
        query_vector = vectors[0]
        return [
            sum(a * b for a, b in zip(query_vector, vector, strict=True))
            for vector in vectors[1:]
        ]


class QdrantProvider:  # pragma: no cover
    """Optional adapter; imported only when the qdrant-client extra is installed."""

    def __init__(self, url: str, collection: str, embedder=None) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as error:
            raise RuntimeError("install qdrant-client to use QdrantProvider") from error
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.embedder = embedder

    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        if self.embedder is None:
            raise RuntimeError("QdrantProvider requires an embedder")
        vector = self.embedder._embed([query])[0]
        result = self.client.query_points(
            collection_name=self.collection, query=vector, limit=len(candidates), with_payload=True
        )
        by_id = {str(point.payload.get("item_id")): float(point.score) for point in result.points}
        return [by_id.get(item.item_id, 0.0) for item in candidates]
