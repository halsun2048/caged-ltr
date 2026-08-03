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


class QdrantProvider:  # pragma: no cover
    """Optional adapter; imported only when the qdrant-client extra is installed."""

    def __init__(self, url: str, collection: str) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as error:
            raise RuntimeError("install qdrant-client to use QdrantProvider") from error
        self.client = QdrantClient(url=url)
        self.collection = collection

    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        raise NotImplementedError(
            "embed query and upsert collection-specific vectors before enabling Qdrant"
        )
