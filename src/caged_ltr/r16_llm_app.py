"""Structured, provider-neutral LLM application helpers for the R16 demo."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QueryUnderstanding:
    rewritten_query: str
    intent: str
    constraints: list[str]
    confidence: float
    provider: str = "deterministic-fallback"


def understand_query(query: str) -> QueryUnderstanding:
    normalized = " ".join(query.strip().split())
    lowered = normalized.lower()
    if any(word in lowered for word in ("best", "recommend", "推荐", "适合")):
        intent = "recommendation"
    elif any(word in lowered for word in ("compare", "vs", "比较", "区别")):
        intent = "comparison"
    else:
        intent = "informational-search"
    constraints = [
        word
        for word in normalized.split()
        if word.lower() in {"near", "cheap", "best", "budget", "附近", "便宜"}
    ]
    return QueryUnderstanding(normalized, intent, constraints, 0.72)


def explain_result(
    query: str, item_id: str, text: str, score: float, backend: str
) -> dict[str, object]:
    understanding = understand_query(query)
    overlap = sorted(set(query.lower().split()) & set(text.lower().split()))
    return {
        "item_id": item_id,
        "reason": "与查询共享关键词并由排序模型选中" if overlap else "由排序模型综合候选特征选中",
        "evidence": overlap,
        "score": score,
        "backend": backend,
        "intent": understanding.intent,
        "grounded": True,
    }


def as_json(query: str) -> dict[str, object]:
    return asdict(understand_query(query))
