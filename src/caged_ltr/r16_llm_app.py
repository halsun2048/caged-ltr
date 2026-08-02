"""Structured, provider-neutral LLM application helpers for the R16 demo."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from urllib.request import Request, urlopen


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


def understand_query_with_provider(query: str) -> QueryUnderstanding:
    """Use an OpenAI-compatible endpoint when configured, otherwise stay offline."""
    endpoint = os.getenv("R16_LLM_ENDPOINT")
    if not endpoint:
        return understand_query(query)
    request_body = {
        "model": os.getenv("R16_LLM_MODEL", "gpt-4o-mini"),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "Return JSON with rewritten_query, intent, constraints, confidence.",
            },
            {"role": "user", "content": query},
        ],
    }
    request = Request(
        endpoint,
        data=json.dumps(request_body).encode(),
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {os.getenv('R16_LLM_API_KEY', '')}",
        },
    )
    try:
        with urlopen(request, timeout=float(os.getenv("R16_LLM_TIMEOUT_SECONDS", "5"))) as response:
            outer = json.loads(response.read())
        content = outer["choices"][0]["message"]["content"]
        payload = json.loads(content)
        return QueryUnderstanding(
            str(payload["rewritten_query"]),
            str(payload["intent"]),
            [str(value) for value in payload.get("constraints", [])],
            float(payload.get("confidence", 0.5)),
            "openai-compatible",
        )
    except Exception:
        return understand_query(query)


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
    return asdict(understand_query_with_provider(query))
