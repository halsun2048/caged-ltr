import json

from caged_ltr.r16_llm_app import as_json, explain_result, understand_query
from caged_ltr.r16_service import (
    CachedBackend,
    Candidate,
    ReplayFirstBackend,
    SearchService,
    lexical_score,
)


def candidates():
    return [Candidate("a", "best restaurants downtown"), Candidate("b", "home cooking guide")]


def test_cached_backend_and_service_routes():
    assert lexical_score("best restaurants", "best restaurants downtown") > 0
    assert CachedBackend().rerank("best restaurants", candidates())[0].item_id == "a"
    service = SearchService(first_budget=1.0)
    result = service.search("best restaurants", candidates(), "gate")
    assert result["backend"] == "first"
    assert service.metrics()["first_calls"] == 1
    service.ab_search("another query", candidates())
    assert sum(service.metrics()["ab_counts"].values()) == 1


def test_replay_first(tmp_path):
    path = tmp_path / "first.jsonl"
    path.write_text(json.dumps({"payload": {"status": "complete", "first_token_ranking": ["B", "A"]}}) + "\n")
    result = ReplayFirstBackend(path).rerank("q", candidates())
    assert result[0].item_id == "b"


def test_structured_query_and_grounded_explanation():
    understanding = understand_query("compare cheap restaurants")
    assert understanding.intent == "comparison"
    assert as_json("best restaurants")["intent"] == "recommendation"
    explanation = explain_result("best restaurants", "a", "best restaurants downtown", 0.9, "student")
    assert explanation["grounded"] is True
    assert "restaurants" in explanation["evidence"]
