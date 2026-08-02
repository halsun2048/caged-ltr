import json

from caged_ltr.r16_llm_app import as_json, explain_result, understand_query
from caged_ltr.r16_service import (
    CachedBackend,
    Candidate,
    PostStudentGateRouter,
    ReplayFirstBackend,
    SearchService,
    lexical_score,
)
from caged_ltr.r18_gate_features import FEATURES, vector_from_ranked


def candidates():
    return [Candidate("a", "best restaurants downtown"), Candidate("b", "home cooking guide")]


class FailingBackend:
    name = "failing"

    def rerank(self, query, candidates):
        raise RuntimeError("provider unavailable")


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
    path.write_text(
        json.dumps({"payload": {"status": "complete", "first_token_ranking": ["B", "A"]}}) + "\n"
    )
    result = ReplayFirstBackend(path).rerank("q", candidates())
    assert result[0].item_id == "b"


def test_structured_query_and_grounded_explanation():
    understanding = understand_query("compare cheap restaurants")
    assert understanding.intent == "comparison"
    assert as_json("best restaurants")["intent"] == "recommendation"
    explanation = explain_result(
        "best restaurants", "a", "best restaurants downtown", 0.9, "student"
    )
    assert explanation["grounded"] is True
    assert "restaurants" in explanation["evidence"]


def test_first_failure_retries_and_degrades():
    service = SearchService(
        student=CachedBackend(),
        first=FailingBackend(),
        first_budget=1.0,
        max_retries=1,
        circuit_failure_threshold=1,
        circuit_cooldown_seconds=60,
    )
    result = service.search("q", candidates(), "first")
    assert result["backend"] == "student"
    assert result["route"]["failure"] == "RuntimeError"
    assert service.metrics()["degradations"] == 1
    second = service.search("q", candidates(), "first")
    assert second["route"]["failure"] == "circuit-open"


def test_post_student_gate_manifest_routes():
    router = PostStudentGateRouter("artifacts/r18_post_student_gate.json")
    service = SearchService(
        student=CachedBackend(),
        first=CachedBackend(),
        route_mode="post_student_gate",
        router=router,
    )
    result = service.search("best restaurants", candidates(), "gate")
    assert result["route"]["mode"] == "post_student_gate"
    assert 0 <= result["route"]["probability"] <= 1


def test_r19_feature_builder_uses_ranked_top_item():
    items = [Candidate("a", "unrelated"), Candidate("b", "best restaurants downtown")]
    ranked = CachedBackend().rerank("best restaurants", items)
    vector = vector_from_ranked("best restaurants", items, ranked)
    assert len(vector) == len(FEATURES) == 13
    assert vector[8] > 0  # lexical overlap belongs to the ranked top item, not input index 0
    assert vector[11] == len(ranked[0].text)
