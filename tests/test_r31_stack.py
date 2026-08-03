import json

from caged_ltr.mcp_server import McpBridge
from caged_ltr.r16_service import Candidate
from caged_ltr.retrieval import HybridRetriever
from caged_ltr.runtime_state import MemoryState
from caged_ltr.storage import EventStore
from caged_ltr.tasks import TaskRunner


def test_mcp_lifecycle_and_tool_dispatch(monkeypatch):
    bridge = McpBridge("http://unused")
    assert (
        bridge.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})["result"]["serverInfo"][
            "name"
        ]
        == "caged-ltr-mcp"
    )
    listed = bridge.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert {tool["name"] for tool in listed["result"]["tools"]} >= {"search", "get_runtime_metrics"}
    monkeypatch.setattr(
        bridge, "_request", lambda path, payload=None: {"path": path, "payload": payload}
    )
    called = bridge.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "q", "candidates": []}},
        }
    )
    assert json.loads(called["result"]["content"][0]["text"])["path"] == "/search"
    assert bridge.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_event_store_and_feedback(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    event = store.record_search(
        {"query": "q", "backend": "student", "route": {"backend": "student"}}
    )
    store.record_feedback(event, "u", "item", "click")
    assert store.summary() == {"search_events": 1, "feedback_events": 1, "first_calls": 0}
    store.close()


def test_memory_state_retrieval_and_task_runner():
    state = MemoryState(max_entries=2)
    state.set("a", 1)
    assert state.get("a") == 1
    assert state.incr("count") == 1
    state.set("b", 2)
    state.set("c", 3)
    assert state.get("a") is None
    state.set("expired", 4, ttl_seconds=-1)
    assert state.get("expired") is None
    candidates = [Candidate("a", "best restaurants"), Candidate("b", "home cooking")]
    assert HybridRetriever().retrieve("best restaurants", candidates)[0].item_id == "a"
    runner = TaskRunner()
    assert runner.submit(lambda: 1).startswith("local-")
    runner.shutdown()


def test_mcp_errors_are_json_rpc_errors():
    bridge = McpBridge("http://unused")
    response = bridge.handle({"jsonrpc": "2.0", "id": 9, "method": "unknown"})
    assert response["error"]["code"] == -32601
