"""Dependency-light MCP bridge for the CAGED-LTR HTTP service.

The implementation supports the MCP JSON-RPC lifecycle needed by local agents
without loading a second copy of the ranking models. It can run over stdio or
as a small HTTP endpoint and delegates every ranking operation to FastAPI.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
ROOT = Path(__file__).resolve().parents[2]

TOOLS = [
    {
        "name": "search",
        "description": "Run CAGED-LTR ranking through the HTTP API.",
        "inputSchema": {
            "type": "object",
            "required": ["query", "candidates"],
            "properties": {
                "query": {"type": "string"},
                "candidates": {"type": "array"},
                "backend": {"type": "string", "enum": ["gate", "student", "first"]},
            },
        },
    },
    {
        "name": "understand_query",
        "description": "Extract query intent and constraints.",
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
        },
    },
    {
        "name": "explain_results",
        "description": "Return grounded explanations for ranked candidates.",
        "inputSchema": {
            "type": "object",
            "required": ["query", "candidates"],
            "properties": {
                "query": {"type": "string"},
                "candidates": {"type": "array"},
                "backend": {"type": "string"},
            },
        },
    },
    {
        "name": "run_ab_search",
        "description": "Run stable user-level A/B assignment.",
        "inputSchema": {
            "type": "object",
            "required": ["query", "candidates", "user_id"],
            "properties": {
                "query": {"type": "string"},
                "candidates": {"type": "array"},
                "user_id": {"type": "string"},
            },
        },
    },
    {
        "name": "get_runtime_metrics",
        "description": "Read request, latency, failure and degradation counters.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_demo_queries",
        "description": "List fixed portfolio demonstration queries.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class McpBridge:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("CAGED_API_URL", "http://127.0.0.1:8000")).rstrip(
            "/"
        )

    def _request(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:  # pragma: no cover
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"content-type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"CAGED-LTR API unavailable: {type(error).__name__}") from error

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search":
            return self._request("/search", arguments)
        if name == "understand_query":
            return self._request(
                "/understand",
                {"query": arguments["query"], "candidates": [{"item_id": "mcp", "text": "mcp"}]},
            )
        if name == "explain_results":
            return self._request("/explain", arguments)
        if name == "run_ab_search":
            return self._request("/ab/search", arguments)
        if name == "get_runtime_metrics":
            return self._request("/metrics")
        if name == "list_demo_queries":
            return {"queries": json.loads((ROOT / "data/demo/queries.json").read_text())}
        raise ValueError(f"unknown MCP tool: {name}")

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method, request_id = message.get("method"), message.get("id")
        if method == "notifications/initialized":
            return None
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "caged-ltr-mcp", "version": "1.0.0"},
                },
            }
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
        if method == "tools/call":
            params = message.get("params", {})
            try:
                output = self.call(params["name"], params.get("arguments", {}))
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(output, ensure_ascii=False)}
                        ],
                        "structuredContent": output,
                    },
                }
            except Exception as error:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": str(error)},
                }
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }


def serve_stdio(bridge: McpBridge) -> None:
    import sys

    for line in sys.stdin:
        if line.strip():
            response = bridge.handle(json.loads(line))
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def serve_http(bridge: McpBridge, host: str, port: int) -> None:  # pragma: no cover
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", "0"))
            response = bridge.handle(json.loads(self.rfile.read(length)))
            body = json.dumps(response or {"ok": True}, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()
