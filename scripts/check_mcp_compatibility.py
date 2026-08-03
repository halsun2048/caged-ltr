"""R39 protocol compatibility check for the local MCP JSON-RPC bridge."""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> None:
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list_demo_queries", "arguments": {}},
        },
        {"jsonrpc": "2.0", "id": 4, "method": "unknown/method", "params": {}},
    ]
    payload = "\n".join(json.dumps(message) for message in messages) + "\n"
    process = subprocess.run(
        [sys.executable, "scripts/run_mcp_server.py"],
        input=payload,
        text=True,
        capture_output=True,
        check=True,
    )
    responses = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    assert responses[0]["result"]["protocolVersion"] == "2024-11-05"
    assert len(responses[1]["result"]["tools"]) >= 6
    assert "queries" in responses[2]["result"]["structuredContent"]
    assert responses[3]["error"]["code"] == -32601
    print(
        json.dumps(
            {
                "stage": "complete",
                "responses": len(responses),
                "tools": len(responses[1]["result"]["tools"]),
            }
        )
    )


if __name__ == "__main__":
    main()
