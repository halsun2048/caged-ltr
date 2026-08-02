"""Exercise the OpenAI-compatible JSON path with a local mock provider."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from caged_ltr.r16_llm_app import as_json


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("content-length", 0))
        json.loads(self.rfile.read(size))
        body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "rewritten_query": "cheap downtown restaurants",
                                "intent": "recommendation",
                                "constraints": ["cheap", "downtown"],
                                "confidence": 0.91,
                            }
                        )
                    }
                }
            ]
        }
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_):
        return


def main() -> None:
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    old = {key: os.environ.get(key) for key in ("R16_LLM_ENDPOINT", "R16_LLM_API_KEY")}
    os.environ["R16_LLM_ENDPOINT"] = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    os.environ["R16_LLM_API_KEY"] = "test-only"
    result = as_json("find a cheap place downtown")
    server.shutdown()
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    report = {
        "schema": "caged_ltr_r18_llm_provider_smoke_v1",
        "provider": result["provider"],
        "result": result,
        "external_api_called": False,
    }
    path = Path("reports/experiments/r18_llm_provider_smoke.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "provider": result["provider"], "report": str(path)}))


if __name__ == "__main__":
    main()
