"""Launch the CAGED-LTR MCP bridge over stdio or HTTP."""

from __future__ import annotations

import argparse

from caged_ltr.mcp_server import McpBridge, serve_http, serve_stdio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--api-url", default=None)
    args = parser.parse_args()
    bridge = McpBridge(args.api_url)
    if args.http:
        serve_http(bridge, args.host, args.port)
    else:
        serve_stdio(bridge)


if __name__ == "__main__":
    main()
