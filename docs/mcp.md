# MCP 使用说明

R31 提供一个不重复加载模型的 MCP bridge。它通过 HTTP 调用已有 FastAPI，支持
stdio（供本地 Agent 启动）和 HTTP（供开发调试）。

## 启动

```bash
PYTHONPATH=src CAGED_API_URL=http://127.0.0.1:8000 \
  python scripts/run_mcp_server.py

PYTHONPATH=src CAGED_API_URL=http://127.0.0.1:8000 \
  python scripts/run_mcp_server.py --http --port 8765
```

支持的工具是 `search`、`understand_query`、`explain_results`、`run_ab_search`、
`get_runtime_metrics` 和 `list_demo_queries`。协议生命周期包括 `initialize`、
`notifications/initialized`、`tools/list` 和 `tools/call`。

## Agent 配置示例

```json
{
  "mcpServers": {
    "caged-ltr": {
      "command": "python",
      "args": ["scripts/run_mcp_server.py"],
      "env": {"CAGED_API_URL": "http://127.0.0.1:8000"}
    }
  }
}
```

当前 bridge 是本地工具服务，不包含认证、远程租户隔离或真实 provider 密钥管理。
