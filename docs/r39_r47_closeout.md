# R39–R47 工程收尾

本轮把检索系统从单机演示扩展为可展示的 MCP/API 应用：MCP 使用 JSON-RPC 2024-11-05，提供 6 个工具；API 支持混合召回、稳定哈希 A/B、反馈事件、可选 Bearer API key 和 request-id；Streamlit 页面可调用搜索、显示路由并提交反馈。混合召回现在提供真实 MiniLM checkpoint、OpenAI-compatible embedding 和 Qdrant 存储适配器，token-overlap 仅作为离线 smoke fallback。

## 验证

- MCP 兼容性：`PYTHONPATH=src python3 scripts/check_mcp_compatibility.py`
- Provider：`PYTHONPATH=src python3 scripts/run_r18_llm_provider_smoke.py`
- 压测：`PYTHONPATH=src python3 scripts/benchmark_r46_api.py --base-url http://127.0.0.1:8000 --requests 20 --concurrency 4 --candidates 20 --output reports/experiments/r46_api_benchmark.json`
- 全量测试：`uv run --frozen pytest -q`

R46 缓存服务实测报告为 p50 7.59 ms、p95 103.33 ms、p99 109.62 ms、146.15 QPS（20 请求、并发 4）。这些数值仅用于当前部署配置的演示，不等价于 GPU/生产模型延迟。

在用户提供的 RTX 4090 服务器上，真实 MiniLM checkpoint（100 候选）实测 p50 5.97 ms、p95 6.85 ms、p99 9.19 ms，峰值显存约 108 MB，详见 `reports/experiments/r46_minilm_gpu.json`。

## 环境边界

`docker-compose.full.yml` 提供 PostgreSQL、Redis、Qdrant、API 和 Streamlit 的真实服务模板，并加入健康检查。可用 `PYTHONPATH=src python3 scripts/check_r40_services.py` 做无写入探活；当前执行环境没有 Docker socket/GPU 权限，因此无法在此主机启动容器或完成 GPU 延迟报告。部署主机上应配置 `CAGED_EVENT_DB`、`REDIS_URL`、`QDRANT_URL` 和 MiniLM checkpoint，然后重新运行 R46 压测。
