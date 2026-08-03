# CAGED-LTR：大模型应用开发路线

这条路线面向实习展示，重点是完整的大模型搜索排序应用，而不是新增训练。

## 展示主线

`Query → lexical/dense hybrid retrieval → MiniLM student → gain-aware gate → Student/FIRST → explanation/feedback/A-B`

可展示组件：

- FastAPI：搜索、路由、反馈、事件、指标接口；
- Streamlit：Query、排序、路由和解释可视化；
- MCP：JSON-RPC 2024-11-05，6 个工具；
- MiniLM：CPU checkpoint 推理；
- FIRST：冻结 replay/provider；
- PostgreSQL、Redis、Qdrant：真实适配器和 Compose 配置；
- Prometheus/Grafana：延迟、QPS、FIRST 调用和降级监控。

## 启动

```bash
scripts/run_cpu_demo.sh cached   # 最快演示
scripts/run_cpu_demo.sh replay   # Student + FIRST replay
scripts/run_cpu_demo.sh cpu      # 真实 MiniLM CPU
```

Docker 部署见 `configs/cpu.env.example` 和 `docker-compose.full.yml`。

## 证据

- RTX 4090 MiniLM：`reports/experiments/r46_minilm_gpu.json`；
- CPU/cached/replay 验收：`reports/experiments/r48_r53_acceptance.json`；
- 架构图：`docs/r48_r53_architecture.mmd`；
- 演示说明：`docs/r48_r53_closeout.md`。

## 边界

当前没有真实线上流量，A/B 是离线稳定分桶；FIRST 是 replay/provider 展示路径，不声称线上 LLM 调用成本或收益。
