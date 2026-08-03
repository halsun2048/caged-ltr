# R48–R53 CPU 工程收尾

## 三种演示模式

```bash
scripts/run_cpu_demo.sh cached
scripts/run_cpu_demo.sh replay
scripts/run_cpu_demo.sh cpu
```

- `cached`：零模型依赖，适合快速展示 API、MCP、反馈和 A/B。
- `replay`：缓存 Student + 冻结 FIRST 结果，适合展示路由和降级语义。
- `cpu`：真实 MiniLM checkpoint 在 CPU 推理，FIRST 使用冻结 replay。

完整 Compose 默认使用 CPU 真实 MiniLM（`R16_BACKEND=real`、`R16_DEVICE=cpu`、
`CAGED_DENSE_PROVIDER=minilm_cpu`）。若只做快速接口演示，可显式设置 `R16_BACKEND=cached`。

默认本地资产已经固定为 `artifacts/models/all-MiniLM-L6-v2`、
`artifacts/r16_runtime/mind_r13_reweight_mild.pt` 和
`runs/mind_r10_0/dev_first/results.jsonl`。

## 最终结果摘要

| 证据 | NDCG@10 | Hit@10 | MRR | FIRST 调用率 |
|---|---:|---:|---:|---:|
| R12 MiniLM dev | 0.53235 | 0.9031 | 0.47634 | 0% |
| R8.11 Tail-floor Gate | 0.65009 | 0.9816 | 0.58531 | 40% |
| R8.11 全量 FIRST | 0.65291 | — | — | 100% |

Tail-floor Gate 与全量 FIRST 的 NDCG@10 差距为 0.00283，同时减少 60% FIRST 调用。
这是锁定离线实验结果，不应解释为真实线上 A/B。

RTX 4090 上真实 MiniLM、100 候选的锁定性能为 p50 5.97ms、p95 6.85ms、
p99 9.19ms、峰值显存约 108MB。CPU 性能需要在目标主机单独测量，不能沿用 GPU 数字。

## 完整 CPU/Docker 服务

```bash
cp configs/cpu.env.example .env
docker compose -p caged-ltr --env-file .env -f docker-compose.full.yml up -d --build
PYTHONPATH=src python3 scripts/check_r40_services.py
PYTHONPATH=src python3 scripts/smoke_demo_http.py
```

默认服务端口：API 8001、MCP 8766、Streamlit 8502、Prometheus 9090、Grafana 3000。
如果端口被占用，`scripts/run_full_stack.sh` 会自动选择空闲端口并在最后打印实际访问地址。
当前 CPU 主机实测端口为 API 8001、MCP 8766、Streamlit 8502、Prometheus 9090、Grafana 3000。
PostgreSQL、Redis、Qdrant 均有健康检查和持久化卷。

当前共享主机没有 Docker socket 权限，所以 Compose 完整启动需要在用户拥有 Docker 权限的
CPU 主机执行；代码、配置和无 Docker 的三模式运行不受影响。

完整启动命令：

```bash
sg docker -c './scripts/run_full_stack.sh'
```

脚本会分阶段显示核心依赖、API、MCP、Streamlit、Prometheus 和 Grafana 的进度，并对
`/health`、Streamlit、Prometheus、Grafana 做就绪检查。

## 演示流程

1. 打开 Streamlit，选择固定 Query 并展示候选排序。
2. 展示 Student/Gate/FIRST 路由、调用预算和结果解释。
3. 对同一 user_id 连续调用 `/ab/search`，证明分桶稳定。
4. 提交 click/like 反馈并展示 `/events/summary`。
5. 用 MCP 调用 `search` 和 `get_runtime_metrics`。
6. 打开 Prometheus/Grafana，展示 QPS、延迟、FIRST 调用和降级指标。

架构图源码见 `docs/r48_r53_architecture.mmd`。

## 最终验收与发布

在 Docker 服务已启动且端口采用默认值（API 8001、MCP 8766）时运行：

```bash
API_URL=http://127.0.0.1:8001 \
MCP_URL=http://127.0.0.1:8766 \
PROM_URL=http://127.0.0.1:9090 \
GRAFANA_URL=http://127.0.0.1:3000 \
scripts/verify_infra_observability.sh
```

该验收会实际调用 MCP JSON-RPC、API 事件存储、Redis、Qdrant、Prometheus
和 Grafana，并输出 `reports/experiments/r54_infrastructure_acceptance.json`。
已锁定的 untouched test 不在此流程中访问。

完成验收后，在本机执行 `scripts/finalize_release.sh`，提交代码并创建
`v1.0.0-cpu-closeout` tag。模型和 checkpoint 只记录在
`reports/data/local_asset_manifest.json`，不进入 Git。
