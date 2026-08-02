# R16 GPU 服务化准入与基准

## 资产准入

R16 在 RTX 4090 24GB 服务器上完成资产审计。MiniLM 配置、Tokenizer 配置、3 个 R13 学生 checkpoint 和 FIRST dev 缓存均存在并完成 SHA-256 清单；CUDA 可用。资产清单见 `r16_asset_manifest.json`。

## 真实 CUDA 基准

使用 R12 dev 候选池、R13 选定的 mild checkpoint，在 500 个请求中预热 30 个后测量 MiniLM。每个请求包含一个 Query 和约 20 个候选，使用同步 CUDA 推理并计入 GPU 同步时间。

| 后端 | P50 | P95 | P99 | 平均 | 吞吐 |
|---|---:|---:|---:|---:|---:|
| MiniLM Student | 4.62 ms | 4.80 ms | 4.90 ms | 4.62 ms | 216.6 QPS |

FIRST 与 Gate 的历史服务基准仍保留在 R14 报告中；本轮没有重新调用 FIRST，也没有访问 confirm 或 untouched test。

## 服务接口

`src/caged_ltr/r16_service.py` 提供统一服务层，`scripts/run_r16_api.py` 可启动 FastAPI 应用。服务提供 `/health`、`/metrics`、`/route` 和 `/search`，支持稳定预算 Gate、显式后端选择、调用率和延迟统计。默认 cached backend 让 GPU 释放后仍可离线演示；真实 Student/FIRST 适配器可替换同一接口。

应用依赖单独列在 `requirements-app.txt`，不改变研究环境的 `uv.lock`。
