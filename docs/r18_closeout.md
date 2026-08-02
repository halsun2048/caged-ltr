# R18.0–R18.6 收尾说明

## 路由模式

- `demo_hash`：仅用于离线展示，按稳定预算哈希分流。
- `post_student_gate`：先运行 Student，再使用 dev-only 训练的 Logistic gain router 决定是否调用 FIRST。
- `explicit_backend`：直接指定 Student 或 FIRST。

API 的 `/health` 会返回 `route_mode`，响应的 `route.reason` 会明确标识模式，避免把演示路由误称为训练 Gate。

## R18.1 Gate

`artifacts/r18_post_student_gate.json` 使用 R12 dev 的 Student 结果和 FIRST-vs-Student gain 标签拟合，固定 40% FIRST 预算。标签只用于离线训练，未进入请求时特征；confirm、large-test 和 NFCorpus test 未访问。由于 R15.4 Pre-Gate 已失败，本 Gate 明确是 post-Student，不声称可以提前跳过 Student。

## 服务治理

MiniLM 推理增加线程锁；HTTP 服务增加每客户端分钟限流、批量接口、结构化请求日志、FIRST timeout/retry/circuit-breaker/fallback 和 Prometheus 计数。

## LLM provider

设置 `R16_LLM_ENDPOINT` 后，`/understand` 调用 OpenAI-compatible JSON 接口；没有 endpoint、请求超时或格式不正确时回退 deterministic provider。R18 provider smoke 使用本地 mock server 验证了真实 JSON 解析路径，不调用外部 API。

## 交付

Dockerfile 已复制报告和演示缓存，GitHub Actions 在打 `r18-*` tag 时构建镜像并检查 API 模块。当前服务器没有 Docker Engine，因此镜像构建由 CI 或本地 Docker runner 执行。
