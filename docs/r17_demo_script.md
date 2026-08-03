# CAGED-LTR 实习展示脚本（约 5 分钟）

1. **问题（20 秒）**：说明所有请求都调用大模型会带来延迟和成本，因此系统用 Student 处理简单请求，用 Gate 把困难请求交给 FIRST。
2. **项目总览（30 秒）**：展示 MiniLM P99、FIRST P99、Gate 调用率和 QPS 四个指标，强调这是可观测的服务，而不只是离线模型。
3. **智能搜索（50 秒）**：输入 `best restaurants near downtown`，展示 Query 意图、路由决策、排序结果、共享关键词证据和单请求延迟。
4. **策略对比（30 秒）**：切换 `student`、`first`、`gate`，说明不同后端的质量—延迟取舍。
5. **A/B 面板（30 秒）**：展示 Gate 与全量 FIRST 的 NDCG、FIRST 调用减少和平均延迟降低；说明结果来自 dev 离线随机回放，不冒充线上 CTR。
6. **故障治理（20 秒）**：说明 FIRST 超时、重试、熔断后会回退 Student，并在 Prometheus 中记录降级次数。
7. **R20 失败边界（40 秒）**：展示统一在线特征的 OOF Gate 未达到近 FIRST 准入，说明系统保留负面结果且没有继续访问 confirm/test。
8. **工程收尾（30 秒）**：展示 Docker Compose、FastAPI、Streamlit 和架构图，说明 GPU 关闭后仍可用 cached API 完整演示。

建议最后明确两点边界：FIRST 当前使用冻结 replay；真实线上 CTR/CVR 需要接入业务日志后再评估。
