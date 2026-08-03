# CAGED-LTR 实习展示材料

## 30 秒介绍

我实现了一个成本感知的搜索重排级联系统。轻量 MiniLM 先完成排序，Gate 识别
困难和长尾请求，再选择性调用 FIRST 大模型。系统不仅包含蒸馏和 OOF Gate，
还实现了 FastAPI、Streamlit、A/B 分流、限流、熔断、降级和 Prometheus。
最强离线策略用 40% FIRST 调用达到 0.6501 NDCG@10；严格部署审计也保留了
未通过准入的负面结果。

## 3 分钟结构

1. 问题：全量大模型排序质量高，但延迟和费用不可接受。
2. 方法：Student → Gate → FIRST，FIRST 失败则回退 Student。
3. 研究：蒸馏、Head/Torso/Tail、五折 OOF、固定 confirm、bootstrap。
4. 结果：R8.9 展示质量—成本潜力；R20 揭示在线特征仍不足。
5. 工程：HTTP API、Streamlit、A/B、监控、Docker 和故障治理。
6. 边界：没有真实 CTR/CVR，不把 replay 说成线上实验。

## 10 分钟追问准备

- 为什么是 Post-Student Gate：请求时需要 Student margin 等特征。
- 为什么 Pre-Gate 没替换：R15.4 未通过质量和延迟准入。
- 为什么 Gate 有时高于 FIRST：Student 在部分 Head query 上更好，两者互补。
- 为什么停止 R20：统一特征下距离 FIRST 仍为 0.0228，超过 0.01 上限。
- 如何真正上线：接入真实日志、认证、TLS、provider 配额和在线 A/B。

## 演示顺序

项目总览 → 智能搜索 → 切换 Student/FIRST/Gate → A/B 稳定分流 →
性能与成本 → Prometheus → 限制与失败结果。
