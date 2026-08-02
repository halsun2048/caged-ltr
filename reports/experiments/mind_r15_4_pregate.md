# R15.4 qrels-free Pre-Gate 锁定结果

R15.4 按预注册协议在 R12 dev 的 10,000 个 query 上执行五折 OOF 路由蒸馏。Pre-Gate 仅使用请求文本、候选文本统计、候选训练频次统计与词面重叠特征，不读取 qrels、相关 item 频次、Student 分数或 FIRST 输出。confirm、large-test 与 NFCorpus test 均未访问。

## 结果

| 系统 | NDCG@10 | Tail NDCG@10 | FIRST 调用率 | 平均延迟 | P99 |
|---|---:|---:|---:|---:|---:|
| 全量 FIRST | 0.645336 | 0.633524 | 100.00% | 68.87 ms | 82.70 ms |
| Post-Student Gate | 0.669133 | 0.633524 | 59.99% | 48.09 ms | 87.18 ms |
| qrels-free Pre-Gate | 0.623928 | 0.597878 | 59.99% | 48.45 ms | 84.47 ms |

Pre-Gate 与 Post-Student Gate 的路由一致率为 70.68%。总体 NDCG@10 下降 0.045206，配对 bootstrap 95% CI 为 [-0.048674, -0.041771]；Tail 下降 0.035646，95% CI 为 [-0.040776, -0.030604]。差异明确且具有实际规模。

Pre-Gate 自身开销为平均 1.78 ms、P99 2.00 ms。其平均延迟相对全量 FIRST 降低 29.65%，但仍未达到预注册的 30% 阈值；P99 比全量 FIRST 高 1.77 ms，也未达到不超过 1 ms 的阈值。

## 判定

本轮 **不准入**。七项验收中仅 FIRST 调用率通过；路由一致率、总体质量、Tail、Pre-Gate P99、端到端 P99 和平均延迟降幅均失败。因此不能用当前静态 Pre-Gate 替换 Post-Student Gate，也不应访问 confirm 或 untouched test。

该结果说明，当前 Gate 的优势依赖 Student 推理后产生的信息以及 Tail 保护逻辑。候选频次聚合等廉价特征不足以在请求前恢复“FIRST 相对 Student 的增益”。下一步若继续，应优先测试共享浅层编码、级联早退或并行投机执行，而不是在同一 dev 上继续调静态阈值。
