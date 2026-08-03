# 论文冻结结果表（离线）

| 系统 | NDCG@10 | Hit@10 | MRR | FIRST 调用率 |
|---|---:|---:|---:|---:|
| MiniLM student（R12 dev） | 0.53235 | 0.90310 | 0.47634 | 0% |
| Tail-floor Gate（R8.11） | 0.65009 | 0.98160 | 0.58531 | 40% |
| 全量 FIRST（同一离线评估） | 0.65291 | — | — | 100% |

Tail-floor Gate 相对全量 FIRST 的 NDCG@10 差距约为 0.00283，同时减少 60% FIRST 调用。该结果是冻结的离线证据，不能外推为线上用户收益。

论文需要进一步从现有 JSON 报告生成 bootstrap 置信区间、三 seed 均值/标准差和 Head/Torso/Tail 图；这些属于统计汇总，不需要 GPU，也不应重新访问 untouched test。
