# R15 离线 A/A 与 A/B replay

本实验严格使用 R12 dev 的 10,000 个 query。A/B 协议在读取 replay 结果前提交；
R12 confirm、large-test 和 NFCorpus test 均未访问。

## A/A 检查

| 项目 | 结果 |
|---|---:|
| Treatment / Control | 5,029 / 4,971 |
| SRM p-value | 0.561915 |
| NDCG@10 差值 | +0.000782 |
| bootstrap 95% CI | [-0.008209, +0.009701] |
| 最大 Head/Torso/Tail 占比差 | 0.005134 |

A/A 的 SRM、分桶平衡和零差异置信区间检查全部通过。

## A/B replay

Control 为全量 FIRST；Treatment 为五折 OOF Tail-safe Gate，FIRST 预算固定为 60%，
Tail 保底 100%，Torso 保底 65%。

| 指标 | Control FIRST | Treatment Gate | 差异 |
|---|---:|---:|---:|
| 随机分组 NDCG@10 | 0.644943 | 0.670171 | +0.025228 |
| FIRST 调用率 | 1.0000 | 0.5999 | -40.01% |
| 平均延迟 ms | 68.866 | 48.095 | -30.16% |
| P50 ms | 73.719 | 74.184 | +0.466 |
| P95 ms | 81.930 | 86.392 | +4.461 |
| P99 ms | 82.697 | 87.184 | +4.486 |

随机化 replay 的 NDCG@10 差值 95% CI 为 `[+0.016245, +0.034526]`；利用离线环境中
两个潜在策略结果均可观测的性质，paired 差值为 `+0.023797`，95% CI 为
`[+0.019999, +0.027489]`。Tail 因 100% FIRST 保底，与 Control 完全相同。

预注册的质量非劣、Tail 非劣、调用率、平均延迟和 P99 护栏均通过。需要同时保留的
限制是：Gate 降低了平均延迟，但 P95/P99 分别增加约 4.46/4.49 ms；CTR、长点击和
CVR 在离线 MIND 数据中不可观测，因此该结果不能称为真实线上 A/B。
