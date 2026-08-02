# R9.0 项目收尾报告

## 锁定主结果（large-test，唯一一次）

| 方法 | NDCG@10 | Hit@10 | MRR | FIRST 调用率 | 延迟 ms/query | 吞吐 q/s |
|---|---:|---:|---:|---:|---:|---:|
| student | 0.526600 | 0.902900 | 0.469758 | 0.000 | 0.670 | 1492.54 |
| first | 0.647185 | 0.992550 | 0.549853 | 1.000 | 69.511 | 14.39 |
| gate | 0.650056 | 0.981550 | 0.586274 | 0.400 | 30.171 | 33.14 |

## Head/Torso/Tail

| 分桶 | MiniLM | FIRST | Tail-floor Gate | Gate 调用率 |
|---|---:|---:|---:|---:|
| head | 0.687712 | 0.653614 | 0.707489 | 0.108 |
| torso | 0.493146 | 0.648108 | 0.597471 | 0.342 |
| tail | 0.398935 | 0.639833 | 0.645200 | 0.750 |

## R9.0 消融与 Pareto

详见 `reports/tables/mind_r9_0_ablation.csv` 和 `reports/tables/mind_r9_0_pareto.csv`。普通 gain gate 在 Tail 上为 0.596950；锁定 Tail-floor gate 将 Tail 提升至 0.645200，同时保持 overall 0.650056。

## 结论、失败结果与限制

- 锁定 gate 的 overall 与 Tail 相对 FIRST 的 NDCG@10 差距均不超过 0.003，FIRST 调用率为 40%，减少约 60%。
- Tail-floor 是必要消融：普通 gain gate 的 Tail 明显低于 FIRST；提高 Tail 保底后恢复准入。
- Head 上学生已经强于 FIRST；Torso 仍是主要质量损失来源。
- large-test guard 已关闭，`evaluation_count=1`；不得重新评估或调参。
- 大型 checkpoint 仅本地保存，报告记录 SHA-256。

## 复现命令

```bash
python scripts/analyze_mind_r8_10_tail.py --progress
python scripts/select_mind_r8_11_tail_floor.py --progress
python scripts/build_r9_0_closeout.py
# R8.9 final 命令仅供审计，large-test guard 已 consumed_closed，禁止再次运行
python scripts/evaluate_mind_r8_9_tail_final_once.py --progress
```

## 哈希与 guard

- student checkpoint: `4cf97dd8d08aa3e9ab7c87d2d3eed87fe7e640edcfc3b461422b0d254e5c56f9`
- frozen gate: `56bdab96fb1d43331308860a2de32439d013eb7d72acca598a13b6c3e9cfbcd1`
- large-test query-id hash: `738c2c9b077ff8a4b7eaa32306bbe70b15ff0f0597d45c4ab76be2db659468fb`
- guard 状态: `consumed_closed`, evaluation_count=`1`
