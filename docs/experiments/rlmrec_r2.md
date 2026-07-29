# R2：LightGCN + RLMRec-Con 本地结构复现

## 当前边界

本阶段使用 RLMRec 作者公开的 Yelp 稀疏矩阵和 profile embedding，复现
LightGCN、semantic-only、RLMRec-Con 与 shuffled-Con。公开数据来自
`HKUDS/RLMRec` 配套 archive.org 压缩包，作者仓库审计固定在 commit
`22413752246de3dee8ab0d509f7f7a8889080f95`。

这不是严格的无泄漏数值复现：

- 作者 profile 示例直接使用交互和评论文本；
- 公开包没有时间戳，也未说明 profile 是否只使用训练矩阵之前的信息；
- 本地为降低 CPU 开销，对公开的 1536 维用户/物品向量联合执行确定性 PCA64；
- 正式本地 batch 为 1024，而作者配置为 4096。

所以结果只能称为“官方数据与目标结构复现”。即使 RLMRec-Con 提升，也不能据此
断言提升来自无泄漏语义，更不能解释为因果去噪。

## 数据审计

安全转换结果：

| 字段 | 数值 |
|---|---:|
| 用户 | 11,091 |
| 物品 | 11,010 |
| 训练交互 | 166,620 |
| 验证交互 | 55,479 |
| 测试交互 | 55,436 |
| 公开 embedding 维数 | 1,536 |
| 本地结构复现维数 | 64 |
| PCA64 解释方差比 | 0.622541 |
| 三组 pair overlap | 0 |

外部 pickle 没有调用 `pickle.load`。矩阵和 ndarray 均先核对固定 SHA-256，再通过
opcode/global allowlist 提取数值 buffer，转换为 NPZ/NPY。

## 与作者代码的一致项和差异

一致项：

- LightGCN 32 维、3 层，累加第 0—3 层表示；
- 训练时保留率 0.8 的 edge dropout，且不对保留边重新缩放；
- BPR、Adam `1e-3`、全目录评测；
- 验证集 `Recall@20` 选 checkpoint，测试只在 checkpoint 确定后访问；
- Con 使用共享 MLP，并分别对用户、正物品、负物品做 InfoNCE；
- 采用作者 Yelp 配置中的 `kd_weight=0.01` 和温度 `0.2`。

本地差异：

- profile embedding 从 1536 维压缩为联合 PCA64；
- batch 从 4096 降为 1024；
- 最大 300 epoch，而不是作者配置的 3000；
- 目前只做 LightGCN 主骨干，没有扩展到其他五种骨干。

论文正文可简写为推荐损失与信息对齐损失之和，但作者发布代码明确在对齐项前乘
`0.01`。工程报告按“作者代码实现”记录这个权重，不把它伪装成无权重的精确目标；
未来研究性 `lambda_align` 搜索必须另设实验名。

## 对照与评测

首轮四组：

- `lightgcn`：纯协同骨干；
- `semantic_only`：冻结 profile embedding 经共享 MLP 后直接做 BPR；
- `rlmrec_con`：真实实体语义与 LightGCN 表示对齐；
- `shuffled_con`：固定打乱实体—语义对应关系，训练预算与目标不变。

验证和测试均执行 11,010 物品的完整目录排序，只屏蔽训练交互，以匹配作者代码。
除 Overall Recall/NDCG@5/10/20 外，工程额外按训练物品频次最低/最高 20% 报告
Tail/Head。epoch time 只表示训练成本，不表示线上推理 latency。

## 执行

数据准备：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/prepare_rlmrec_yelp.py
```

真实数据 smoke 已通过：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_rlmrec_r2.py --smoke --progress
```

首轮正式实验只跑 seed 42；四组均完成并通过身份对照后，再扩展到
`2024`、`3407`：

```bash
systemd-inhibit --what=sleep --mode=block \
  --why="R2 LightGCN RLMRec-Con Yelp training" \
  uv --cache-dir .uv-cache run --frozen \
    python scripts/run_rlmrec_r2.py \
      --seeds 42 \
      --variants lightgcn semantic_only rlmrec_con shuffled_con \
      --progress
```

脚本实时显示 epoch 内 batch、验证/测试用户进度，并在每个 epoch 写入
`latest.pt`。中断后重复同一命令会从已完成 epoch 恢复；完成的 variant 直接读取
`summary.json`。正式报告写入
`reports/experiments/rlmrec_r2.json`。

## 后续验收

seed 42 首轮首先检查：

1. LightGCN 是否接近论文 Yelp `Recall@20=0.1157`、`NDCG@20=0.0733`；
2. RLMRec-Con 是否超过 LightGCN；
3. 真实语义是否超过 shuffled-Con；
4. semantic-only 是否明显弱于协同模型；
5. Overall 提升是否伴随 Tail 下降。

只有这些身份门通过，才运行三种子。由于 profile 截止时间不可验证，三种子通过后
仍需在带时间戳、严格 train-only profile 的本地 Yelp 路线上做泄漏安全确认。

## Seed 42 正式结果

四组均完成验证集 checkpoint 选择和一次最终测试：

| 模型 | Recall@20 | NDCG@20 | 最佳 epoch |
|---|---:|---:|---:|
| 论文 LightGCN | 0.115700 | 0.073300 | — |
| 本地 LightGCN | 0.115823 | 0.073247 | 108 |
| RLMRec-Con | 0.116277 | 0.073074 | 45 |
| shuffled-Con | 0.094562 | 0.058749 | 39 |
| semantic-only | 0.020669 | 0.013242 | 78 |

LightGCN 与论文值仅相差 `+0.000123/-0.000053`，说明数据、图传播和全目录评测
协议已基本对齐。RLMRec-Con 相对 LightGCN 的 Recall@20 为 `+0.000454`，但
NDCG@20 为 `-0.000173`；Recall/NDCG 方向不一致，未复现论文
`0.1230/0.0776` 的整体提升。

所有 RLMRec-Con 相对 LightGCN 的 Test 下降单元格：

| 分桶 | 下降指标 | 绝对差 |
|---|---|---:|
| Overall | NDCG@10 | -0.000372 |
| Overall | NDCG@20 | -0.000173 |
| Head | Recall@10 | -0.001739 |
| Head | NDCG@10 | -0.001760 |
| Head | Recall@20 | -0.003272 |
| Head | NDCG@20 | -0.002132 |

Torso Recall/NDCG@20 分别增加 `+0.003403/+0.001453`，但 Tail 仅增加
`+0.000082/+0.000025`，不具备实际规模。结果更像从 Head 向 Torso 的排序
重分配，而非整体质量提升。真实语义显著优于 shuffled-Con，说明实体身份对应关系
确实影响目标；但“真实对齐不差于错误对齐”不等价于“真实对齐优于不做对齐”。

semantic-only 的 Overall 明显较弱，不能替代协同模型；其 Tail NDCG@20
`0.001266` 高于协同模型，表示更容易把尾部物品排入前列，但没有转化为可接受的
整体相关性。

当前身份门为部分通过，不直接扩展三种子。下一个最小判别实验固定 seed 42 和
PCA64，只把 batch 从 1024 恢复为作者配置的 4096，同时运行 LightGCN、
RLMRec-Con、shuffled-Con。Con 的批内负样本数量直接随 batch 改变，这一步能够
先隔离 batch 忠实度；若仍失败，再决定是否把原始 1536 维 MLP 迁移到服务器。

## R2.1：Batch 4096 忠实度消融

R2.1 使用独立的输出目录和报告，禁止读取 R2 batch-1024 checkpoint。runner 会把
完整解析配置写入 summary/latest checkpoint；配置不一致时直接报错，不静默复用。

```bash
systemd-inhibit --what=sleep --mode=block \
  --why="R2.1 RLMRec batch 4096 fidelity training" \
  uv --cache-dir .uv-cache run --frozen \
    python scripts/run_rlmrec_r2.py \
      --seeds 42 \
      --variants lightgcn rlmrec_con shuffled_con \
      --batch-size 4096 \
      --max-epochs 3000 \
      --output-root runs/rlmrec_r2_batch4096 \
      --report reports/experiments/rlmrec_r2_batch4096.json \
      --progress
```

进度输出覆盖 epoch 内 batch、验证用户和最终测试用户；每个 epoch 保存
`latest.pt`，中断后重复命令从最后完成的 epoch 继续。报告自动计算 Con 相对
LightGCN/shuffled-Con 的 Overall/Head/Torso/Tail Recall/NDCG@20 差值。

进入三种子的必要条件：

- LightGCN 的论文 Recall/NDCG@20 绝对误差均不超过 `0.002`；
- Con 的 Overall Recall@20、NDCG@20 同时超过 LightGCN；
- 真实 Con 的两个 Overall 指标同时超过 shuffled-Con；
- Tail NDCG@20 绝对增益至少 `0.001`；
- Head NDCG@20 绝对下降不超过 `0.002`。

任一核心条件失败，都不在本地搜索对齐权重；下一决策只能是原始 1536 维 GPU
忠实审计，或记录 R2 未复现后进入 R3。

### R2.1 正式结果

三组 seed 42 均完成，所有输入数据与 R2 batch-1024 的 SHA-256 一致：

| 模型 | Recall@20 | NDCG@20 | 最佳 epoch | 总耗时 |
|---|---:|---:|---:|---:|
| LightGCN | 0.117697 | 0.073911 | 210 | 11m03s |
| RLMRec-Con | 0.116937 | 0.072712 | 96 | 1h11m39s |
| shuffled-Con | 0.093074 | 0.059081 | 69 | 54m40s |

耗时是包含定期全目录验证和最终测试的实验训练成本，不是线上推理 latency。

RLMRec-Con 相对同 batch LightGCN：

| 分桶 | Recall@20 差值 | NDCG@20 差值 |
|---|---:|---:|
| Overall | -0.000760 | -0.001199 |
| Head | -0.006436 | -0.004014 |
| Torso | +0.004830 | +0.001862 |
| Tail | +0.000321 | +0.000068 |

五项门槛只通过两项：LightGCN 在论文值 `0.002` 容差内，且真实 Con 显著超过
shuffled-Con。Con 同时超过 LightGCN、Tail `0.001` 实际规模、Head `0.002`
下降容差均失败。

相较 batch-1024，batch-4096 LightGCN 的 Recall/NDCG@20 增加
`+0.001874/+0.000665`；RLMRec-Con 仅变化 `+0.000659/-0.000362`。因此 Con
相对基线从 Recall 略升、NDCG 略降，恶化为两个整体指标同时下降。作者 batch
未恢复论文增益，batch 假设被排除。

依停止规则，不运行本地 seed 2024/3407，也不搜索对齐权重。R2 只保留一次最终
忠实度决策：在 GPU 上使用公开原始 1536 维 embedding、batch 4096 运行 seed 42
的真实/打乱 Con，并复用本轮 LightGCN。若原始维度仍不能同时提升 Recall/NDCG，
则正式记录 R2 未复现，进入 R3。
