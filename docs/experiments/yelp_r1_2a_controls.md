# Yelp R1.2a 语义身份与单分支对照

## 目的

R1.2a 的 seed 42 验证结果中，`dual_view_no_ca` 优于 LLMInit，但它增加了约 123 万
可训练参数，并使用 1536 维冻结原始 embedding。当前对照用于判断增益来自正确的物品语义
对应，还是高维随机特征、adapter 容量或单一分支。

所有模型继续采用相同的 seed 42、sampled-100 验证协议和 early stopping。脚本固定
`test_after_selection=false`，不构造测试集结果。

## 阶段 A：语义身份门控

- `real_raw`：复用已完成的 `dual_view_no_ca` checkpoint，不重新训练；
- `shuffled_raw`：按 control seed `20240725` 打乱物品与完整语义向量的对应关系，向量集合、
  范数和各维联合结构不变；
- `matched_random_raw`：生成 Gaussian 随机矩阵，并逐维匹配真实矩阵的经验均值和标准差。

真实语义必须相对两个控制都满足：

- Overall NDCG@10 绝对增益至少 `0.003`；
- Tail NDCG@10 绝对增益至少 `0.005`。

任一条件失败，运行器立即生成停止报告，不再训练单分支，也不进入 R1.2b 自蒸馏。

## 阶段 B：单分支对照

仅在阶段 A 通过后运行：

- `raw_semantic_only`（w/o collaborative）：冻结原始语义、两层无激活 adapter 和一个因果
  SASRec 编码器；
- `collaborative_only`（w/o raw semantic）：PCA64 初始化的标准 SASRec/LLMInit。

两者都重新进行 validation-only early stopping，避免复用已经访问过测试集的历史 R1.1
summary。双视图需要在 Overall 和 Tail 上分别优于两个单分支，才能说明两条分支共同解释
增益。

## 控制生成

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/prepare_yelp_raw_semantic_controls.py
```

矩阵保存在被 Git 忽略的数据目录，路径、定义、维度误差和 SHA-256 记录在
`reports/data/yelp_raw_semantic_controls.json`。训练运行器会重新校验源文件和两个控制文件
的哈希。

## 正式运行

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_yelp_r1_2a_semantic_controls.py --progress
```

进度显示当前控制、epoch、最好验证 NDCG、stale、loss、耗时和预估停止时间。完整 summary
会在重启时校验模型、seed、语义哈希及全部关键训练配置后复用。

## 正式结果

阶段 A 的三个模型均训练至 early stopping，测试集未访问。

| 变体 | Overall | Head | Torso | Tail | Cold-start | 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|
| `real_raw` | **0.426584** | **0.542450** | 0.181464 | 0.031652 | 0.000000 | 38 |
| `shuffled_raw` | 0.419538 | 0.531647 | **0.184686** | 0.032154 | 0.000000 | 44 |
| `matched_random_raw` | 0.421399 | 0.536151 | 0.177528 | **0.033146** | 0.000000 | 38 |

真实语义相对 shuffled 和 matched-random 的 Overall 增益分别为 `0.007046` 和
`0.005185`，超过 `0.003` 门槛；Head 也分别提高 `0.010803` 和 `0.006299`。这说明正确
语义对应在 Head 主导的 Overall 上有一定作用。

但 Tail 分别变化 `-0.000502` 和 `-0.001494`，不仅未达到 `+0.005`，方向也为负。
Cold-start 三组均为零。因此此前 no-CA 相对 LLMInit 的 Tail 提升不能归因于真实语义，
更可能由额外分支、adapter 容量或高维随机特征解释。

## 停止结论

身份门控失败，运行器按规则将状态写为 `stopped_after_identity_controls`，并跳过阶段 B
的 `raw_semantic_only` 与 `collaborative_only`。这两项在报告中保留为 `null`，不是运行
故障。

不覆盖阈值补跑单分支，不进入 R1.2b 自蒸馏，不对当前双视图模型进行多种子或测试集评测。
当前可信边界是：raw 语义改善 Overall/Head，但没有建立 Tail 或 cold-start 机制。
