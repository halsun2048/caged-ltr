# Yelp R1.2a 双视图结构对照

## 目标

R1.2a 先复现 LLM-ESR 的双视图主体，不加入语义近邻自蒸馏，以便分别判断共享编码器、
双向 Cross-Attention 和额外参数容量是否有效。正式实验只使用 seed 42 验证集选择结构，
不访问测试集。

## 与作者实现对齐

- 语义分支读取作者发布的原始 1536 维 item embedding，并保持冻结；
- adapter 为 `1536 → 768 → 64` 的两个线性层，中间没有显式激活；
- 协同分支使用作者 PCA64 item embedding 初始化，训练中可更新；
- 两条分支共用位置 embedding，默认共用同一个 SASRec 序列编码器；
- Cross-Attention 为语义查询协同、协同查询语义两个方向，输出替换各自原输入；
- 最终分数是两个视图的内积之和，使用下一物品 BPR 训练。

参考：[LLM-ESR 官方仓库](https://github.com/liuqidong07/LLM-ESR)及其
[`DualLLMSRS.py`](https://github.com/liuqidong07/LLM-ESR/blob/main/models/DualLLMSRS.py)。

作者的 Cross-Attention 只传入 padding mask。序列训练中这会允许当前位置读取后续位置，
因此本工程增加 causal mask；当前实现是“泄漏安全的结构复现”，不宣称与作者代码逐值一致。

## 四组结构对照

| 变体 | Cross-Attention | 编码器共享 | 等参数位置变换 |
|---|:---:|:---:|:---:|
| `dual_view_no_ca` | 否 | 是 | 否 |
| `dual_view` | 双向、因果 | 是 | 否 |
| `dual_view_unshared` | 双向、因果 | 否 | 否 |
| `dual_view_capacity` | 否 | 是 | 是 |

`dual_view_capacity` 每个视图使用四个同位置线性映射，其可训练参数数目与一条
Cross-Attention 完全相同，但不允许视图或时间位置之间交换信息。它用于区分交互机制收益
与单纯增加参数的收益。

## 数据转换

原始作者 embedding 是 ndarray pickle。训练不直接反序列化 pickle，而是先用受限 opcode
解码器转换：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/convert_llmesr_raw_embeddings.py --dataset yelp --kind item
```

输出为 `data/processed/yelp_llmesr_author/raw_item_embeddings.npy`，转换来源、SHA-256、
形状和 dtype 记录在 `reports/data/yelp_raw_item_conversion.json`。

## 正式运行

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_yelp_r1_2a_dual_view.py --progress
```

脚本按 epoch 显示进度、当前最好验证 NDCG、early-stopping stale 计数和预估停止耗时。
完整变体已有 `summary.json` 时会校验并复用；所有输出都锁定为 validation-only。

完成后的主要判断顺序：

1. `dual_view` 是否优于 `dual_view_no_ca`；
2. `dual_view` 是否优于等参数量的 `dual_view_capacity`；
3. 共享编码器相对 `dual_view_unshared` 的效果；
4. 只有结构证据成立后，才进入 R1.2b 的语义近邻自蒸馏。

## seed 42 正式验证结果

四个模型均训练至 validation early stopping，测试集未访问。

| 变体 | Overall | Head | Torso | Tail | Cold-start | 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|
| `dual_view_no_ca` | **0.426584** | **0.542450** | **0.181464** | **0.031652** | 0.000000 | 38 |
| `dual_view` | 0.403566 | 0.524208 | 0.135019 | 0.022243 | 0.000000 | 37 |
| `dual_view_unshared` | 0.397480 | 0.517180 | 0.128824 | 0.024748 | 0.000000 | 65 |
| `dual_view_capacity` | 0.400849 | 0.528510 | 0.110884 | 0.008082 | 0.004463 | 10 |

既有 seed 42 验证基线中，LLMInit Overall/Tail 为 `0.419476/0.020106`，SASRec
为 `0.394789/0.019838`。因此 no-CA 双视图相对 LLMInit 的 Overall 和 Tail 分别增加
`0.007109` 和 `0.011546`。

因果 Cross-Attention 相对 no-CA 的 Overall、Head、Torso、Tail 分别变化
`-0.023018`、`-0.018241`、`-0.046446`、`-0.009408`，主验收失败。它相对等参数
位置变换的 Overall 提高 `0.002717`，说明交互并非完全没有信号，但不足以抵消用
Cross-Attention 输出直接替换原视图造成的损失。共享编码器相对非共享编码器 Overall
提高 `0.006085`，可以保留。

## 当前边界与下一步

本阶段只允许得出“no-CA 双视图在 seed 42 验证集最好”和“当前因果 CA 结构不成立”。
no-CA 模型约有 202 万可训练参数，明显多于约 79 万参数的 LLMInit；原始高维 embedding
也可能只提供可利用的随机特征。因此还不能把增益归因于真实语义。

不对当前 CA 做三种子或测试集评测。随后完成的 R1.2a-control 包含：

1. shuffled raw semantic；
2. 与真实向量逐维统计匹配的 random raw semantic；
3. 删除语义分支；
4. 删除协同分支。

身份门控结果见
[`yelp_r1_2a_controls.md`](yelp_r1_2a_controls.md)：真实语义虽改善 Overall/Head，
但 Tail 略低于 shuffled 和 matched-random，门控失败；单分支按预注册规则跳过，R1.2b
语义近邻自蒸馏停止。
