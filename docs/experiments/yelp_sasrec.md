# Yelp SASRec 与 frozen semantic late fusion

本阶段实现 LLM-ESR 最小可信路线，不等同于完整 LLM-ESR。主干采用因果 SASRec，语义
增强采用固定的 late fusion；Cross-Attention、相似用户自蒸馏和完整双视图机制留在后续
R1.2。

## 实现

纯协同模型使用可训练 item ID embedding、位置 embedding 和两层 Transformer。每个位置
只能关注自己及此前位置，并使用 BPR 预测下一物品。训练负样本排除该用户的 train、valid、
test 全部已知物品。

late fusion 使用作者发布的 64 维 PCA item embedding。语义矩阵注册为不可训练 buffer，
用户语义是当前历史前缀中 item 语义向量的归一化均值：

\[
s(u,i)=s_{SASRec}(u,i)+\lambda\,
\hat{\mathbf e}_{i}^{sem\top}
\operatorname{norm}\!\left(\frac{1}{|H_u|}\sum_{j\in H_u}
\hat{\mathbf e}_j^{sem}\right).
\]

默认 `lambda=1.0`，正式实验需要只在验证集上选择该值。语义分支没有可训练参数，因此两种
模型的可训练参数量相同。作者的 [LLM-ESR 官方实现](https://github.com/liuqidong07/LLM-ESR)
采用 64 维、最大长度 200、两层 Transformer、batch 128 和 early stop 20；本工程保持这些
主要设置，但按清单改用 BPR，并显式启用 causal mask。

## 安全语义转换

作者归档中的 `pca64_itm_emb_np.pkl` 是形状 `11383×64` 的 NumPy ndarray pickle。训练代码
不直接加载 pickle。转换器用 `pickletools` 检查全部 opcode、全局名称、shape、dtype、数据
长度和源 SHA-256，然后直接从数值 buffer 构造数组，不执行 `REDUCE` 或 `BUILD`：

```bash
uv run python scripts/convert_yelp_author_semantics.py
```

输出为 `data/processed/yelp_llmesr_author/pca64_item_embeddings.npy`，只能用
`numpy.load(..., allow_pickle=False)` 读取。转换报告见
`reports/data/yelp_semantic_conversion.json`。

## 防泄漏评测协议

- 训练：只使用 `train_item_ids`，对所有可用下一物品位置计算 BPR；
- 验证：历史只有 train，目标是倒数第二次交互；
- 测试：历史为 train 加验证目标，目标是最后一次交互；
- checkpoint：只按验证集 NDCG@10 选择，测试集仅在选择后评测；
- sampled-100：目标加 100 个固定、互异且从未与该用户交互的负样本；
- 报告：Overall、用户/物品 Head/Torso/Tail/cold-start 和论文 20%/80% 桶。

## 运行

开发 smoke：

```bash
uv run python scripts/train_yelp_sasrec.py --model sasrec
uv run python scripts/train_yelp_sasrec.py \
  --model late_fusion --output-dir runs/yelp_late_fusion_smoke
```

正式配置：

```bash
uv run python scripts/train_yelp_sasrec.py \
  --config configs/reproduction/yelp_sasrec.yaml \
  --model sasrec --output-dir runs/yelp_sasrec_seed42
uv run python scripts/train_yelp_sasrec.py \
  --config configs/reproduction/yelp_sasrec.yaml \
  --model late_fusion --output-dir runs/yelp_late_fusion_seed42
```

## 全量 1-epoch 规模验收

下表使用全部 15,720 用户、11,383 物品、seed 42、CPU、sampled-100。它只证明全规模训练与
评测链路可运行；单轮结果尚未收敛，不能与论文最终表格直接比较。

| 模型 | 可训练参数 | 测试 Hit@10 | 测试 NDCG@10 | 耗时 |
|---|---:|---:|---:|---:|
| SASRec | 792,000 | 0.5112 | 0.2946 | 66.5 秒 |
| Frozen late fusion | 792,000 | 0.5462 | 0.3219 | 69.5 秒 |

late fusion 另持有 728,576 个冻结语义值。当前 Tail-item NDCG@10 仍很低
（SASRec 0.00075、late fusion 0.00157），必须完成 early stopping 和至少三个种子后才能判断
趋势。聚合记录见 `reports/experiments/yelp_sasrec_fullscale_1epoch.json`。

## 尚未完成

- 三个种子训练至验证集 early stopping；
- `lambda` 验证集网格和 LLMInit、semantic-only 对照；
- full-ranking H@10/NDCG@10；
- 收敛后 Head/Tail 权衡、显著性与参数匹配结论；
- 作者 item embedding 的事件时间 provenance 仍不足，因此本路线属于作者资产复现，不能
  宣称严格的历史时点文本无泄漏。
