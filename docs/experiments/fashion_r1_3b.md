# Fashion R1.3b 语义有效性审计

本阶段验证 R1.3 中较强的 semantic-only 结果究竟来自真实物品语义、向量尺度，
还是 sampled-negative 候选构造。

## 已完成

- 审计作者 `get_item_embedding.ipynb`：
  - 模型声明为 `text-embedding-ada-002`；
  - item prompt 使用 title、brand、date、price、feature、description；
  - item prompt 不包含用户交互或测试历史；
  - PCA-64 在完整物品目录上拟合；
  - 当前实验未使用作者 user embedding。
- 固定 seed `20240725` 生成两类控制：
  - `shuffled`：打乱物品 ID 与语义向量对应，完整保留向量和范数分布；
  - `matched_random`：逐维匹配真实向量的均值与标准差。
- full-catalog 测试：
  - 对全部 4,722 个物品排序；
  - 测试历史仅使用 train 与 validation target；
  - 屏蔽已观察历史，但始终保留测试目标；
  - per-query z-score 只在可推荐物品上计算。
- 计算 LLMInit 初始向量与最终 checkpoint 的逐物品 cosine 和相对 L2 漂移。

## 当前结果

真实语义融合 full-catalog NDCG@10 为 `0.337715`，LLMInit 为 `0.323864`；
三个种子的 Overall 和 Tail 增益方向均为正。真实 semantic-only 为 `0.306430`，
而 shuffled 和 matched-random 分别只有 `0.062626`、`0.077340`，说明真实
物品—语义对应关系具有信息量。

full-catalog 下 SASRec NDCG@10 为 `0.329204`，高于 LLMInit 的 `0.323864`。
因此 sampled-1000 下“LLMInit 优于 SASRec”的结论不能推广到完整目录。

重训控制进一步显示：

- shuffled LLMInit：`0.347501`；
- matched-random LLMInit：`0.333709`；
- 真实 LLMInit：`0.323864`。

因此真实语义初始化没有优于控制，LLMInit 不能作为真实语义有效性的证据。
真实融合 `0.337715` 高于 shuffled fusion `0.336702` 和 matched-random fusion
`0.320860`，但仍低于 shuffled LLMInit。真实融合相对 shuffled LLMInit 的优势主要
出现在 Tail、Torso 和 cold-start，而不是 Overall。

## 来源边界

作者包没有原始 Amazon 元数据快照、抓取时间和 raw ID map，无法独立确认每个字段
在推荐时点前已经可见。prompt 还将 `date` 占位符描述成了 `score`。所以目前只能
确认 item prompt 不直接使用交互，不能给出“严格无未来信息”的最终结论。

## 执行

重新生成控制向量：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/prepare_fashion_semantic_controls.py
```

对现有真实语义 checkpoint 做 full-catalog 与推理支路控制：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_fashion_r1_3b_semantic_audit.py
```

正式重训 shuffled 与 matched-random LLMInit（2 种控制 × 3 个种子）：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_fashion_r1_3b_retrain_controls.py
```

三个入口均可重复执行；完整任务与测试缓存会显示为 `cached`。
