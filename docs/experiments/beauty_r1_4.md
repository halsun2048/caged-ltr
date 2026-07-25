# Beauty R1.4 置信感知语义融合

R1.4 的目标是检验 frozen semantic late fusion 的长尾收益，能否在不明显损害
总体排序质量的前提下迁移到第三个数据集。

## 锁定规则

所有融合超参数只在 Yelp 验证集选择，未访问 Yelp 或 Beauty 测试集：

- 基础 late-fusion：逐查询 z-score，语义权重 0.25。
- 协同不确定性：`1 / (1 + top1 - top2 margin)`。
- 候选稀有度：Head 0、Torso 0.5、Tail/Cold-start 1。
- 门控残差权重候选：0、0.05、0.1、0.25、0.5、1、2。
- 可行性约束：总体 NDCG@10 相对基础融合下降不超过 0.002。
- 可行集合内依次最大化 Tail、Torso、Overall NDCG@10。

Yelp seed 42 选择残差权重 0.1；在 42、2024、3407 三个种子上，门控相对固定
融合的 Tail NDCG@10 均提升，且真实语义门控均优于 shuffled 语义门控。三种子
平均 Overall 从 0.428054 变为 0.427190，Tail 从 0.045787 变为 0.056030。
完整验证结果在 `reports/experiments/yelp_r1_4_gate_validation.json`。

## Beauty 数据边界

作者包包含 52,204 用户、57,289 物品和 394,908 条交互，与论文表格一致。作者
代码将不足三次交互的 1,706 个用户全部留在训练集，因此正式验证和测试各包含
50,498 个用户。用户语义向量的时间边界未建立，本实验不使用用户语义向量。

## 正式对照

每个种子训练：

- SASRec；
- 使用真实物品语义初始化的 LLMInit；
- 使用固定打乱物品语义初始化的 shuffled LLMInit。

冻结语义推理还报告 semantic-only、固定 late fusion、置信感知门控和 shuffled
语义门控。Beauty 不重新选择任何融合权重。checkpoint 只由验证集确定，全部
设置锁定后，以固定 seed 的目标物品加 1,000 个未见负例测试一次。该评测属于
sampled-1000，不表述为 full-catalog。

## 执行

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_beauty_r1_4.py --progress
```

运行目录为 `runs/r1_4/beauty`。已有且哈希一致的训练和测试缓存会自动复用。
最终报告写入 `reports/experiments/beauty_r1_4.json` 和同名 Markdown 文件。
