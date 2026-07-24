# Fashion R1.3 独立确认

R1.3 用 LLM-ESR 作者发布的 Amazon Fashion 处理后资产，验证 Yelp 上观察到的
校准语义融合收益是否能跨数据集成立。Fashion 上不搜索融合方法或权重。

## 已锁定协议

- 数据：作者包中的 `fashion/inter.txt` 与 `pca64_itm_emb_np.pkl`。
- 切分：保持作者交互顺序；每用户最后一条测试、倒数第二条验证。
- 模型：SASRec、LLMInit、semantic-only、calibrated fusion。
- checkpoint：每个训练种子只按 Fashion 验证集 sampled-100 NDCG@10 early stopping。
- 融合：Yelp 验证集已锁定的 per-query z-score，语义权重 `0.25`。
- 种子：`42`、`2024`、`3407`。
- 最终测试：checkpoint 和融合设置全部固定后，使用 sampled-1000 测试一次。
- 输出：Overall、Head/Torso/Tail、cold-start，以及均值和样本标准差。

作者包实际包含 9,094 个用户、4,722 个物品；论文表格写作 9,049 个用户、
4,722 个物品。工程保留作者包全部记录，不为匹配论文表格而静默删用户，并在数据
manifest 与最终报告中记录这 45 个用户的差异。

## 执行

作者压缩包已经存在时，准备和安全转换数据：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/prepare_llmesr_author.py --dataset fashion

uv --cache-dir .uv-cache run --frozen \
  python scripts/convert_llmesr_author_semantics.py --dataset fashion
```

先运行端到端冒烟：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_fashion_r1_3.py \
  --config configs/reproduction/fashion_sasrec_smoke.yaml \
  --run-root runs/r1_3_smoke/fashion \
  --report-json /tmp/fashion_r1_3_smoke.json \
  --report-markdown /tmp/fashion_r1_3_smoke.md \
  --seeds 42 --progress
```

正式三种子实验：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_fashion_r1_3.py --progress
```

脚本可断点续跑：已完成的训练 checkpoint 与最终分数缓存会显示为 `cached`。
正式结果写入 `reports/experiments/fashion_r1_3.{json,md}`。
