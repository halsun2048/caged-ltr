# Yelp 数据准备

本工程明确区分论文一致版与当前官方版。复现 LLM-ESR 时使用作者发布的
`yelp_llmesr_author`；`yelp_current` 只能作为数据版本变化下的扩展实验，不能与论文表格
直接比较。

## 数据来源与本地位置

| 版本 | 来源 | 本地压缩包 | 用途 |
|---|---|---|---|
| LLM-ESR 作者版 | [官方实现](https://github.com/liuqidong07/LLM-ESR) README 中的 processed data | `data/raw/yelp/LLMESR_author_processed.zip` | 论文一致复现 |
| 当前官方版 | [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) | `data/raw/yelp/Yelp-JSON.zip` | 扩展实验 |

Yelp 官方数据受其 Open Dataset Agreement 约束。原始数据、作者 pickle 和处理后的逐条
记录均不提交 Git；仓库只保存配置、处理代码、哈希和聚合统计。

## 运行

项目环境固定后，分别执行：

```bash
uv sync --frozen
uv run python scripts/prepare_yelp_author.py
uv run python scripts/prepare_yelp.py
```

对应配置为 `configs/data/yelp_llmesr_author.yaml` 和
`configs/data/yelp_current.yaml`，汇总报告写入 `reports/data/`。重复运行会复用大小一致的
已提取文件，并重新生成确定性的 Parquet 表和数据指纹。
最终 Parquet 固定为单线程写入，以避免并行行组布局让同一逻辑数据产生不同物理哈希；
当前官方版的 JSON 解析和中间清洗仍按配置并行执行。

## 作者版验收结果

作者压缩包中的 Yelp 有序交互被原样保序，然后按用户采用 leave-two-out：最后一次为测试，
倒数第二次为验证，其余为训练。

| 指标 | 处理结果 | 论文参考 |
|---|---:|---:|
| 交互数 | 192,214 | — |
| 用户数 | 15,720 | 15,720 |
| 物品数 | 11,383 | 11,383 |
| 平均序列长度 | 12.2274 | 12.23 |
| 训练 / 验证 / 测试 | 160,774 / 15,720 / 15,720 | leave-two-out |
| 切分约束违规 | 0 | 0 |

作者包不包含原始 Yelp ID 映射和事件时间。所有外部 `.pkl` 仅记录大小与 SHA-256，准备阶段
不会反序列化；其中用户 embedding 的时间截断来源尚未证实，在完成独立泄漏审计前不得用于
正式实验。

## 当前官方版验收结果

当前官方归档按 LLM-ESR 作者代码中的 Yelp 规则处理：时间范围为 2000-01-01 至
2019-12-31，评分大于 0，用户和物品交互数均至少为 3，再按用户过滤后序列长度至少为 3。
同一用户—物品的重复评论不去重，以保持作者规则一致。

| 指标 | 结果 |
|---|---:|
| 原始评论 | 6,990,280 |
| 过滤后交互 | 4,300,562 |
| 用户 | 440,996 |
| 物品 | 139,914 |
| 训练 / 验证 / 测试 | 3,418,570 / 440,996 / 440,996 |
| 训练期零交互物品 | 2,957 |

频次分桶只使用训练交互计数，同时输出论文 Top 20%/80% 分桶与分位数
Head/Torso/Tail；训练期零交互物品单列为 `cold_start`。用户画像原料
`profile_reviews_train.parquet` 只包含训练行。

当前 business JSON 是下载时的单一快照，不是按事件时间冻结的历史元数据，因此不能把其中
的累计评论数、营业状态等字段用于声称“无未来信息”的正式特征。该限制已写入报告。

## 主要产物

- `interactions.parquet`：统一索引、序列位置、切分和频次桶；
- `users.parquet`、`items.parquet`：训练期频次、论文桶、分位数桶和 cold-start 标记；
- `sequences.parquet`：每用户训练历史、验证目标和测试目标；
- `profile_reviews_train.parquet`：仅当前官方版生成的训练期画像文本原料；
- `manifest.json`：源文件、产物哈希、统计、数据指纹和已知限制。

可提交的聚合报告为 `reports/data/yelp_llmesr_author_summary.json` 和
`reports/data/yelp_current_summary.json`。
