# R4 Instruction Distillation

## R4.0 本地准入

R3.2 已确定 R4 正式教师继续使用双向 Allpair FLAN-T5-XL；Sliding-10 只保留为
质量—成本对照。R4.0 不做模型训练，只冻结数据身份、隔离测试集并验收学生接口。

数据使用官方 MS MARCO `queries.tar.gz` 中的 `queries.train.tsv`。按 seed 42 的
SHA-256 顺序无放回选择 1,000 个唯一规范化查询，其中 900 个训练、100 个验证。
选择时同时排除 TREC-DL19/20 的 Query ID 与规范化精确文本；qrels 没有进入选择、
检索或教师输入。

已有的 Pyserini 2.3.0 `msmarco-v1-passage` 索引使用固定
BM25 `k1=0.9,b=0.4` 检索 Top-10，共生成 10,000 个候选。冻结身份见
`reports/data/r4_msmarco_1k_summary.json`。

作者公开代码（固定 revision
`0d62bc3855c7c118048a7c47c18e719b938e291a`）的 Pointwise 学生是
`microsoft/deberta-v3-base` 的单 logit
cross-encoder，并非 FLAN-T5 生成式学生。当前实现因此保留 FLAN-T5-XL 作为
Pairwise 教师，使用 DeBERTa-v3-base 作为 Pointwise RankNet 学生；最大长度 500、
AdamW `5e-5`、全局有效 Query Batch 8、训练 3 epoch，与作者
`specialization.py` 对齐。正式训练使用 8 卡 DDP，每卡 Query Batch 1、梯度累积
1，保持相同的全局有效 Batch。

四组正式对照为：

1. vanilla Pointwise：未训练的预训练 checkpoint；
2. BM25 RankNet：按 BM25 次序生成伪标签；
3. random RankNet：逐 Query 固定随机全排列；
4. PRP RankNet：双向 Allpair 教师的 Borda 全排序。

测试集只允许在验证集选定 checkpoint 后访问一次。训练阶段的 checkpoint 指标是
验证查询相对各自伪教师次序的 NDCG@10，不读取 TREC-DL qrels。

本地准入命令：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_r4_0_local_admission.py
```

准入报告写入 `reports/experiments/r4_0_local_admission.json`。通过后下一阶段需要
完成 1,000 Query × 90 ordered pairs，即 90,000 个冻结教师 prompt；教师完成前
不能启动 PRP 学生，但 BM25 与 random 学生可独立训练。

## R4.1 教师与学生训练

8 卡正式运行完成了全部 90,000 个教师 prompt，截断 0、无效输出 0、精确似然
平局 1；双向交换一致率均值为 0.806689。Borda 排序导出为 1,000 个 Query、
10,000 个候选的 RankNet 标签。

三组训练控制均以 seed 42 训练 3 epoch，仅按 100 个验证 Query 相对各自伪教师
次序的 NDCG@10 选择 checkpoint：

| 控制 | 最佳 epoch | 验证 teacher-NDCG@10 |
| --- | ---: | ---: |
| BM25 RankNet | 3 | 0.936521 |
| random RankNet | 3 | 0.835442 |
| PRP Allpair RankNet | 3 | 0.949094 |

这些值衡量的是每个模型对其自身伪教师标签的拟合度；不同伪教师之间的
teacher-NDCG 不可直接当作真实检索质量比较。它们证明 PRP 信号可被学生稳定学习，
但 PRP 是否优于 BM25、random 与 vanilla Pointwise，必须由锁定 checkpoint 后的
TREC-DL19/20 test-once 评价决定。

正式报告为
`reports/experiments/r4_1_teacher_student_training.json`。教师、学生训练和
checkpoint 选择阶段均保持 `qrels_accessed=false`、`test_accessed=false`。

## R4.2 锁定 test-once 评价

正式评价使用 TREC-DL19/20 的 97 个 Query 和固定 BM25 Top-100。四组 checkpoint
在读取 qrels 前已经锁定；GPU 预测只读取
`data/processed/prp_trec_dl_top100/teacher_inputs.jsonl`，其结构不含 judged、
relevance 或 qrels 字段。四组各 9,700 条预测先合并、校验并冻结 SHA-256，随后
评价子命令显式授权并读取 qrels 一次。访问回执阻止后续重复读取。

四模型使用 8 张 RTX 4090 并行推理，每个模型两个独立分片；全部推理墙钟时间
24.03 秒，8 个 worker 零失败。主结果采用完整官方 qrels 的线性增益
trec_eval NDCG@10：

| 方法 | 整体 | DL19 | DL20 | 平均秒/Query |
| --- | ---: | ---: | ---: | ---: |
| 初始 BM25 | 0.491249 | 0.505831 | 0.479637 | — |
| vanilla Pointwise | 0.136074 | 0.164292 | 0.113604 | 0.1918 |
| random RankNet | 0.162119 | 0.182632 | 0.145785 | 0.1876 |
| BM25 RankNet | 0.476489 | 0.502845 | 0.455502 | 0.1824 |
| PRP Allpair RankNet | **0.639791** | **0.673911** | **0.612621** | 0.1840 |
| FLAN-T5-XL Allpair 教师 | 0.693747 | 0.709431 | 0.681258 | 165.07 |

PRP 相对 BM25 RankNet 的绝对增益为 0.163302；10,000 次 paired bootstrap 的
95% CI 为 `[0.124484, 0.202811]`，双侧 `p=0.000200`。相对 vanilla 与 random
的增益分别为 0.503717 与 0.477672，三个对照在 DL19、DL20 上均保持同方向。

PRP 学生保留教师相对初始 BM25 增益的 73.35%，与教师仍相差 0.053956 NDCG@10。
因此结论是“高效伪标签适配”，不是无训练能力，也不是完全替代教师。PRP 的
Top-100 推理为 100 次 Pointwise 打分，而双向 Allpair 为 9,900 个逻辑 prompt，
复杂度从 \(O(N^2)\) 降为 \(O(N)\)，并非参数量默认缩小；在当前 RTX 4090 worker
参考下实测约加速 897 倍。

未经测试集校准的 raw sigmoid 校准结果同样支持 PRP 信号有效：PRP AUC 为
0.870301、Brier 为 0.170450、ECE 为 0.182147，均优于其他三个 Pointwise
对照。机器可读报告与唯一访问回执分别为
`reports/experiments/r4_2_test_once.json` 和
`reports/experiments/r4_2_test_once_access_receipt.json`。
