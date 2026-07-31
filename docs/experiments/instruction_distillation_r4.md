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
