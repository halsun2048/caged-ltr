# R3：Pairwise Ranking Prompting 教师复现

## R3.0 边界

R3.0 先验证 PRP 数据和控制协议，不调用真实语言模型。合成数据与
`DeterministicMockTeacher` 只用于发现方向交换、聚合、计数、恢复和保存错误，
任何指标都不能称为 PRP 论文结果或真实教师质量。

真实模型阶段必须另外记录模型名称与 revision、tokenizer 与 revision、量化方式、
prompt 名称、版本与 SHA-256，以及完整生成参数。缺失这些字段的教师标签不得进入
学生蒸馏。

## 已实现协议

- 每个无序候选对分别以 A/B、B/A 两种顺序询问；
- 两次输出选择同一候选时形成严格偏好，否则统一记为 tie；
- Allpair 使用胜者一分、tie 各半分的 Borda 聚合；
- Sliding-K 从列表尾部向前做相邻比较，默认三轮；
- 运行 BM25 正序、逆序和固定随机 permutation；
- 报告 swap agreement、tie ratio、pair coverage、pair accuracy、三元循环率、
  NDCG 和排序稳定性；
- 每完成一个 Query 追加一条 JSONL 并执行落盘，同一身份重复运行只处理未完成
  Query；
- manifest 将数据指纹、教师元数据和协议配置绑定为 SHA-256，配置变化后拒绝
  静默复用缓存。

对于每个 Query 的十个候选，Allpair 精确需要
\(10(10-1)=90\) 个有序 prompt。三种初始顺序、100 Query 共 27,000 个
Allpair prompt；三轮 Sliding-K 另需 16,200 个，总计 43,200 个。并行或 batch
只能降低 wall-clock time，不能改写该成本计数。

## 本机执行

R3.0 不训练模型，可直接运行：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_prp_r3_smoke.py --progress
```

进度条同时显示完成 Query、已处理有序 prompt、当前聚合器和耗时。中断后重复同一
命令将按 Query 恢复。逐 Query 输出保存在 `runs/prp_r3_smoke/`，正式汇总写入
`reports/experiments/prp_r3_smoke.json`。

## 进入 R3.1 的门槛

R3.0 的全部自动门槛通过后，下一步才接入真实文本候选和小模型：

1. 固定 TREC-DL/MS MARCO 查询、qrels 和 BM25 候选版本；
2. 先以小规模候选验证真实 tokenizer、输出解析和 OOM 边界；
3. 再用约 100 个 Query 运行 0.5B—3B 管线 smoke；
4. 本地小模型结果仍只代表管线验证，不冒充 FLAN-UL2 20B 忠实复现；
5. 质量门检查教师是否优于 BM25 顺序伪标签、Allpair 是否逆序稳定，以及
   Sliding-K 的质量—成本是否可接受。

## R3.0 正式 smoke 结果

100 Query、每 Query 10 个候选已完成，八项自动门槛全部通过：

| 项目 | 结果 |
|---|---:|
| Allpair 有序 prompt | 27,000 / 27,000 |
| Sliding-K 有序 prompt | 16,200 / 16,200 |
| 总有序 prompt | 43,200 / 43,200 |
| swap agreement | 0.988000 |
| tie ratio | 0.012000 |
| pair coverage | 0.992857 |
| pair accuracy | 0.980100 |
| 三元循环率 | 0.000000 |

模拟 BM25 顺序的 NDCG@10 为 `0.824543`；Allpair、三轮 Sliding-K 分别为
`0.978787/0.977132`。这只说明合成教师信号能够经过两个聚合器正确转化为排序，
不是语言模型质量结论。

Allpair 在 BM25 正序、逆序和随机 permutation 下的 NDCG@10 都为 `0.978787`，
逐 Query 排序完全一致。Sliding-K 的 NDCG@10 则为
`0.977132/0.955434/0.968662`，逆序和随机顺序的逐 Query 完全匹配率分别为
`0.00/0.03`。这符合算法边界：Allpair 穷举全部候选对，应与初始顺序无关；
Sliding-K 只做有限局部传递，保留初排依赖。该结果验证了 permutation 诊断能够
识别这种差异，不能据此决定真实模型上的质量—成本取舍。

重复执行只读取 100 条已完成 Query，显示 `43200/43200` 后直接重新汇总，没有
追加重复记录，断点身份与缓存复用验收通过。正式输出见
`reports/experiments/prp_r3_smoke.json`。

## R3.1a 官方数据冻结

R3.1a 已固定 TREC-DL 2019/2020 passage reranking 的官方查询、NIST qrels，
以及 PRP 复现使用的 Pyserini BM25 排名。候选不能按微软
`msmarco-passagetest*-top1000.tsv` 的物理行序截取：该 TSV 提供文本候选但不含
rank/score，行序不是可审计的 BM25 rank。标准 run 由 Pyserini 2.3.0 的
`msmarco-v1-passage` 预构建索引生成，参数固定为 `k1=0.9, b=0.4`。

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PYSERINI_CACHE="$PWD/data/raw/prp_trec_dl/pyserini_cache"

uv --cache-dir .uv-cache run --with pyserini==2.3.0 \
  python -m pyserini.search.lucene \
  --threads 14 --batch-size 128 --index msmarco-v1-passage \
  --topics dl19-passage \
  --output data/raw/prp_trec_dl/run.msmarco-v1-passage.bm25-default.dl19.txt \
  --bm25 --k1 0.9 --b 0.4 --hits 1000

uv --cache-dir .uv-cache run --with pyserini==2.3.0 \
  python -m pyserini.search.lucene \
  --threads 14 --batch-size 128 --index msmarco-v1-passage \
  --topics dl20-passage \
  --output data/raw/prp_trec_dl/run.msmarco-v1-passage.bm25-default.dl20.txt \
  --bm25 --k1 0.9 --b 0.4 --hits 1000

uv --cache-dir .uv-cache run --with pyserini==2.3.0 \
  python scripts/export_prp_bm25_top10.py \
  --run data/raw/prp_trec_dl/run.msmarco-v1-passage.bm25-default.dl19.txt \
  --run data/raw/prp_trec_dl/run.msmarco-v1-passage.bm25-default.dl20.txt \
  --progress

uv --cache-dir .uv-cache run --frozen \
  python scripts/prepare_prp_trec_dl.py --progress
```

Pyserini 首次执行下载约 2.02 GB 的索引并显示进度，之后复用缓存；这是 CPU
检索，不需要 GPU。官方查询和 qrels 下载支持 HTTP Range 与 `.part` 断点续传，
并在原子改名前验证 MD5。教师输入单独写入 `teacher_inputs.jsonl`，其中不含
judged、relevance 或 qrels 字段，避免推理阶段标签泄漏。数据配置见
`configs/data/prp_trec_dl_top10.yaml`，机器可读审计见
`reports/data/prp_trec_dl_summary.json`。

| 项目 | 结果 |
|---|---:|
| DL19 / DL20 judged Query | 43 / 54 |
| 候选总数 / 唯一 passage | 970 / 970 |
| Top-10 内 judged 比例 | 0.996907 |
| Top-10 内 grade ≥ 2 passage | 366 |
| 至少命中一个 grade ≥ 2 的 Query | 88 / 97 |
| DL19 BM25 `trec_eval` NDCG@10 | 0.505831 |
| DL20 BM25 `trec_eval` NDCG@10 | 0.479637 |
| 97 Query 加权均值 NDCG@10 | 0.491249 |

DL19/20 的 NDCG@10 分别精确对齐 PRP/Pyserini 公布的 `0.5058/0.4796`。
`trec_eval ndcg_cut` 使用完整 qrels、线性 graded gain；`-l 2` 对二值指标把
grade 1 视为 non-relevant，但 NDCG 仍是分级指标。报告同时保留纯 0/1、
grade 1 清零和项目通用指数 gain 三种口径，主对照使用 `bm25_trec_eval`。

这些数值只是冻结候选集上的 BM25 初排，不是 PRP 教师结果。下一步 R3.1b 才调用
真实 FLAN-T5-XL 教师；该阶段需要 24 GB NVIDIA GPU，并先做少量 pair 的
tokenizer、输出解析、显存和吞吐准入。

## R3.1b 真实 FLAN-T5-XL 教师

R3.1b 已在 RTX 4090 上完成真实 `google/flan-t5-xl` 推理。模型和 tokenizer
固定到 revision
`7d6315df2c2fb742f0f5b556879d730926ca9001`，使用论文 Appendix E.1 的 pairwise
prompt；模板 SHA-256 为
`03cbcdf0bc51111f3ac5f2ad1609e2a86892a33c3efb31c8b7c95bd6b8b0a827`。
主实验比较完整目标 `Passage A`、`Passage B` 的序列 log-likelihood，不采样，
`float16`、最大输入 512 token、batch 8。

GPU 环境不能使用仓库 `uv.lock` 中显式锁定的 CPU PyTorch。需在保留 CUDA
PyTorch 的独立环境中直接执行：

```bash
PYTHONPATH=src .venv-gpu/bin/python \
  scripts/run_prp_r3_1b_flan_t5.py \
  --queries 97 \
  --batch-size 8 \
  --max-ordered-prompts 0 \
  --scoring-mode likelihood \
  --output-dir runs/prp_r3_1b_flan_t5_xl_97q \
  --progress
```

脚本按有序 pair 逐条追加 JSONL 并 `fsync`；重复执行会校验 manifest 后只补齐
缺失 pair。教师输入不含 qrels 字段，且程序只在 8,730 个 prompt 全部完成后
读取独立 qrels 文件。1 Query/32 prompt 的 likelihood 准入和 8 prompt 的
generation 严格解析审计均为零截断、零非法输出；generation 原始输出全部严格为
`Passage A` 或 `Passage B`。

97 Query 的全量 Top10 结果如下：

| 数据 | Query | BM25 NDCG@10 | FLAN-T5-XL PRP | 绝对增益 |
|---|---:|---:|---:|---:|
| DL19 | 43 | 0.505831 | 0.547891 | +0.042060 |
| DL20 | 54 | 0.479637 | 0.519842 | +0.040205 |
| 合计 | 97 | 0.491249 | 0.532276 | +0.041028 |

两个年度方向一致，说明真实教师能在冻结 BM25 Top10 内提供有效的重排序信号，而
不是只依赖某一年度。平均 swap agreement 为 `0.805956`；其余 `0.194044`
双向比较发生冲突并按预注册协议记为 tie。这里的 tie 是 A/B 对调后选择了不同
候选的冲突，不是 likelihood 数值相等。8,730 个单向输出中没有 raw tie/invalid，
也没有输入被 512-token 上限截断。

本次全量推理耗时 `249.51s`，吞吐 `34.99 prompt/s`；CUDA peak allocated /
reserved 分别为 `8.55/13.57 GB`，24 GB 显存余量充足。机器可读结果见
`reports/experiments/prp_r3_1b_flan_t5_xl_top10.json`，复现配置见
`configs/reproduction/prp_r3_1b_flan_t5_xl.yaml`。

需要严格限制结论：论文 Table 2 的 FLAN-T5-XL PRP-Allpair
`0.6975/0.6812` 来自 BM25 Top100 重排序，本实验只重排 Top10。候选深度不同会
改变可达到的上限、pair 数和聚合行为，因此当前结果是“真实教师 Top10 管线与质量
验证”，不能宣称数值复现论文 Top100。若要进行论文级对照，下一步应冻结 Top100
文本并运行每 Query 9,900 个有序 prompt；在查看该结果前不得用 test qrels 调整
prompt、tie 规则或聚合器。

## R3.1c Top100 论文深度复现

R3.1c 已在 8 张 RTX 4090 上完成冻结 BM25 Top100 的真实
`google/flan-t5-xl` 双向 Allpair 推理。97 个 Query 各有 100 个候选，每个
Query 生成 \(100\times99=9,900\) 个有序 prompt，总计 `960,300` 个。8 个
GPU worker 按 Query 分片，全部正常退出；合并前各分片禁止读取 qrels，合并后
校验得到 `960,300` 个全局唯一 cache key，随后才执行一次正式评测。

| 数据 | Query | BM25 NDCG@10 | 本次 PRP-Allpair | 论文值 | 与论文差值 |
|---|---:|---:|---:|---:|---:|
| DL19 | 43 | 0.505831 | 0.709431 | 0.6975 | +0.011931 |
| DL20 | 54 | 0.479637 | 0.681258 | 0.6812 | +0.000058 |
| 合计 | 97 | 0.491249 | 0.693747 | — | +0.202498（相对 BM25） |

两个年度均落在预注册的论文值绝对误差 `±0.02` 内，且相对 BM25 的绝对提升分别
为 `+0.203600/+0.201621`。因此可以得出“FLAN-T5-XL、作者 prompt、Top100
双向 Allpair 的论文深度数值已复现”；不能由此推出模型在其他数据集或在线场景
同样有效。

平均 swap agreement 为 `0.794987`，其余 `0.205013` 的无序 pair 在 A/B
交换后发生冲突。这说明双向询问不是可删除的冗余步骤，也为后续置信度路由提供了
直接动机。原始 likelihood 中只有 9 个精确等分，非法输出为 0。

预注册协议仍有一项未通过：`960,300` 个输入中有 184 个触发 512-token 截断，
占 `0.0192%`，而门槛要求 0。因此当前状态应写作“数值复现通过，截断敏感性审计
待完成”，不能把所有验收项标成通过。下一步先在隔离缓存中仅用更大 token 上限
重算这 184 个有序 prompt，并在不根据 qrels 调参的前提下重聚合；若排序和指标
稳定，再进入真实 Sliding-10 的质量—成本对照。机器可读结论见
`reports/experiments/prp_r3_1c_flan_t5_xl_top100.json`。
