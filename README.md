# CAGED-LTR

面向长尾搜索广告的置信度感知语义—协同增强与多目标排序蒸馏实验工程。

当前阶段聚焦可复现基础设施和 R0 公共管线。研究路线与验收条件见
[`docs/清单/00_完整复现与新实验清单.md`](docs/清单/00_完整复现与新实验清单.md)。

## 环境

项目固定使用 Python 3.12 和 CPU 版 PyTorch；本地环境用于数据处理、轻量模型、
校准和学生实验。教师模型标签应在服务器生成后持久化回传。

```bash
uv sync --frozen
uv run pytest
uv run ruff check .
uv run caged-env --output reports/environment/local_baseline.json
```

`uv.lock` 是唯一依赖解析基准。修改依赖后需显式执行 `uv lock`，并同时提交
`pyproject.toml` 与 `uv.lock`。

## 目录

```text
configs/{data,reproduction,experiments}/  数据、复现和新实验配置
data/{raw,interim,processed,teacher_labels}/  本地数据（内容不入库）
src/caged_ltr/  数据、特征、模型、教师、损失、去偏、校准和评测代码
scripts/         可重复执行的入口脚本
tests/           单元测试与回归测试
runs/            每次运行的配置、元数据和逐样本输出（内容不入库）
reports/         脚本生成的表格、图片、失败记录与环境快照
```

## 可复现约定

- 开发结果至少使用种子 `42`、`2024`、`3407`，核心结果增加 `7`、`1009`。
- 数据、语义画像和 embedding 必须按训练时点截断，禁止未来信息泄漏。
- 每次正式运行保存解析后的配置、Git commit、数据指纹、模型版本和 prompt 版本。
- 原始数据、密钥和大型模型权重不得提交到仓库。

## R0 公共管线

R0 已提供统一的 request/candidate-list 数据格式、AUC/GAUC/MRR/Recall/NDCG、
LogLoss/Brier/ECE、频次分桶、BCE/RankNet/Listwise KL，以及 MLP、DCN-v2、
LightGBM LambdaMART 三个学生基线。可用合成数据执行完整验收：

```bash
uv run caged-r0-smoke --output-dir runs/r0_smoke --seed 42
```

该命令会保存解析后的配置和数据指纹、运行环境、逐样本 Parquet、
Overall/Head/Torso/Tail 指标表，以及 P50/P95/P99、QPS 和参数量报告。

## Yelp 数据

Yelp 已提供两条明确分离的数据路线：

- `yelp_llmesr_author`：LLM-ESR 作者发布的论文一致版，作为复现实验主数据；
- `yelp_current`：当前 Yelp 官方快照按作者规则重处理，只作为新版扩展实验。

原始数据和 Parquet 产物均被 Git 忽略。准备好对应压缩包后运行：

```bash
uv run python scripts/prepare_yelp_author.py
uv run python scripts/prepare_yelp.py
```

作者版得到 192,214 条交互、15,720 用户和 11,383 物品，与论文统计一致。
当前官方版得到 4,300,562 条过滤后交互、440,996 用户和 139,914 物品。
下载方式、版本边界、切分规则、安全限制和产物说明见
[`docs/data/yelp.md`](docs/data/yelp.md)。

## Yelp SASRec

作者版 Yelp 已接入因果 SASRec、BPR 训练和 frozen semantic late fusion。先安全转换
作者的 legacy NumPy pickle，再运行缩小版验收：

```bash
uv run python scripts/convert_yelp_author_semantics.py
uv run python scripts/train_yelp_sasrec.py --model sasrec
uv run python scripts/train_yelp_sasrec.py \
  --model late_fusion --output-dir runs/yelp_late_fusion_smoke
```

正式配置为 `configs/reproduction/yelp_sasrec.yaml`。实现边界、评测协议和当前全量
1-epoch 规模验收结果见 [`docs/experiments/yelp_sasrec.md`](docs/experiments/yelp_sasrec.md)。

## Fashion R1.3

第二数据集确认固定使用 Yelp 验证集选出的 per-query z-score 与语义权重 `0.25`，
Fashion 不重新调融合参数。数据准备、两模型三种子训练和 sampled-1000 最终测试：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/prepare_llmesr_author.py --dataset fashion
uv --cache-dir .uv-cache run --frozen \
  python scripts/convert_llmesr_author_semantics.py --dataset fashion
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_fashion_r1_3.py --progress
```

作者包的 Fashion 用户数为 9,094，论文表格为 9,049；工程保留全部作者包记录并
显式报告差异。完整锁定协议见
[`docs/experiments/fashion_r1_3.md`](docs/experiments/fashion_r1_3.md)。

R1.3b 进一步提供 shuffled/matched-random 语义对照、完整 4,722 物品目录评测和
LLMInit embedding 漂移分析：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/prepare_fashion_semantic_controls.py
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_fashion_r1_3b_semantic_audit.py
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_fashion_r1_3b_retrain_controls.py
```

审计协议与已知来源边界见
[`docs/experiments/fashion_r1_3b.md`](docs/experiments/fashion_r1_3b.md)。

## Beauty R1.4

R1.4 在 Yelp 验证集锁定置信感知门控：固定融合权重 `0.25`，再以 `0.1`
追加由协同不确定性和物品稀有度控制的语义残差。Beauty 不调参，正式运行同时
训练 SASRec、真实 LLMInit 和 shuffled LLMInit 三种协同对照：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_beauty_r1_4.py --progress
```

脚本按模型和种子复用已完成 checkpoint，并在训练与最终 sampled-1000 测试阶段
显示进度条。作者版 Beauty 的 52,204 用户和 57,289 物品与论文一致；其中
1,706 个不足三次交互的用户依作者代码保留为仅训练用户。协议和运行状态见
[`docs/experiments/beauty_r1_4.md`](docs/experiments/beauty_r1_4.md)。

三种子正式结果中，置信门控 NDCG@10 为 `0.160692 ± 0.001793`，超过
LLMInit 的 `0.143487 ± 0.001889` 和固定融合的 `0.159443 ± 0.001792`；
Tail 相对固定融合获得 `+0.007048` 绝对增益。完整输出见
`reports/experiments/beauty_r1_4.json`。

R1.5 进一步复用同一组 checkpoint，对全部 57,289 个物品执行完整目录排名：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_beauty_r1_5_full_catalog.py
```

门控相对固定融合仍在三个种子上方向一致，但 Overall/Tail NDCG@10 的绝对增益
缩小到 `+0.000164`/`+0.000644`，Tail 未达到预设 `0.005` 实际规模阈值。
协议敏感性和配对置信区间见
[`docs/experiments/beauty_r1_5.md`](docs/experiments/beauty_r1_5.md)。

## RLMRec R2

R2 使用作者公开 Yelp split 搭建 LightGCN、semantic-only、RLMRec-Con 和
shuffled-Con。外部 SciPy/NumPy pickle 通过非执行式 allowlist 解码；公开
profile embedding 的训练截止时间不可验证，因此本地结果明确标记为 PCA64
结构复现，而不是严格无泄漏数值复现。

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/prepare_rlmrec_yelp.py
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_rlmrec_r2.py --smoke --progress
```

正式训练支持每 epoch checkpoint、断点续跑以及训练/全目录评测进度条。协议、
作者代码差异和首轮正式命令见
[`docs/experiments/rlmrec_r2.md`](docs/experiments/rlmrec_r2.md)。

Seed 42 中 LightGCN 的 Recall/NDCG@20 为 `0.115823/0.073247`，基本重现论文
`0.1157/0.0733`；PCA64 RLMRec-Con 为 `0.116277/0.073074`，未重现论文的
整体增益，因此暂不扩展三种子。

R2.1 恢复作者 batch 4096 后，LightGCN 为 `0.117697/0.073911`，RLMRec-Con
为 `0.116937/0.072712`，两个整体指标均低于同 batch 基线。batch 假设已排除，
不扩展本地三种子；仅保留原始 1536 维 GPU 忠实审计。

R2.2 通过 `--semantic-space raw1536 --device cuda` 选择作者公开原始语义资产。
仓库锁定的 PyTorch 是 CPU 版，正式命令必须在外部 CUDA PyTorch 环境执行。
RTX 4090 上 seed 42 的 LightGCN 与 RLMRec-Con Recall/NDCG@20 分别为
`0.117730/0.073781` 和 `0.122898/0.077361`，基本恢复论文整体结果；真实语义
也显著超过 shuffled-Con 的 `0.101566/0.063230`。但 Tail NDCG@20 绝对增益
仅 `+0.000012`，未达到 `0.001` 预设门槛，因此 R2 不扩展三种子，也不据此
宣称长尾改善。

## PRP R3

R3.0 提供双向 A/B 比较、严格一致偏好、Allpair/Borda、Sliding-K、循环与顺序
稳定性诊断，以及按 Query 断点恢复。首轮使用确定性模拟教师验证 100 Query 管线，
明确不作为真实 PRP 结果：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_prp_r3_smoke.py --progress
```

协议、成本计数和真实模型准入门槛见
[`docs/experiments/prp_r3.md`](docs/experiments/prp_r3.md)。

100 Query 正式 smoke 已完成，43,200 个有序 prompt 计数准确，swap agreement
为 `0.988`，Allpair 在正序、逆序和随机初排下逐 Query 完全一致。以上仅为模拟
教师管线验收；真实教师质量从 R3.1 开始评估。
