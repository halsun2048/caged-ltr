# FIRST R5：Listwise 教师轻量复现

## 当前状态

R5.0 本地准入已完成，尚未执行 7B 模型推理。该阶段只冻结数据、prompt、模型
身份、token 身份和扰动协议，不产生任何 FIRST 效果结论。

固定身份：

- 模型：`rryisthebest/First_Model`
- 模型与 tokenizer revision：
  `64eba9b83c174439d2b6f5d333fbb822b38d73a7`
- 作者代码：
  `gangiswag/llm-reranker@2d7cba423ad555064bdfc719313570b5f9525887`
- prompt：`first-author-alpha-v1`
- 窗口：20，step：10，上下文：4096

模型不会在本项目中训练。R5 只把公开 7B checkpoint 作为离线 Listwise 教师。

## R5.0 输入与隔离

从 MS MARCO train 中确定性抽取 100 个查询，用固定 Pyserini 2.3.0、
`msmarco-v1-passage` 和 BM25 `k1=0.9,b=0.4` 检索每个查询的 Top-20。
抽样时排除了 TREC-DL19/20 的 Query ID 和规范化文本；qrels 没有被读取。

准备命令：

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
PYSERINI_CACHE="$PWD/data/raw/prp_trec_dl/pyserini_cache" \
uv --cache-dir .uv-cache run --frozen --with pyserini==2.3.0 \
  python scripts/prepare_r4_instruction_distillation.py \
    --output-dir data/processed/r5_first_msmarco_100 \
    --report reports/data/r5_first_msmarco_100_summary.json \
    --query-count 100 --validation-count 10 --top-k 20 --seed 42 --progress
```

这里复用的是已验收的 R4 泄漏安全抽样/检索构建器，输出目录和 R5 配置彼此独立。

本地准入命令：

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_first_r5_0_local_admission.py
```

首次运行只下载 tokenizer 文件，不下载 7B 权重；之后可增加
`--local-files-only` 完全离线复跑。

## 已冻结协议

对每个 Query 均生成以下四个独立 slate：

1. BM25 正序；
2. 候选逆序；
3. seed 42 的确定性随机 permutation；
4. 候选位置不变、A–T 标识符确定性重映射。

首 token 与完整生成共用同一 chat-template prompt。首 token 模式只在 prompt
末尾追加 `[`，再读取下一位置 A–T 的 logits。完整生成必须能解析为 A–T 的
精确排列；缺失、重复或越界标识符均记为失败，不能静默补齐。

预注册的 GPU 指标包括 pair agreement、Kendall tau、归一化熵、Top1–Top2
margin、顺序/标识符/窗口边界一致性，并分别记录 prefill 与 decoding 耗时。
完整生成只对固定的 20 Query 审计子集执行。

## R5.0 结果

- 100 Query × 20 candidates；
- 四类扰动共 400 条冻结 prompt；
- A–T 均为 `[` 后的不同单 token；
- prompt token 数范围为 1,528—3,638，全部低于 4,096 上下文预留线；
- 输入与 TREC-DL19/20 Query ID、规范化文本重叠均为零；
- 输入和生成 prompt 均保存 SHA-256；
- 未加载模型权重、未使用 GPU、未读取 qrels。

机器可读结果：
`reports/experiments/r5_0_first_local_admission.json`。

## 解释边界与下一步

R5.0 只能说明协议可执行，不能说明 FIRST 的排序质量、概率校准性或鲁棒性。
首 token 与模型自身完整生成一致也不等于与真实相关性一致。

下一步 R5.1 先在单张 24GB GPU 上运行 8 Query：

- BF16 权重与 prompt 显存准入；
- 校验所有 A–T logits 可读取且有限；
- 验证完整生成解析；
- 验证 prefill/decoding 分离计时和断点缓存。

8 Query 全部通过后，才运行 100 Query 的首 token 四扰动推理，以及固定 20
Query 的完整生成对照。

## R5.1 runner

本地可先验证缓存与输出协议（不使用 GPU）：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_first_r5_1_gpu_admission.py \
  --dry-run --query-limit 8 --progress
```

真正执行以下命令时明确需要 CUDA GPU；它会加载 7B BF16 checkpoint：

```bash
uv --cache-dir .uv-cache run --frozen \
  python scripts/run_first_r5_1_gpu_admission.py \
  --query-limit 8 --variant baseline --full-generation --progress
```

输出写入 `runs/r5_1_first_gpu_admission/`，按 prompt fingerprint 断点续跑。
只有 `gpu_admission_complete=true` 才表示真实模型准入完成；dry-run 的
`all_acceptance_pass=true` 仅表示本地协议通过。
