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
