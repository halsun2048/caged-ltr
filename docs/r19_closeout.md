# R19 Gate 收尾记录

R19 修复了 R18 中训练端与服务端 Gate 特征不一致的问题。特征提取现在由
`src/caged_ltr/r18_gate_features.py` 统一实现，并带有版本号
`r19.0-exact-v1`。服务端仍兼容旧的 R18 manifest，但 R19 manifest 会拒绝
特征列表不匹配的加载。

## R19.1：五折 OOF

训练脚本：

```bash
PYTHONPATH=src python -u scripts/train_r19_post_student_gate_oof.py --progress
```

结果保存在 `artifacts/r19_post_student_gate_oof.json`。仅使用 R12 dev 的
10,000 条 query；按 query-id 的 SHA-256 稳定分为五折，预算固定为 40%。
OOF NDCG@10 为 0.60451，FIRST 调用率 40%。Tail 调用率 46.82%，Head
25.65%，没有访问 confirm 或 large-test。

## R19.2：固定 confirm

```bash
PYTHONPATH=src python -u scripts/evaluate_r19_gate_confirm.py --progress
```

该步骤直接读取已冻结的 R19 dev manifest，不搜索阈值。confirm NDCG@10
为 0.60518，Student 为 0.52801，FIRST 上限为 0.64553，调用率 39.49%。
Tail NDCG@10 为 0.53264，调用率 47.42%。large-test 仍保持锁定。

## 当前限制

- 这是离线 query-level Gate 证据，不等价于线上业务 A/B 提升。
- OOF 标签来自已有 FIRST 与 Student 结果；尚未在新的跨域数据集重复验证。
- 线上服务的 batch endpoint 当前为顺序执行，P50/P95/P99 需要在目标部署环境
  重新压测，不能用离线耗时替代。
- FIRST 仍为 replay/适配器接口，真实供应商成本、限流和配额需单独接入验证。

## 复现边界

large-test 没有被 R19 访问；manifest 中记录了数据源哈希、特征版本、预算、
阈值和边界状态。大型模型权重不提交 Git，只保留本地 checkpoint 与 SHA-256。
