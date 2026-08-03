# 统计学本科毕业论文路线

建议题目：**基于置信度感知路由的语义排序系统质量—成本权衡研究**。

## 研究问题

1. MiniLM 学生模型能否接近 FIRST 的排序质量？
2. Gain-aware/Tail-floor Gate 能否减少 FIRST 调用并保持质量？
3. 增益来自 Head、Torso 还是 Tail？
4. 多随机种子和 bootstrap 区间下结论是否稳定？

## 冻结证据

- 数据划分、seed、checkpoint 和 test guard 不再修改；
- untouched test 不再访问；
- 主结果：`reports/experiments/mind_r8_11_tail_floor.json`；
- 学生模型：`reports/experiments/mind_r12_dev_student.json`、`mind_r12_confirm_student.json`；
- Gate 稳定性：`reports/experiments/mind_r13_oof_gate.json`；
- 跨切分与效率：`mind_r13_cross_split_stability.json`、`mind_r13_efficiency_summary.json`。

## 论文表述边界

结论仅适用于当前离线数据、预注册划分和冻结 replay 证据；不得表述为真实线上 CTR、商业收益或真实用户 A/B 结论。MCP、Streamlit、Docker 只作为工程附录，不进入统计主假设。

## 建议章节

1. 问题与背景；
2. 数据、划分与指标；
3. 学生模型与置信度路由方法；
4. 多种子、Head/Torso/Tail 和 bootstrap 结果；
5. 质量—成本权衡、失败结果与限制；
6. 结论与可复现说明。
