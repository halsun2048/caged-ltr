# R20–R24 准入式执行结论

## 结论

R20 没有产生可进入 confirm 的 Gate。固定 40% FIRST 预算下，最好的 OOF
候选是需要 `frequency_bucket` 元数据的 Tail-floor 0.90，NDCG@10 为
0.62258；全量 FIRST 为 0.64534，差距 0.02275，超过预注册上限 0.01。
可直接部署的 Logistic Gate 为 0.60415，差距 0.04119。

因此没有候选被选中，最终脚本没有在 confirm 上执行评分，也没有访问
large-test。首版实现曾在准入判断前加载 confirm，但没有利用 confirm 评分、
选模或调阈值；该实现缺陷已修复并保留在协议审计字段中。

## R21 诊断

R21 只分析 dev OOF 结果：

- Logistic Gate 相对 Student 提升 0.07180，95% CI `[0.06728, 0.07626]`；
- Logistic Gate 相对 FIRST 下降 0.04119，95% CI
  `[-0.04649, -0.03613]`；
- Tail-floor 0.90 相对 FIRST 下降 0.02275，95% CI
  `[-0.02746, -0.01798]`；
- Tail-floor 的 Tail 差距只有 0.00483，但代价是 Torso 下降 0.10864。

失败原因不是 Tail 保护不足，而是在 40% 总预算下将 90% Tail 路由给 FIRST
后，Torso 获得的预算不足。ExtraTrees gain 的 MAE 为 0.2514，也表明现有
13 个在线特征不足以精确预测逐 query 的 FIRST 增益。

## R22 与 R23

R22 新 GPU 压测和 R23 新跨域数据集按准入规则跳过。原因是统一 Gate 尚未在
in-domain OOF 达标；继续消耗 GPU 或访问新数据不能修复当前决策特征的识别
能力。R13 真实服务延迟和 R14 NFCorpus 审计继续作为历史参考，但不被重新
包装为 R20 Gate 的证据。

## R24 工程收尾

Streamlit 已改为只通过 FastAPI 请求排序与 A/B 接口，不再在前端进程直接
实例化模型。Docker Compose 为 UI 注入 `http://caged-api:8000` 并等待 API
健康检查。cached 和 GPU real 后端仍由 API 环境变量切换。

## 下一决策

停止扩展 Gate 模型。若未来获得真实曝光/点击日志或更强的请求前特征，再建立
新的独立协议；在此之前，项目应以 R8.9 离线 Tail-floor 结果和 R19/R20
可部署性审计并列展示，不能合并成一个线上结论。
