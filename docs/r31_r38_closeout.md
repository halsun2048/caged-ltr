# R31–R38 应用平台扩展收尾

## 已实现

- R31：MCP JSON-RPC bridge，支持 stdio/HTTP，并通过 FastAPI 调用搜索、解释、A/B
  和运行指标。
- R32：SQLite 事件与反馈存储，schema 可迁移到 PostgreSQL；新增 `/feedback`、
  `/events/summary`。
- R33：MemoryState缓存/计数器，RedisState 可选适配器；无Redis时本地运行。
- R34：HybridRetriever 的 lexical fallback 与 QdrantProvider 接口；Qdrant需要
  显式安装客户端并实现embedding provider。
- R35：TaskRunner本地后台任务抽象；现有OpenAI-compatible provider校验和降级路径
  继续沿用。
- R36：Streamlit增加MCP与反馈页面、固定工具列表、事件摘要和反馈提交。
- R37：保留cached演示默认值，增加完整服务编排模板和可选依赖说明。
- R38：MCP initialize/tools/list/tools/call、搜索、召回、反馈和事件统计端到端通过。

## 尚未宣称完成的部分

- 本机没有PostgreSQL、Redis、Qdrant常驻服务，因此没有声称生产级分布式验证。
- Qdrant适配器需要真实embedding模型和collection迁移后才可启用。
- TaskRunner是本地线程池，不是Celery/RQ分布式队列。
- MCP bridge当前没有认证、TLS和租户隔离，适合本机或受控网络。
- Docker完整编排受当前机器Docker socket权限限制，Compose模板已提供但未构建镜像。
