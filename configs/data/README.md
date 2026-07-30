# 数据配置

每个数据集配置应记录来源、版本、下载日期、许可、原始文件校验和、过滤规则、
时间切分点和处理后数据指纹。不要把本地绝对路径或访问凭据写入配置。

Yelp 包含两个不可混用的配置：`yelp_llmesr_author.yaml` 用于论文一致复现，
`yelp_current.yaml` 用于当前官方快照扩展实验。详细说明见 `docs/data/yelp.md`。

Fashion 与 Beauty 使用 LLM-ESR 作者包中已经编号的交互序列。Beauty 中不足三次
交互的用户依作者 loader 保留为 train-only，不应强行构造验证或测试目标。

RLMRec Yelp 使用作者公开的稀疏矩阵和 profile embedding。profile 生成截止时间
不可验证，配置必须保留 `temporally_unverified` 标记；不得与严格 train-only
语义资产混用。

PRP 的 TREC-DL19/20 配置固定官方查询与 NIST qrels 的 URL、MD5 和记录数，
并固定 Pyserini 版本、预构建索引、BM25 参数和生成 run 的 SHA-256。微软
Top-1000 文本 TSV 的物理行序不得冒充 BM25 rank；教师输入必须与 qrels 字段
物理分离。
