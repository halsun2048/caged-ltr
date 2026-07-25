# 数据配置

每个数据集配置应记录来源、版本、下载日期、许可、原始文件校验和、过滤规则、
时间切分点和处理后数据指纹。不要把本地绝对路径或访问凭据写入配置。

Yelp 包含两个不可混用的配置：`yelp_llmesr_author.yaml` 用于论文一致复现，
`yelp_current.yaml` 用于当前官方快照扩展实验。详细说明见 `docs/data/yelp.md`。

Fashion 与 Beauty 使用 LLM-ESR 作者包中已经编号的交互序列。Beauty 中不足三次
交互的用户依作者 loader 保留为 train-only，不应强行构造验证或测试目标。
