# FIRST R5.4 year-shift analysis

The DL19/DL20 split has observable covariate differences:

| Year | Query words | Judged rate | Grade-2+ qrels/query | BM25 NDCG@10 | FIRST−PRP |
|---|---:|---:|---:|---:|---:|
| 2019 | 5.40 | 0.525 | 58.16 | 0.5058 | -0.00345 |
| 2020 | 6.04 | 0.529 | 30.85 | 0.4796 | +0.05255 |

Across queries, the FIRST−PRP difference correlates weakly with query length
(`r=0.175`), judged rate (`r=-0.077`), relevant-count (`r=-0.095`), and BM25
NDCG (`r=0.018`). The large year split is therefore not explained by one
simple covariate; it is consistent with a broader temporal/query-distribution
shift. A third independent benchmark is required before generalizing the
FIRST advantage.
