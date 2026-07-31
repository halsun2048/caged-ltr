# R5.7 preregistered query-length fallback

Before any test-set access, the dev-only policy space was fixed to:

`query_words < threshold -> BM25; otherwise -> FIRST`, with thresholds
`1, 2, 3, 4`. The selection metric was mean linear graded NDCG@10 on dev;
ties would choose the smaller threshold. No test files were read.

| Policy | Dev NDCG@10 | Fallback queries |
|---|---:|---:|
| Always FIRST | 0.305608 | 0 |
| Threshold 1 | 0.305608 | 0 |
| Threshold 2 | 0.304319 | 105 |
| Threshold 3 | 0.298993 | 169 |
| Threshold 4 | 0.296313 | 196 |

The preregistered winner is **always FIRST** (equivalently threshold 1). The
length diagnostic is real, but replacing short-query FIRST rankings with BM25
reduces overall dev NDCG. Therefore no fallback will be applied to the
untouched test set; the test evaluation remains the original locked FIRST
protocol.
