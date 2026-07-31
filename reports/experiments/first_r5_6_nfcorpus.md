# FIRST R5.6 BEIR/NFCorpus external validation

NFCorpus was downloaded from the BEIR public release and frozen before qrels
were used for evaluation. A deterministic lexical BM25 implementation produced
20 candidates for each of 323 test queries; four FIRST perturbations were run
on 8 RTX 4090 GPUs (1,292 prompts total).

| Method | Linear graded NDCG@10 |
|---|---:|
| Frozen BM25 top-20 | 0.306935 |
| FIRST baseline | 0.343543 |
| FIRST debiased ensemble | 0.334447 |

Paired bootstrap 95% CIs:

- FIRST baseline − BM25: `[0.02493, 0.04899]`
- FIRST debiased − BM25: `[0.01366, 0.04223]`
- Debiased − baseline: `[-0.01523, -0.00333]`

FIRST baseline therefore improves over the independently prepared BM25 pool;
the identity/z-score debiasing post-process is significantly worse on this
dataset. This is the first external replication beyond TREC-DL, but the BM25
retrieval is a frozen in-project lexical implementation rather than Pyserini;
an official BEIR BM25 control and matched PRP run remain follow-up controls.
