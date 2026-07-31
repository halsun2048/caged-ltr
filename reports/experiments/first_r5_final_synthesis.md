# FIRST R5 evidence synthesis

## Evidence currently established

- R5.1/R5.2 GPU execution is reproducible: the pinned FIRST checkpoint runs on
  CUDA, all frozen prompts parse, and the cache is resumable.
- On held-out TREC-DL top-20 candidates (97 queries), FIRST baseline reaches
  NDCG@10 `0.630504` versus BM25 `0.491249`.
- Against the frozen PRP control on the identical top-20 pool, FIRST is
  `+0.027725` overall (paired 95% CI `[0.01154, 0.04473]`).
- The gain is year-dependent: DL2019 `-0.00345` (CI includes zero), DL2020
  `+0.05255` (CI excludes zero).
- Identity/position debiasing is not yet useful: its difference from raw FIRST
  is statistically indistinguishable from zero.

## Claim boundary

The defensible claim is: **FIRST is a strong held-out reranker on this
TREC-DL top-20 candidate snapshot, with a significant advantage over the
frozen PRP control in aggregate, but the advantage is not stable across years.**
No cross-dataset or production-fusion claim is currently justified.

## Next executable gate

Acquire a third independent benchmark with official qrels, freeze its candidate
pool before opening qrels, run the existing 2–26 identifier GPU protocol, and
repeat the same paired FIRST/PRP bootstrap. Until that gate is available, more
single-checkpoint GPU runs on TREC-DL are not informative.
