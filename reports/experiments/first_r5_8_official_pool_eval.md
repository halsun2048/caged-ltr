# R5.8 strict official candidate-pool evaluation

This supplementary evaluation uses the official Pyserini NFCorpus BM25 top-20
candidate pool.  It contains 295 queries with at least two retrieved
candidates; 28 raw test queries had fewer than two official hits and are
excluded from this matched-pool comparison.  FIRST inference was frozen before
qrels were read.

Results (linear graded NDCG@10):

- BM25: 0.3380
- FIRST baseline: 0.3680 (absolute gain +0.0300)
- FIRST debiased: 0.3656 (absolute gain +0.0276)

Paired bootstrap 95% CI for FIRST baseline minus BM25 is [0.0191, 0.0412];
the interval excludes zero.  The debiased-minus-baseline interval is
[-0.0084, 0.0034], so the debiasing adjustment does not produce a reliable
change on this candidate pool.

This confirms the direction of the locked R5.7 result under a stricter,
official Pyserini candidate pool, but it is not a replacement for the locked
test: the query subset and candidate construction differ.  PRP remains an
optional teacher comparison and is not needed to establish the FIRST-vs-BM25
effect.
