# CAGED-LTR v1 locked result

CAGED-LTR v1 is locked to the frozen dev gate (`entropy <= 0.7`, permutation
stability `>= 0.2`) and the one-time strict official test evaluation.

- Test queries: 295
- BM25 NDCG@10: 0.5683
- FIRST NDCG@10: 0.6066
- Frozen gate NDCG@10: 0.6066
- Gate route: FIRST 286 / BM25 fallback 9

The gate does not yet show an incremental gain over FIRST because it routes
almost all queries to FIRST.  This is a locked v1 baseline, not evidence that
confidence routing is solved.
