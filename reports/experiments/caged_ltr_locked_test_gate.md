# CAGED-LTR frozen gate: untouched test

The gate thresholds were frozen from dev and applied once to the strict
official Pyserini candidate pool.  No test-set tuning was performed.

- Frozen thresholds: entropy `<= 0.7`, permutation stability `>= 0.2`
- Queries: 295
- Routes: FIRST 286, BM25 fallback 9
- BM25 NDCG@10: `0.5683`
- FIRST NDCG@10: `0.6066`
- Frozen gate NDCG@10: `0.6066`
- Gate minus BM25: `+0.0383`

The gate is essentially FIRST-dominant on this pool, so its improvement over
FIRST itself is negligible (`+0.00004`).  The result supports the FIRST gain
but does not yet demonstrate a meaningful routing benefit.  The next research
step should therefore target a richer gate signal or a larger/independent
validation design, not claim routing superiority from this run.
