# CAGED-LTR final evidence synthesis

## Locked evidence

1. **FIRST vs BM25 on the strict official test pool**
   - BM25 NDCG@10: `0.5683`
   - FIRST NDCG@10: `0.6066`
   - Frozen v1 gate NDCG@10: `0.6066`
   - Gate routes: FIRST 286/295, BM25 fallback 9/295

2. **Independent dev2 confirmation**
   - 2,380 official NFCorpus train queries, separate from the prior dev work
   - Frozen threshold selection produced entropy `<= 0.7`, stability `>= 0.2`
   - Held-out dev2 NDCG@10: `0.6166`

3. **Tiny student**
   - A CPU MLP using FIRST/BM25 ranks, entropy and margin reached dev NDCG@10
     `0.5177`; it is a protocol student, not a production text encoder.

## Conclusions

- FIRST provides a reproducible ranking improvement over BM25 on the locked
  official candidate pool.
- The current confidence gate is threshold-stable but FIRST-dominant; it does
  not demonstrate an incremental routing gain over FIRST.
- Public NFCorpus has no exposure/click behavior, so behavior uncertainty and
  causal debiasing cannot be claimed.
- The project contribution is currently a reproducible confidence-aware
  teacher-to-student protocol, not a validated industrial search-ad system.

## Stopping rule

The original untouched test is not reopened.  Further work requires either a
new independent benchmark or real advertising logs with behavior fields.
The next publishable engineering step is to package the frozen teacher labels
and train a text-aware lightweight student on a newly acquired independent
dataset, followed by efficiency and calibration measurements.
