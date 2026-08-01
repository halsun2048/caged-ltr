# CAGED-LTR v2 independent dev selection

The official NFCorpus train split was used as a new independent gate-selection
corpus (2,380 queries, Pyserini top-20, 4 prompt variants).  No untouched test
files were read in this run.

- Train portion: 1,904 queries
- Validation portion: 476 queries
- Mean permutation stability: `0.5576`
- Selected thresholds: entropy `<= 0.7`, stability `>= 0.2`
- Threshold-selection NDCG@10: `0.6147`
- Held-out independent-dev NDCG@10: `0.6166`

The selected thresholds agree with CAGED-LTR v1.  This supports threshold
stability, but the original untouched test was already evaluated for v1; it
must not be reopened for a second claim.  The v1 test result therefore remains
the sole locked test result, while v2 is an independent-dev confirmation.
