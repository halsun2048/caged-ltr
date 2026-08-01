# CAGED-LTR dev stability gate

Permutation stability was added to the gate using only the dev split.  For
each query, stability is the mean fraction of candidates whose FIRST position
stays within two ranks under reverse and random-permutation prompts.

- Mean stability: `0.4775`
- Selected dev-only thresholds: entropy `<= 0.7`, stability `>= 0.2`
- Threshold-selection NDCG@10: `0.5475`
- Frozen dev validation NDCG@10: `0.5167`
- Untouched test accessed: `false`

Adding stability gives a small improvement over the entropy-only validation
gate (`0.5105` → `0.5167`), but the gain is modest.  The gate remains a
development result; no test threshold has been selected.
