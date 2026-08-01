# CAGED-LTR dev-only gate and tiny student

Threshold selection used only the existing NFCorpus dev split.  The untouched
test set was not read.  Queries were deterministically split into 259 training
and 65 validation queries; entropy thresholds were selected on the 259-query
training portion.

- Selected entropy threshold: `0.6`
- Gate NDCG@10 on threshold-selection portion: `0.5478`
- Frozen gate NDCG@10 on dev validation portion: `0.5105`
- Tiny CPU student validation NDCG@10: `0.5177`
- Student features: FIRST logit/rank, BM25 rank, entropy, and top-1/top-2 margin

This is a pipeline/protocol result, not a locked test claim.  The student is a
small MLP trained only on dev data and is not yet a production ranking model.
The validation gap indicates that the entropy-only gate is not sufficient;
future routing should add permutation stability and behavior features before
any untouched-test evaluation.
