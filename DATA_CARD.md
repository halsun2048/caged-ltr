# Data Card

The repository contains code, manifests, aggregate reports, and small demo
records. Raw datasets, large checkpoints, private logs, and credentials are not
committed.

## Main public evidence

- Yelp/Fashion/Beauty author-derived recommendation datasets for replication.
- TREC-DL/MS MARCO for PRP and instruction-distillation experiments.
- NFCorpus for independent public ranking validation.
- MIND-derived English ranking packages for Student/Gate experiments.

## Splits and leakage controls

Experiments store query IDs, hashes, seeds, and split guards. Validation selects
models and thresholds; locked test artifacts are accessed only under their
recorded protocols. Some later MIND confirm data had already been used by prior
stages and must not be described as untouched.

## Prohibited claims

Public ranking data does not expose production ad auctions, impressions,
clicks, dwell time, or conversions. It cannot establish causal CTR/CVR lift,
position-bias correction, or production fairness.
