# R7.8 frozen English MIND calibration evaluation

The trained English MiniLM was evaluated once on the 4,686-query calibration
split after model selection had finished.

| Metric | Pretrained | Trained | Absolute change |
|---|---:|---:|---:|
| NDCG@10 | 0.303340 | 0.387407 | +0.084068 |
| Hit@10 | 0.634870 | 0.746052 | +0.111182 |
| MRR | 0.278043 | 0.353296 | +0.075253 |

The paired NDCG@10 bootstrap interval is `[+0.075217, +0.092811]` with
probability-positive `1.0`. Frequency-bucket absolute NDCG@10 changes are
Head `+0.084656`, Torso `+0.065834`, and Tail `+0.103393`; every bucket is
positive. Tail contains only 87 queries, so its estimate is less precise
despite a positive interval.

The calibration split was not used for tuning. The 2,305,960-query MIND
holdout and the locked NFCorpus test remain unaccessed.
