# Yelp R1.2 validation-only calibrated fusion

No test split was scored or used.

Selected on seed 42 validation: `zscore` with semantic weight `0.25`.

| Method | Overall NDCG@10 | Torso | Tail | Cold-start | Feasible |
|---|---:|---:|---:|---:|:---:|
| zscore weight=0.1 | 0.426780 | 0.184599 | 0.028098 | 0.000000 | yes |
| zscore weight=0.25 | 0.427630 | 0.199471 | 0.039950 | 0.000000 | yes |
| zscore weight=0.5 | 0.415327 | 0.214334 | 0.058648 | 0.000000 | no |
| zscore weight=1 | 0.379866 | 0.225312 | 0.086888 | 0.014516 | no |
| zscore weight=2 | 0.326563 | 0.221420 | 0.111038 | 0.029209 | no |
| zscore weight=4 | 0.282050 | 0.211803 | 0.127251 | 0.056660 | no |
| zscore weight=8 | 0.251591 | 0.204430 | 0.136868 | 0.077852 | no |
| rank weight=0.1 | 0.419641 | 0.187267 | 0.029587 | 0.000000 | yes |
| rank weight=0.25 | 0.408467 | 0.204307 | 0.041234 | 0.000000 | no |
| rank weight=0.5 | 0.388571 | 0.216598 | 0.055129 | 0.000000 | no |
| rank weight=1 | 0.361940 | 0.226968 | 0.070468 | 0.000000 | no |
| rank weight=2 | 0.326724 | 0.227779 | 0.086949 | 0.000000 | no |
| rank weight=4 | 0.294323 | 0.224884 | 0.106311 | 0.007726 | no |
| rank weight=8 | 0.264532 | 0.215874 | 0.120271 | 0.032285 | no |

## Three-seed validation replication

| System | Overall NDCG@10 | Head | Torso | Tail | Cold-start |
|---|---:|---:|---:|---:|---:|
| llm_init | 0.419509 ± 0.000404 | 0.535658 ± 0.002610 | 0.172495 ± 0.005740 | 0.025791 ± 0.005928 | 0.000000 ± 0.000000 |
| calibrated | 0.428054 ± 0.000412 | 0.537458 ± 0.002146 | 0.201562 ± 0.004915 | 0.045787 ± 0.007427 | 0.000000 ± 0.000000 |

## Acceptance

- seed42_candidate_fully_feasible: pass
- overall_within_tolerance_all_seeds: pass
- torso_non_decrease_all_seeds: pass
- tail_direction_positive_all_seeds: pass
- tail_mean_absolute_gain_at_least_0p005: pass
- cold_start_mean_ndcg_nonzero: fail
