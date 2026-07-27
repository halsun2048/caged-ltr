# Yelp R1.8 adaptive fixed-budget retrieval

Validation only. No training or test access. Every route has 500 candidates.

## Seed 42 uncertainty-threshold selection

| Target injection | Actual | Threshold | Overall | Head | Tail | Cold | Feasible |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0% | 0.00% | 0.999963 | 0.516158 | 0.669157 | 0.011464 | 0.000000 | yes |
| 10% | 10.00% | 0.984495 | 0.516349 | 0.668534 | 0.014109 | 0.000000 | yes |
| 25% | 25.00% | 0.958816 | 0.516603 | 0.667467 | 0.016755 | 0.000000 | yes |
| 50% | 50.04% | 0.910032 | 0.516412 | 0.664888 | 0.023810 | 0.000000 | yes |
| 75% | 75.00% | 0.839046 | 0.516603 | 0.663198 | 0.029982 | 0.005076 | no |
| 100% | 100.00% | 0.359505 | 0.516730 | 0.660708 | 0.043210 | 0.030457 | no |

Selected target injection rate: `50%`; locked uncertainty threshold: `0.910032`.

## Three-seed validation

| Method | Overall | Head | Torso | Tail | Cold-start |
|---|---:|---:|---:|---:|---:|
| collaborative | 0.514546 ± 0.001561 | 0.666815 ± 0.002233 | 0.181346 ± 0.006619 | 0.018225 ± 0.006003 | 0.000000 ± 0.000000 |
| always_450_50 | 0.515373 ± 0.001475 | 0.657536 ± 0.004118 | 0.205299 ± 0.006011 | 0.049971 ± 0.006256 | 0.030457 ± 0.000000 |
| adaptive_real | 0.514631 ± 0.001544 | 0.661953 ± 0.003291 | 0.193641 ± 0.006570 | 0.031746 ± 0.006999 | 0.010152 ± 0.010152 |
| adaptive_shuffled | 0.504983 ± 0.001564 | 0.655194 ± 0.003642 | 0.174669 ± 0.006910 | 0.018812 ± 0.004857 | 0.003384 ± 0.002931 |

## Acceptance

- validation_only_no_test_access: pass
- exactly_500_unique_candidates_all_routes: pass
- overall_within_0p01_all_seeds: pass
- head_within_0p01_all_seeds: pass
- tail_mean_absolute_gain_at_least_0p03: fail
- cold_mean_absolute_gain_at_least_0p03: fail
- tail_direction_positive_all_seeds: pass
- tail_real_beats_shuffled_all_seeds: pass
- adaptive_injection_nonzero_all_seeds: pass
- adaptive_injection_below_100_percent_all_seeds: pass
