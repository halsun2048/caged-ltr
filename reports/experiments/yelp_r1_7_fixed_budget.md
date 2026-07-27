# Yelp R1.7 fixed-budget dual-route retrieval

Validation only. Every route contains exactly 500 unique candidates.

## Seed 42 quota selection

| Collaborative / semantic | Overall | Head | Torso | Tail | Cold | Overlap | Feasible |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 500/0 | 0.516158 | 0.669157 | 0.183466 | 0.011464 | 0.000000 | 0.0 | yes |
| 450/50 | 0.516730 | 0.660708 | 0.203180 | 0.043210 | 0.030457 | 14.2 | yes |
| 400/100 | 0.506679 | 0.642654 | 0.209539 | 0.062610 | 0.045685 | 24.5 | no |
| 300/200 | 0.485305 | 0.605923 | 0.219078 | 0.100529 | 0.065990 | 37.6 | no |
| 250/250 | 0.471374 | 0.584667 | 0.220032 | 0.113757 | 0.076142 | 40.4 | no |

Selected quota: collaborative `450`, semantic `50`.

## Three-seed validation

| Method | Overall | Head | Torso | Tail | Cold-start |
|---|---:|---:|---:|---:|---:|
| collaborative | 0.514546 ± 0.001561 | 0.666815 ± 0.002233 | 0.181346 ± 0.006619 | 0.018225 ± 0.006003 | 0.000000 ± 0.000000 |
| semantic_only | 0.250700 ± 0.000000 | 0.269744 ± 0.000000 | 0.224801 ± 0.000000 | 0.158730 ± 0.000000 | 0.106599 ± 0.000000 |
| fixed_real | 0.515373 ± 0.001475 | 0.657536 ± 0.004118 | 0.205299 ± 0.006011 | 0.049971 ± 0.006256 | 0.030457 ± 0.000000 |
| fixed_shuffled | 0.496819 ± 0.000902 | 0.645826 ± 0.003399 | 0.167144 ± 0.007739 | 0.018225 ± 0.005738 | 0.010152 ± 0.000000 |

## Acceptance

- validation_only_no_test_access: pass
- exactly_500_unique_candidates_all_routes: pass
- overall_within_0p01_all_seeds: pass
- head_within_0p01_all_seeds: fail
- tail_mean_absolute_gain_at_least_0p03: pass
- cold_mean_absolute_gain_at_least_0p03: pass
- tail_direction_positive_all_seeds: pass
- tail_real_beats_shuffled_all_seeds: pass
