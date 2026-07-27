# Yelp R1.6 dual-route candidate recall

Validation only. No Yelp, Fashion, or Beauty test split was accessed.

Each route is the set union of collaborative Top-K and semantic Top-K, so its candidate budget lies between K and 2K.

## Recall@100

| Route | Overall | Head | Torso | Tail | Cold-start |
|---|---:|---:|---:|---:|---:|
| collaborative | 0.231064 ± 0.002217 | 0.312997 ± 0.002033 | 0.035506 ± 0.004026 | 0.001176 ± 0.002037 | 0.000000 ± 0.000000 |
| semantic_real | 0.088868 ± 0.000000 | 0.092494 ± 0.000000 | 0.089984 ± 0.000000 | 0.057319 ± 0.000000 | 0.045685 ± 0.000000 |
| semantic_shuffled | 0.008397 ± 0.000000 | 0.008004 ± 0.000000 | 0.008267 ± 0.000000 | 0.011464 ± 0.000000 | 0.015228 ± 0.000000 |
| union_real | 0.281107 ± 0.001575 | 0.353848 ± 0.001748 | 0.116163 ± 0.001863 | 0.058201 ± 0.001527 | 0.045685 ± 0.000000 |
| union_shuffled | 0.237426 ± 0.002215 | 0.318185 ± 0.002131 | 0.043667 ± 0.003846 | 0.012640 ± 0.002037 | 0.015228 ± 0.000000 |

Mean union candidate count: real `191.5`, shuffled `199.1`.

## Recall@500

| Route | Overall | Head | Torso | Tail | Cold-start |
|---|---:|---:|---:|---:|---:|
| collaborative | 0.514546 ± 0.001561 | 0.666815 ± 0.002233 | 0.181346 ± 0.006619 | 0.018225 ± 0.006003 | 0.000000 ± 0.000000 |
| semantic_real | 0.250700 ± 0.000000 | 0.269744 ± 0.000000 | 0.224801 ± 0.000000 | 0.158730 ± 0.000000 | 0.106599 ± 0.000000 |
| semantic_shuffled | 0.045865 ± 0.000000 | 0.045980 ± 0.000000 | 0.043561 ± 0.000000 | 0.050265 ± 0.000000 | 0.050761 ± 0.000000 |
| union_real | 0.591836 ± 0.000808 | 0.716767 ± 0.001382 | 0.327504 ± 0.006359 | 0.170488 ± 0.004438 | 0.106599 ± 0.000000 |
| union_shuffled | 0.537129 ± 0.001430 | 0.682705 ± 0.001853 | 0.216534 ± 0.006570 | 0.067313 ± 0.005014 | 0.050761 ± 0.000000 |

Mean union candidate count: real `900.6`, shuffled `978.0`.

## Acceptance

- validation_only_no_test_access: pass
- tail_union_absolute_gain_at_500_at_least_0p01: pass
- tail_real_union_beats_shuffled_all_seeds_at_500: pass
- tail_union_direction_positive_all_seeds_at_500: pass
- head_union_not_below_collaborative_at_500: pass
