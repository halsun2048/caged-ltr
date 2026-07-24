# Fashion R1.3 locked external confirmation

No Fashion validation or test metric was used to select the fusion rule. The rule was locked on Yelp: per-query z-score with semantic weight 0.25.

Final evaluation uses one target plus 1000 fixed unseen negatives.

| Method | H@10 mean ± std | NDCG@10 mean ± std |
|---|---:|---:|
| sasrec | 0.398834 ± 0.009218 | 0.366025 ± 0.010585 |
| llm_init | 0.432190 ± 0.000606 | 0.380346 ± 0.007587 |
| semantic_only | 0.452936 ± 0.000000 | 0.390092 ± 0.000000 |
| calibrated_fusion | 0.449124 ± 0.001754 | 0.395513 ± 0.002854 |

## Item-frequency NDCG@10

| Method | Head | Torso | Tail | Cold-start |
|---|---:|---:|---:|---:|
| sasrec | 0.619466 ± 0.017131 | 0.073236 ± 0.005890 | 0.094256 ± 0.008521 | 0.000000 ± 0.000000 |
| llm_init | 0.646323 ± 0.009762 | 0.089444 ± 0.009571 | 0.083366 ± 0.012247 | 0.000674 ± 0.000432 |
| semantic_only | 0.585272 ± 0.000000 | 0.154392 ± 0.000000 | 0.171287 ± 0.000000 | 0.120188 ± 0.000000 |
| calibrated_fusion | 0.659761 ± 0.004113 | 0.104379 ± 0.007153 | 0.110864 ± 0.001970 | 0.009959 ± 0.001359 |

## Acceptance

- overall_direction_positive_all_seeds: pass
- tail_direction_positive_all_seeds: pass
- tail_mean_absolute_gain_at_least_0p005: pass
- semantic_only_weaker_than_llm_init_all_seeds: fail

## Data audit

- Author bundle: 9094 users, 4722 items.
- Paper table: 9049 users, 4722 items.
- The 45-user discrepancy is retained and explicitly reported.
