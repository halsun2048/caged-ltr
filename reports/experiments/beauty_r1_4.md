# Beauty R1.4 locked confidence-aware fusion

All fusion settings were locked on Yelp validation before Beauty test access.

| Method | H@10 mean ± std | NDCG@10 mean ± std |
|---|---:|---:|
| sasrec | 0.191202 ± 0.003841 | 0.121663 ± 0.003402 |
| llm_init | 0.241217 ± 0.003143 | 0.143487 ± 0.001889 |
| shuffled_init | 0.188318 ± 0.002282 | 0.120228 ± 0.001346 |
| semantic_only | 0.191017 ± 0.000000 | 0.117710 ± 0.000000 |
| fixed_fusion | 0.261964 ± 0.002927 | 0.159443 ± 0.001792 |
| confidence_gate | 0.263779 ± 0.002773 | 0.160692 ± 0.001793 |
| shuffled_gate | 0.232412 ± 0.002578 | 0.136830 ± 0.001593 |

## Item-frequency NDCG@10

| Method | Head | Torso | Tail | Cold-start |
|---|---:|---:|---:|---:|
| sasrec | 0.198387 ± 0.003769 | 0.045184 ± 0.004304 | 0.021151 ± 0.002891 | 0.000000 ± 0.000000 |
| llm_init | 0.245034 ± 0.001581 | 0.036196 ± 0.003593 | 0.009096 ± 0.001657 | 0.000000 ± 0.000000 |
| shuffled_init | 0.195661 ± 0.001212 | 0.044765 ± 0.002307 | 0.022581 ± 0.001619 | 0.000000 ± 0.000000 |
| semantic_only | 0.108536 ± 0.000000 | 0.127723 ± 0.000000 | 0.130345 ± 0.000000 | 0.129273 ± 0.000000 |
| fixed_fusion | 0.262948 ± 0.001064 | 0.056388 ± 0.003795 | 0.019138 ± 0.002404 | 0.000319 ± 0.000083 |
| confidence_gate | 0.260419 ± 0.000803 | 0.063586 ± 0.003912 | 0.026186 ± 0.003468 | 0.000838 ± 0.000204 |
| shuffled_gate | 0.233126 ± 0.001531 | 0.035618 ± 0.002955 | 0.008826 ± 0.001468 | 0.000000 ± 0.000000 |

## Acceptance

- gate_beats_strongest_collaborative_all_seeds: pass
- gate_tail_beats_fixed_fusion_all_seeds: pass
- real_gate_tail_beats_shuffled_gate_all_seeds: pass
- semantic_only_weaker_than_strongest_collaborative_all_seeds: pass
