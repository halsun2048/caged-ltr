# Beauty R1.5 full-catalog robustness audit

Post-hoc protocol audit with the gate and all weights locked before full-catalog access.

Evaluated users per seed: `50498`; catalog items: `57289`.

| Method | H@10 mean ± std | NDCG@10 mean ± std |
|---|---:|---:|
| llm_init | 0.024192 ± 0.001195 | 0.012107 ± 0.000714 |
| semantic_only | 0.028714 ± 0.000000 | 0.015501 ± 0.000000 |
| fixed_fusion | 0.029731 ± 0.000541 | 0.015172 ± 0.000340 |
| confidence_gate | 0.030028 ± 0.000573 | 0.015336 ± 0.000351 |
| shuffled_gate | 0.022483 ± 0.000899 | 0.011208 ± 0.000512 |

## Item-frequency NDCG@10

| Method | Head | Torso | Tail | Cold-start |
|---|---:|---:|---:|---:|
| llm_init | 0.021884 ± 0.001159 | 0.000737 ± 0.000329 | 0.000089 ± 0.000087 | 0.000000 ± 0.000000 |
| semantic_only | 0.011651 ± 0.000000 | 0.019541 ± 0.000000 | 0.020853 ± 0.000000 | 0.020727 ± 0.000000 |
| fixed_fusion | 0.026986 ± 0.000458 | 0.001732 ± 0.000391 | 0.000431 ± 0.000271 | 0.000000 ± 0.000000 |
| confidence_gate | 0.026836 ± 0.000405 | 0.002430 ± 0.000464 | 0.001075 ± 0.000288 | 0.000000 ± 0.000000 |
| shuffled_gate | 0.020151 ± 0.000747 | 0.000860 ± 0.000399 | 0.000209 ± 0.000136 | 0.000000 ± 0.000000 |

## Acceptance

- overall_gate_beats_fixed_all_seeds: pass
- tail_gate_beats_fixed_all_seeds: pass
- tail_real_gate_beats_shuffled_all_seeds: pass
- tail_gate_mean_absolute_gain_at_least_0p005: fail
