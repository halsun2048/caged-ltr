# Beauty R1.4 paired bootstrap audit

Post-hoc uncertainty audit of the already locked sampled-1000 test. No method or weight was selected from these intervals.

| Bucket | Comparison | Mean ΔNDCG@10 | 95% CI | p (two-sided) |
|---|---|---:|---:|---:|
| overall | gate_minus_llm_init | +0.017206 | [+0.016428, +0.017972] | 0.000400 |
| overall | gate_minus_fixed_fusion | +0.001249 | [+0.001065, +0.001437] | 0.000400 |
| overall | real_gate_minus_shuffled_gate | +0.023862 | [+0.022884, +0.024859] | 0.000400 |
| head | gate_minus_llm_init | +0.015385 | [+0.014211, +0.016558] | 0.000400 |
| head | gate_minus_fixed_fusion | -0.002529 | [-0.002700, -0.002357] | 0.000400 |
| head | real_gate_minus_shuffled_gate | +0.027293 | [+0.025747, +0.028916] | 0.000400 |
| torso | gate_minus_llm_init | +0.027390 | [+0.025963, +0.028877] | 0.000400 |
| torso | gate_minus_fixed_fusion | +0.007198 | [+0.006708, +0.007691] | 0.000400 |
| torso | real_gate_minus_shuffled_gate | +0.027967 | [+0.026314, +0.029672] | 0.000400 |
| tail | gate_minus_llm_init | +0.017091 | [+0.015343, +0.018961] | 0.000400 |
| tail | gate_minus_fixed_fusion | +0.007049 | [+0.006267, +0.007893] | 0.000400 |
| tail | real_gate_minus_shuffled_gate | +0.017361 | [+0.015331, +0.019502] | 0.000400 |
| cold_start | gate_minus_llm_init | +0.000838 | [+0.000484, +0.001255] | 0.000400 |
| cold_start | gate_minus_fixed_fusion | +0.000519 | [+0.000302, +0.000781] | 0.000400 |
| cold_start | real_gate_minus_shuffled_gate | +0.000838 | [+0.000484, +0.001255] | 0.000400 |

## Acceptance

- overall_gate_beats_llm_ci_excludes_zero: pass
- overall_gate_beats_fixed_ci_excludes_zero: pass
- tail_gate_beats_fixed_ci_excludes_zero: pass
- tail_real_gate_beats_shuffled_ci_excludes_zero: pass
