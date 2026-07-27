# Beauty R1.5 full-catalog paired bootstrap

Post-hoc uncertainty audit. No method or weight was selected from full-catalog results.

| Bucket | Comparison | Mean ΔNDCG@10 | 95% CI | p (two-sided) |
|---|---|---:|---:|---:|
| overall | gate_minus_llm_init | +0.003229 | [+0.002920, +0.003541] | 0.000400 |
| overall | gate_minus_fixed_fusion | +0.000164 | [+0.000109, +0.000222] | 0.000400 |
| overall | real_gate_minus_shuffled_gate | +0.004128 | [+0.003698, +0.004561] | 0.000400 |
| overall | gate_minus_semantic_only | -0.000165 | [-0.001158, +0.000844] | 0.777045 |
| head | gate_minus_llm_init | +0.004952 | [+0.004408, +0.005509] | 0.000400 |
| head | gate_minus_fixed_fusion | -0.000150 | [-0.000193, -0.000109] | 0.000400 |
| head | real_gate_minus_shuffled_gate | +0.006685 | [+0.005923, +0.007460] | 0.000400 |
| head | gate_minus_semantic_only | +0.015185 | [+0.013792, +0.016578] | 0.000400 |
| torso | gate_minus_llm_init | +0.001693 | [+0.001365, +0.002048] | 0.000400 |
| torso | gate_minus_fixed_fusion | +0.000699 | [+0.000542, +0.000868] | 0.000400 |
| torso | real_gate_minus_shuffled_gate | +0.001570 | [+0.001195, +0.001975] | 0.000400 |
| torso | gate_minus_semantic_only | -0.017111 | [-0.019026, -0.015255] | 0.000400 |
| tail | gate_minus_llm_init | +0.000986 | [+0.000595, +0.001450] | 0.000400 |
| tail | gate_minus_fixed_fusion | +0.000644 | [+0.000382, +0.000959] | 0.000400 |
| tail | real_gate_minus_shuffled_gate | +0.000866 | [+0.000474, +0.001323] | 0.000400 |
| tail | gate_minus_semantic_only | -0.019779 | [-0.022733, -0.016737] | 0.000400 |
| cold_start | gate_minus_llm_init | +0.000000 | [+0.000000, +0.000000] | 1.000000 |
| cold_start | gate_minus_fixed_fusion | +0.000000 | [+0.000000, +0.000000] | 1.000000 |
| cold_start | real_gate_minus_shuffled_gate | +0.000000 | [+0.000000, +0.000000] | 1.000000 |
| cold_start | gate_minus_semantic_only | -0.020727 | [-0.024044, -0.017502] | 0.000400 |

## Acceptance

- overall_gate_beats_fixed_ci_excludes_zero: pass
- tail_gate_beats_fixed_ci_excludes_zero: pass
- tail_real_gate_beats_shuffled_ci_excludes_zero: pass
- overall_gate_beats_semantic_only_ci_excludes_zero: fail
