# Yelp R1.2a semantic identity controls

Status: `stopped_after_identity_controls`. Seed 42 validation only; test was not accessed.

| Variant | Overall | Head | Torso | Tail | Cold-start |
|---|---:|---:|---:|---:|---:|
| real_raw | 0.426584 | 0.542450 | 0.181464 | 0.031652 | 0.000000 |
| shuffled_raw | 0.419538 | 0.531647 | 0.184686 | 0.032154 | 0.000000 |
| matched_random_raw | 0.421399 | 0.536151 | 0.177528 | 0.033146 | 0.000000 |

Identity gate: fail; single branches were skipped.

## Acceptance

- test_not_accessed: pass
- identity_gate_passed: fail
- real_beats_each_identity_control_overall: pass
- real_beats_each_identity_control_tail: fail
- dual_beats_each_single_branch_overall: None
- dual_beats_each_single_branch_tail: None
