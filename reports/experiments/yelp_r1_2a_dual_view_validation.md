# Yelp R1.2a dual-view structural controls

Seed 42 validation only; the test split was not evaluated.

| Variant | Overall | Head | Torso | Tail | Cold-start | Trainable params |
|---|---:|---:|---:|---:|---:|---:|
| dual_view_no_ca | 0.426584 | 0.542450 | 0.181464 | 0.031652 | 0.000000 | 2021632 |
| dual_view | 0.403566 | 0.524208 | 0.135019 | 0.022243 | 0.000000 | 2054528 |
| dual_view_unshared | 0.397480 | 0.517180 | 0.128824 | 0.024748 | 0.000000 | 2105088 |
| dual_view_capacity | 0.400849 | 0.528510 | 0.110884 | 0.008082 | 0.004463 | 2054528 |

Cross attention uses causal plus padding masks. This is a leakage-safe correction to the author implementation's padding-only cross-attention mask.
