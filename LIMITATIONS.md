# Limitations

- R8.9 is a strong offline Tail-floor result; R19/R20 are stricter deployability
  audits. Their metrics cannot be merged into one online claim.
- R20 found no Gate within 0.01 NDCG@10 of FIRST at no more than 45% FIRST calls.
- Tail-floor protects Tail by spending budget that otherwise serves Torso.
- Offline A/B replay observes both potential policies and is not live traffic.
- The portable FIRST path is replay/provider-based; real quota, pricing, and
  network failures require a production integration.
- Cross-domain evidence supports the routing principle but not zero-shot
  transfer of thresholds.
- Existing public data has no real advertising behavior or causal treatment.
