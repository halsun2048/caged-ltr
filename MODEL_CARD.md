# Model Card

## Components

- **Student:** MiniLM bi-encoder initialized from a public text encoder and
  distilled on frozen ranking data.
- **Teacher:** FIRST-compatible ranking adapter; the portable demo uses cached
  or replayed outputs unless a real provider is explicitly configured.
- **Gate:** post-Student routing over versioned request-time features. R19 uses
  a portable Logistic model; R20 also audits ExtraTrees gain and Tail-floor
  policies offline.

## Intended use

Research, reproducibility, ranking-system demonstrations, and internship
portfolio review. It is not approved for medical, legal, financial, or
production advertising decisions.

## Known behavior

The Student is strongest on Head queries. FIRST contributes most on Tail and
uncertain requests. Current deployable features cannot predict FIRST gain well
enough to stay within 0.01 NDCG@10 of FIRST at a 40% call budget.

## Runtime modes

- `cached`: deterministic local demonstration without GPU.
- `real`: MiniLM checkpoint plus configured FIRST replay/provider adapter.

Runtime responses expose the backend and route mode. Cached results must never
be reported as real model latency or online quality.
