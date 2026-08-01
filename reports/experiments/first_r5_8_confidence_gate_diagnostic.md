# FIRST confidence-gate diagnostic

Using the frozen 295-query official candidate pool, the diagnostic compares
per-query FIRST baseline NDCG@10 with BM25 and bins queries by normalized FIRST
entropy.  This is a development diagnostic only; no test-set threshold is
selected from it.

The largest mean gain occurs in the second entropy quartile (0.0611), while
the highest-entropy quartile averages 0.0265.  Thus entropy is not a monotone
proxy for gain: a naive “route every high-entropy query to PRP” rule is not
justified.  A future gate should use validation-only calibration and include
margin, permutation stability, and behavior features rather than entropy
alone.
