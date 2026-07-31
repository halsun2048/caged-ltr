# FIRST R5.5 external benchmark gate

The repository currently contains official qrels only for TREC-DL 2019 and
2020 (`data/processed/prp_trec_dl_top10` and `top100`). No third independent
benchmark with frozen candidates, query identities, and official qrels is
available locally.

Therefore no third-dataset GPU inference is launched: doing so without qrels
would produce another protocol/admission result, not an effectiveness result.
The current evidence must remain bounded to TREC-DL 2019/2020, with the
explicit finding that the FIRST-vs-PRP gain is significant on DL20 but not on
DL19.

Required inputs for the next executable stage:

1. an independently sourced benchmark and qrels;
2. a frozen candidate snapshot and leakage audit;
3. the existing `prepare_first_r5_3_trec.py`/GPU runner adapted to that
   candidate count (2–26 identifiers);
4. paired FIRST/PRP evaluation with the same bootstrap protocol.
