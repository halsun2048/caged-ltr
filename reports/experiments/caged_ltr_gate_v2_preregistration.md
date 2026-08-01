# CAGED-LTR gate v2 preregistration

Gate v2 adds query length, passage text length/short-passage rate, candidate
frequency proxies, and a reserved behavior-uncertainty field.  The public
NFCorpus data do not contain exposure/click behavior, so behavior uncertainty
is explicitly marked unavailable and is not imputed from qrels.

The threshold grid and one-time test-evaluation rule are frozen in
`configs/experiments/caged_ltr_gate_v2_preregistered.json`.

There is currently no genuinely independent public dev set with the required
qrels in this repository: the existing dev split has already supported prior
gate development.  Therefore v2 is preregistered but not test-evaluated yet.
A second final evaluation requires a newly acquired or newly released
independent validation set; otherwise it would be another reuse of the same
dev evidence.
