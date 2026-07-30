"""Offline ranking teacher integrations."""

from caged_ltr.teachers.prp import (
    DEFAULT_PRP_PROMPT,
    DeterministicMockTeacher,
    PairComparison,
    PairwiseTeacher,
    PRPCandidate,
    PRPQuery,
    PRPRanking,
    TeacherMetadata,
    TeacherResponse,
    allpair_borda,
    compare_bidirectional,
    pair_diagnostics,
    prompt_sha256,
    query_fingerprint,
    render_pair_prompt,
    sliding_k,
)

__all__ = [
    "DEFAULT_PRP_PROMPT",
    "DeterministicMockTeacher",
    "PRPCandidate",
    "PRPQuery",
    "PRPRanking",
    "PairComparison",
    "PairwiseTeacher",
    "TeacherMetadata",
    "TeacherResponse",
    "allpair_borda",
    "compare_bidirectional",
    "pair_diagnostics",
    "prompt_sha256",
    "query_fingerprint",
    "render_pair_prompt",
    "sliding_k",
]
