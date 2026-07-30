from __future__ import annotations

from dataclasses import replace

import pytest

from caged_ltr.teachers import (
    DeterministicMockTeacher,
    PairComparison,
    PRPCandidate,
    PRPQuery,
    allpair_borda,
    compare_bidirectional,
    pair_diagnostics,
    sliding_k,
)
from caged_ltr.teachers.prp_smoke import run_prp_smoke


def _query() -> PRPQuery:
    return PRPQuery(
        query_id="q1",
        text="which passage is most relevant",
        candidates=(
            PRPCandidate("a", "passage a", 3.0, 0.1, 3.0),
            PRPCandidate("b", "passage b", 2.0, 0.2, 2.0),
            PRPCandidate("c", "passage c", 1.0, 0.3, 1.0),
        ),
    )


def test_bidirectional_consensus_and_allpair_prompt_count() -> None:
    query = _query()
    teacher = DeterministicMockTeacher()
    comparison = compare_bidirectional(
        teacher,
        query,
        query.candidates[0],
        query.candidates[1],
    )

    assert comparison.consensus == "left"
    assert comparison.swap_agree
    progress: list[tuple[str, int, int]] = []
    ranking = allpair_borda(
        teacher,
        query,
        progress_callback=lambda *event: progress.append(event),
    )
    assert ranking.prompt_count == 6
    assert ranking.ranking == ("a", "b", "c")
    assert ranking.scores == {"a": 2.0, "b": 1.0, "c": 0.0}
    assert progress[-1] == ("allpair", 6, 6)


def test_sliding_k_moves_best_candidate_from_tail_to_top() -> None:
    query = _query().reordered(("c", "b", "a"))
    ranking = sliding_k(DeterministicMockTeacher(), query, passes=1)

    assert ranking.prompt_count == 4
    assert ranking.ranking[0] == "a"
    with pytest.raises(ValueError, match="passes must be positive"):
        sliding_k(DeterministicMockTeacher(), query, passes=0)


def test_pair_diagnostics_detect_strict_cycle() -> None:
    query = _query()

    def edge(left: str, right: str, consensus: str) -> PairComparison:
        return PairComparison(
            query_id=query.query_id,
            left_id=left,
            right_id=right,
            forward_choice="first",
            reverse_choice="second",
            consensus=consensus,
            swap_agree=True,
            input_tokens=2,
            output_tokens=2,
            latency_ms=0.0,
        )

    diagnostics = pair_diagnostics(
        query,
        (
            edge("a", "b", "left"),
            edge("b", "c", "left"),
            edge("a", "c", "right"),
        ),
    )

    assert diagnostics["swap_agreement"] == 1.0
    assert diagnostics["pair_accuracy"] == pytest.approx(2.0 / 3.0)
    assert diagnostics["complete_triples"] == 1
    assert diagnostics["cycle_rate"] == 1.0


def test_validation_rejects_invalid_candidates_and_permutations() -> None:
    query = _query()
    with pytest.raises(ValueError, match="exact permutation"):
        query.reordered(("a", "b", "missing"))
    with pytest.raises(ValueError, match="finite"):
        replace(query.candidates[0], teacher_score=float("nan"))


def test_resumable_prp_smoke_writes_each_query_once(tmp_path) -> None:
    output_dir = tmp_path / "smoke"
    first = run_prp_smoke(
        output_dir,
        query_count=5,
        candidates_per_query=5,
        sliding_passes=2,
    )
    second = run_prp_smoke(
        output_dir,
        query_count=5,
        candidates_per_query=5,
        sliding_passes=2,
    )

    assert first == second
    assert all(first["acceptance"].values())
    assert first["expected_allpair_prompt_count"] == 5 * 3 * 5 * 4
    assert (
        first["ranking"]["reverse/initial"]["NDCG@5"]
        < first["ranking"]["bm25/initial"]["NDCG@5"]
    )
    assert (
        first["ranking"]["reverse/allpair"]["NDCG@5"]
        == first["ranking"]["bm25/allpair"]["NDCG@5"]
    )
    assert len((output_dir / "query_results.jsonl").read_text().splitlines()) == 5
    with pytest.raises(ValueError, match="identity mismatch"):
        run_prp_smoke(
            output_dir,
            seed=2024,
            query_count=5,
            candidates_per_query=5,
            sliding_passes=2,
        )
