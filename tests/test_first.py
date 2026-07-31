from __future__ import annotations

import pytest

from caged_ltr.teachers.first import (
    FirstCandidate,
    alphabetic_identifiers,
    audit_identifier_tokens,
    build_prompt_entries,
    pair_agreement,
    parse_generated_ranking,
    rank_identifiers_from_logits,
    render_first_user_prompt,
    sliding_window_ranges,
)


class _BracketTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        values = {"[": [10]}
        if text in values:
            return values[text]
        if len(text) == 2 and text[0] == "[" and text[1].isalpha():
            return [10, 100 + ord(text[1])]
        raise AssertionError(text)

    def decode(self, token_ids: list[int]) -> str:
        assert len(token_ids) == 1
        return chr(token_ids[0] - 100)


def _candidates(size: int = 4) -> tuple[FirstCandidate, ...]:
    return tuple(
        FirstCandidate(f"d{index}", f"passage [{chr(64 + index)}] {index}", index)
        for index in range(1, size + 1)
    )


def test_prompt_variants_preserve_candidate_identity() -> None:
    candidates = _candidates()
    baseline = build_prompt_entries(
        candidates,
        query_id="q1",
        variant="baseline",
    )
    reverse = build_prompt_entries(
        candidates,
        query_id="q1",
        variant="reverse",
    )
    random_a = build_prompt_entries(
        candidates,
        query_id="q1",
        variant="random_permutation",
        seed=42,
    )
    random_b = build_prompt_entries(
        candidates,
        query_id="q1",
        variant="random_permutation",
        seed=42,
    )
    remapped = build_prompt_entries(
        candidates,
        query_id="q1",
        variant="identifier_remap",
        seed=42,
    )

    assert [entry.identifier for entry in baseline] == list("ABCD")
    assert [entry.candidate.candidate_id for entry in reverse] == [
        "d4",
        "d3",
        "d2",
        "d1",
    ]
    assert random_a == random_b
    assert {entry.candidate.candidate_id for entry in random_a} == {
        "d1",
        "d2",
        "d3",
        "d4",
    }
    assert [entry.candidate.candidate_id for entry in remapped] == [
        "d1",
        "d2",
        "d3",
        "d4",
    ]
    assert {entry.identifier for entry in remapped} == set("ABCD")
    assert [entry.identifier for entry in remapped] != list("ABCD")

    prompt = render_first_user_prompt("example query", baseline)
    assert "[A] passage (A) 1" in prompt
    assert "Only respond with the ranking results" in prompt


def test_identifier_token_audit_uses_next_token_after_open_bracket() -> None:
    token_ids = audit_identifier_tokens(_BracketTokenizer(), tuple("ABCD"))
    assert token_ids == {letter: 100 + ord(letter) for letter in "ABCD"}


def test_generation_parser_logit_ranking_and_pair_agreement() -> None:
    expected = tuple("ABCD")
    generated = parse_generated_ranking("[B] > [A] > [D] > [C]", expected)
    from_logits = rank_identifiers_from_logits(
        {"A": 3.0, "B": 4.0, "C": 1.0, "D": 2.0},
        expected,
    )

    assert generated == from_logits
    assert pair_agreement(generated, from_logits) == 1.0
    assert pair_agreement(expected, tuple(reversed(expected))) == 0.0
    with pytest.raises(ValueError, match="exact identifier permutation"):
        parse_generated_ranking("[A] > [A] > [B]", expected)
    with pytest.raises(ValueError, match="cover every"):
        rank_identifiers_from_logits({"A": 1.0}, expected)


def test_window_plans_cover_edges_and_shift_internal_boundaries() -> None:
    canonical = sliding_window_ranges(100)
    shifted = sliding_window_ranges(100, boundary_offset=5)

    assert canonical == (
        (80, 100),
        (70, 90),
        (60, 80),
        (50, 70),
        (40, 60),
        (30, 50),
        (20, 40),
        (10, 30),
        (0, 20),
    )
    assert shifted[0] == (80, 100)
    assert shifted[-1] == (0, 15)
    assert shifted != canonical
    assert {index for start, end in shifted for index in range(start, end)} == set(
        range(100)
    )
    with pytest.raises(ValueError, match="between 2"):
        alphabetic_identifiers(1)
