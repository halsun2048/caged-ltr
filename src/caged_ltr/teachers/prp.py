"""Pairwise Ranking Prompting primitives and deterministic offline controls."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

PairChoice = Literal["first", "second", "tie"]
ConsensusChoice = Literal["left", "right", "tie"]
PairProgressCallback = Callable[[str, int, int], None]

DEFAULT_PRP_PROMPT = """Given a query and two passages, decide which passage is more relevant.
Return only A, B, or TIE.

Query: {query}

Passage A:
{first}

Passage B:
{second}

Answer:"""


@dataclass(frozen=True, slots=True)
class PRPCandidate:
    """One text candidate and its offline evaluation/control signals."""

    candidate_id: str
    text: str
    relevance: float
    initial_score: float
    teacher_score: float

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not self.text:
            raise ValueError("candidate text must not be empty")
        values = (self.relevance, self.initial_score, self.teacher_score)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candidate numeric fields must be finite")
        if self.relevance < 0:
            raise ValueError("relevance must be non-negative")


@dataclass(frozen=True, slots=True)
class PRPQuery:
    """A query with an explicitly ordered candidate list."""

    query_id: str
    text: str
    candidates: tuple[PRPCandidate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.query_id or not self.text:
            raise ValueError("query_id and query text must not be empty")
        if len(self.candidates) < 2:
            raise ValueError("PRP requires at least two candidates")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique within a query")

    def reordered(self, candidate_ids: Sequence[str]) -> PRPQuery:
        """Return the same query in a validated candidate order."""
        normalized = tuple(str(candidate_id) for candidate_id in candidate_ids)
        expected = {candidate.candidate_id for candidate in self.candidates}
        if len(normalized) != len(expected) or set(normalized) != expected:
            raise ValueError("reordered candidate IDs must be an exact permutation")
        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        return PRPQuery(
            query_id=self.query_id,
            text=self.text,
            candidates=tuple(by_id[candidate_id] for candidate_id in normalized),
        )


@dataclass(frozen=True, slots=True)
class TeacherMetadata:
    """Versioned metadata required for every persisted teacher label."""

    backend: str
    model_name: str
    model_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    quantization: str
    prompt_name: str
    prompt_version: str
    prompt_sha256: str
    generation_parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        required = (
            self.backend,
            self.model_name,
            self.model_revision,
            self.tokenizer_name,
            self.tokenizer_revision,
            self.quantization,
            self.prompt_name,
            self.prompt_version,
            self.prompt_sha256,
        )
        if not all(required):
            raise ValueError("teacher metadata string fields must not be empty")
        if len(self.prompt_sha256) != 64:
            raise ValueError("prompt_sha256 must be a hexadecimal SHA-256 digest")
        try:
            int(self.prompt_sha256, 16)
        except ValueError as error:
            raise ValueError("prompt_sha256 must be a hexadecimal SHA-256 digest") from error

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["generation_parameters"] = dict(self.generation_parameters)
        return payload


@dataclass(frozen=True, slots=True)
class TeacherResponse:
    """One ordered prompt response."""

    choice: PairChoice
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw_output: str
    score_first: float | None = None
    score_second: float | None = None
    input_truncated: bool = False

    def __post_init__(self) -> None:
        if self.choice not in {"first", "second", "tie"}:
            raise ValueError(f"unsupported pair choice: {self.choice}")
        if self.input_tokens < 0 or self.output_tokens < 0 or self.latency_ms < 0:
            raise ValueError("teacher cost fields must be non-negative")
        scores = (self.score_first, self.score_second)
        if any(score is not None and not math.isfinite(score) for score in scores):
            raise ValueError("teacher likelihood scores must be finite when provided")


class PairwiseTeacher(Protocol):
    """Backend contract shared by mock, local-model, and API teachers."""

    @property
    def metadata(self) -> TeacherMetadata: ...

    def compare(
        self,
        query: PRPQuery,
        first: PRPCandidate,
        second: PRPCandidate,
    ) -> TeacherResponse: ...


@dataclass(frozen=True, slots=True)
class PairComparison:
    """Two swapped prompts collapsed into a strict preference or tie."""

    query_id: str
    left_id: str
    right_id: str
    forward_choice: PairChoice
    reverse_choice: PairChoice
    consensus: ConsensusChoice
    swap_agree: bool
    input_tokens: int
    output_tokens: int
    latency_ms: float

    def winner_id(self) -> str | None:
        if self.consensus == "left":
            return self.left_id
        if self.consensus == "right":
            return self.right_id
        return None

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PRPRanking:
    """One aggregation result and the comparisons used to produce it."""

    method: str
    ranking: tuple[str, ...]
    scores: Mapping[str, float]
    comparisons: tuple[PairComparison, ...]
    prompt_count: int

    def payload(self, *, include_comparisons: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "method": self.method,
            "ranking": list(self.ranking),
            "scores": dict(self.scores),
            "prompt_count": self.prompt_count,
        }
        if include_comparisons:
            payload["comparisons"] = [
                comparison.payload() for comparison in self.comparisons
            ]
        return payload


def prompt_sha256(template: str = DEFAULT_PRP_PROMPT) -> str:
    """Hash the unrendered prompt template used as the protocol identity."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def render_pair_prompt(
    query: PRPQuery,
    first: PRPCandidate,
    second: PRPCandidate,
    *,
    template: str = DEFAULT_PRP_PROMPT,
) -> str:
    """Render an ordered PRP prompt."""
    return template.format(query=query.text, first=first.text, second=second.text)


class DeterministicMockTeacher:
    """A deterministic score comparator for pipeline tests, never a real PRP result."""

    def __init__(
        self,
        *,
        position_bias: float = 0.0,
        noise_scale: float = 0.0,
        tie_margin: float = 0.0,
        seed: int = 42,
        prompt_template: str = DEFAULT_PRP_PROMPT,
    ) -> None:
        if noise_scale < 0 or tie_margin < 0:
            raise ValueError("noise_scale and tie_margin must be non-negative")
        self.position_bias = float(position_bias)
        self.noise_scale = float(noise_scale)
        self.tie_margin = float(tie_margin)
        self.seed = int(seed)
        self.prompt_template = prompt_template
        self._metadata = TeacherMetadata(
            backend="deterministic_mock",
            model_name="synthetic-score-oracle",
            model_revision="r3.0",
            tokenizer_name="whitespace-estimator",
            tokenizer_revision="r3.0",
            quantization="not_applicable",
            prompt_name="prp_pair_ab",
            prompt_version="1",
            prompt_sha256=prompt_sha256(prompt_template),
            generation_parameters={
                "position_bias": self.position_bias,
                "noise_scale": self.noise_scale,
                "tie_margin": self.tie_margin,
                "seed": self.seed,
            },
        )

    @property
    def metadata(self) -> TeacherMetadata:
        return self._metadata

    def _jitter(self, query_id: str, first_id: str, second_id: str) -> float:
        key = f"{self.seed}\0{query_id}\0{first_id}\0{second_id}".encode()
        integer = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        unit = integer / (2**64 - 1)
        return self.noise_scale * (2.0 * unit - 1.0)

    def compare(
        self,
        query: PRPQuery,
        first: PRPCandidate,
        second: PRPCandidate,
    ) -> TeacherResponse:
        prompt = render_pair_prompt(
            query,
            first,
            second,
            template=self.prompt_template,
        )
        difference = (
            first.teacher_score
            - second.teacher_score
            + self.position_bias
            + self._jitter(query.query_id, first.candidate_id, second.candidate_id)
        )
        if abs(difference) <= self.tie_margin:
            choice: PairChoice = "tie"
            raw_output = "TIE"
        elif difference > 0:
            choice = "first"
            raw_output = "A"
        else:
            choice = "second"
            raw_output = "B"
        return TeacherResponse(
            choice=choice,
            input_tokens=len(prompt.split()),
            output_tokens=1,
            latency_ms=0.0,
            raw_output=raw_output,
        )


def _chosen_candidate_id(
    response: TeacherResponse,
    first: PRPCandidate,
    second: PRPCandidate,
) -> str | None:
    if response.choice == "first":
        return first.candidate_id
    if response.choice == "second":
        return second.candidate_id
    return None


def compare_bidirectional(
    teacher: PairwiseTeacher,
    query: PRPQuery,
    left: PRPCandidate,
    right: PRPCandidate,
    *,
    progress_callback: PairProgressCallback | None = None,
    stage: str = "pair",
) -> PairComparison:
    """Query both A/B orders and retain a strict preference only when they agree."""
    forward = teacher.compare(query, left, right)
    if progress_callback is not None:
        progress_callback(stage, 1, 2)
    reverse = teacher.compare(query, right, left)
    if progress_callback is not None:
        progress_callback(stage, 2, 2)

    forward_candidate = _chosen_candidate_id(forward, left, right)
    reverse_candidate = _chosen_candidate_id(reverse, right, left)
    swap_agree = forward_candidate == reverse_candidate
    if swap_agree and forward_candidate == left.candidate_id:
        consensus: ConsensusChoice = "left"
    elif swap_agree and forward_candidate == right.candidate_id:
        consensus = "right"
    else:
        consensus = "tie"
    return PairComparison(
        query_id=query.query_id,
        left_id=left.candidate_id,
        right_id=right.candidate_id,
        forward_choice=forward.choice,
        reverse_choice=reverse.choice,
        consensus=consensus,
        swap_agree=swap_agree,
        input_tokens=forward.input_tokens + reverse.input_tokens,
        output_tokens=forward.output_tokens + reverse.output_tokens,
        latency_ms=forward.latency_ms + reverse.latency_ms,
    )


def _ranking_from_scores(
    query: PRPQuery,
    scores: Mapping[str, float],
) -> tuple[str, ...]:
    candidates = {candidate.candidate_id: candidate for candidate in query.candidates}
    return tuple(
        sorted(
            scores,
            key=lambda candidate_id: (
                -float(scores[candidate_id]),
                -candidates[candidate_id].initial_score,
                candidate_id,
            ),
        )
    )


def allpair_borda(
    teacher: PairwiseTeacher,
    query: PRPQuery,
    *,
    progress_callback: PairProgressCallback | None = None,
) -> PRPRanking:
    """Compare every unordered pair twice and aggregate wins with half-credit ties."""
    scores = {candidate.candidate_id: 0.0 for candidate in query.candidates}
    comparisons: list[PairComparison] = []
    pairs = list(itertools.combinations(query.candidates, 2))
    prompt_total = len(query.candidates) * (len(query.candidates) - 1)
    completed_prompts = 0

    def pair_progress(stage: str, done: int, total: int) -> None:
        del stage, total
        nonlocal completed_prompts
        completed_prompts += done - (1 if done == 2 else 0)
        if progress_callback is not None:
            progress_callback("allpair", completed_prompts, prompt_total)

    for left, right in pairs:
        comparison = compare_bidirectional(
            teacher,
            query,
            left,
            right,
            progress_callback=pair_progress,
            stage="allpair",
        )
        comparisons.append(comparison)
        if comparison.consensus == "left":
            scores[left.candidate_id] += 1.0
        elif comparison.consensus == "right":
            scores[right.candidate_id] += 1.0
        else:
            scores[left.candidate_id] += 0.5
            scores[right.candidate_id] += 0.5
    return PRPRanking(
        method="allpair_borda",
        ranking=_ranking_from_scores(query, scores),
        scores=scores,
        comparisons=tuple(comparisons),
        prompt_count=prompt_total,
    )


def sliding_k(
    teacher: PairwiseTeacher,
    query: PRPQuery,
    *,
    passes: int,
    progress_callback: PairProgressCallback | None = None,
) -> PRPRanking:
    """Run backward adjacent comparison passes, prioritizing the top of the ranking."""
    if passes <= 0:
        raise ValueError("passes must be positive")
    ranking = list(query.candidates)
    comparisons: list[PairComparison] = []
    prompt_total = 2 * passes * (len(ranking) - 1)
    completed_prompts = 0

    def pair_progress(stage: str, done: int, total: int) -> None:
        del stage, total
        nonlocal completed_prompts
        completed_prompts += done - (1 if done == 2 else 0)
        if progress_callback is not None:
            progress_callback("sliding_k", completed_prompts, prompt_total)

    for _ in range(passes):
        for index in range(len(ranking) - 2, -1, -1):
            comparison = compare_bidirectional(
                teacher,
                query,
                ranking[index],
                ranking[index + 1],
                progress_callback=pair_progress,
                stage="sliding_k",
            )
            comparisons.append(comparison)
            if comparison.consensus == "right":
                ranking[index], ranking[index + 1] = ranking[index + 1], ranking[index]

    scores = {
        candidate.candidate_id: float(len(ranking) - index)
        for index, candidate in enumerate(ranking)
    }
    return PRPRanking(
        method=f"sliding_{passes}",
        ranking=tuple(candidate.candidate_id for candidate in ranking),
        scores=scores,
        comparisons=tuple(comparisons),
        prompt_count=prompt_total,
    )


def pair_diagnostics(
    query: PRPQuery,
    comparisons: Sequence[PairComparison],
) -> dict[str, float | int]:
    """Measure swap agreement, qrels accuracy, coverage, ties, and strict cycles."""
    if not comparisons:
        raise ValueError("pair diagnostics require at least one comparison")
    relevance = {
        candidate.candidate_id: candidate.relevance for candidate in query.candidates
    }
    swap_agreements = sum(comparison.swap_agree for comparison in comparisons)
    ties = sum(comparison.consensus == "tie" for comparison in comparisons)
    unequal = 0
    decisive_unequal = 0
    correct = 0
    tie_credit = 0.0
    directed_edges: set[tuple[str, str]] = set()
    for comparison in comparisons:
        left_relevance = relevance[comparison.left_id]
        right_relevance = relevance[comparison.right_id]
        winner = comparison.winner_id()
        if winner is not None:
            loser = (
                comparison.right_id
                if winner == comparison.left_id
                else comparison.left_id
            )
            directed_edges.add((winner, loser))
        if left_relevance == right_relevance:
            continue
        unequal += 1
        expected = (
            comparison.left_id
            if left_relevance > right_relevance
            else comparison.right_id
        )
        if winner is None:
            tie_credit += 0.5
            continue
        decisive_unequal += 1
        if winner == expected:
            correct += 1
            tie_credit += 1.0

    complete_triples = 0
    cycles = 0
    candidate_ids = [candidate.candidate_id for candidate in query.candidates]
    for triple in itertools.combinations(candidate_ids, 3):
        triple_edges = [
            edge
            for edge in directed_edges
            if edge[0] in triple and edge[1] in triple
        ]
        if len(triple_edges) != 3:
            continue
        complete_triples += 1
        wins = {candidate_id: 0 for candidate_id in triple}
        for winner, _ in triple_edges:
            wins[winner] += 1
        if sorted(wins.values()) == [1, 1, 1]:
            cycles += 1

    comparison_count = len(comparisons)
    return {
        "comparisons": comparison_count,
        "swap_agreement": swap_agreements / comparison_count,
        "tie_ratio": ties / comparison_count,
        "unequal_qrels_pairs": unequal,
        "decisive_qrels_pairs": decisive_unequal,
        "pair_coverage": decisive_unequal / unequal if unequal else 0.0,
        "pair_accuracy": correct / decisive_unequal if decisive_unequal else 0.0,
        "pair_accuracy_with_tie_credit": tie_credit / unequal if unequal else 0.0,
        "complete_triples": complete_triples,
        "cycles": cycles,
        "cycle_rate": cycles / complete_triples if complete_triples else 0.0,
    }


def query_fingerprint(queries: Sequence[PRPQuery]) -> str:
    """Hash all query/candidate inputs in a stable JSON representation."""
    payload = [
        {
            "query_id": query.query_id,
            "text": query.text,
            "candidates": [asdict(candidate) for candidate in query.candidates],
        }
        for query in queries
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
