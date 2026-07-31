"""Faithful, model-independent protocol utilities for the FIRST listwise teacher.

The GPU runner is deliberately kept out of this module.  R5.0 can therefore
freeze prompts, audit token identities, parse complete generations, and test
perturbations without importing vLLM or loading the 7B checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

FIRST_MODEL = "rryisthebest/First_Model"
FIRST_MODEL_REVISION = "64eba9b83c174439d2b6f5d333fbb822b38d73a7"
FIRST_CODE_REPOSITORY = "https://github.com/gangiswag/llm-reranker"
FIRST_CODE_REVISION = "2d7cba423ad555064bdfc719313570b5f9525887"
FIRST_SYSTEM_MESSAGE = (
    "You are RankLLM, an intelligent assistant that can rank passages based on "
    "their relevancy to the query"
)
FIRST_PROMPT_VERSION = "first-author-alpha-v1"
FIRST_WINDOW_SIZE = 20
FIRST_WINDOW_STEP = 10
_IDENTIFIER_PATTERN = re.compile(r"\[([A-Z])\]")
_BRACKETED_ALPHA_PATTERN = re.compile(r"\[([A-z]+)\]")


@dataclass(frozen=True, slots=True)
class FirstCandidate:
    """One immutable candidate in a FIRST slate."""

    candidate_id: str
    text: str
    retrieval_rank: int

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.text.strip():
            raise ValueError("candidate ID and text must not be empty")
        if self.retrieval_rank <= 0:
            raise ValueError("retrieval_rank must be positive")


@dataclass(frozen=True, slots=True)
class FirstPromptEntry:
    """A candidate bound to its prompt position and alphabetic identifier."""

    candidate: FirstCandidate
    input_position: int
    identifier: str

    def __post_init__(self) -> None:
        if self.input_position <= 0:
            raise ValueError("input_position must be positive")
        if len(self.identifier) != 1 or not self.identifier.isascii():
            raise ValueError("identifier must be one ASCII character")
        if not self.identifier.isupper():
            raise ValueError("identifier must be uppercase")


def stable_sha256(payload: object) -> str:
    """Hash a JSON-serializable object using a canonical representation."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def alphabetic_identifiers(size: int) -> tuple[str, ...]:
    """Return A..Z identifiers, rejecting unsupported slate sizes."""
    if not 2 <= size <= 26:
        raise ValueError("FIRST slate size must be between 2 and 26")
    return tuple(chr(ord("A") + index) for index in range(size))


def _stable_permutation(
    values: Sequence[object],
    *,
    namespace: str,
    query_id: str,
    seed: int,
) -> list[object]:
    randomizer = random.Random(
        int(
            hashlib.sha256(
                f"{namespace}\0{query_id}\0{seed}".encode()
            ).hexdigest(),
            16,
        )
    )
    result = list(values)
    randomizer.shuffle(result)
    return result


def build_prompt_entries(
    candidates: Sequence[FirstCandidate],
    *,
    query_id: str,
    variant: str,
    seed: int = 42,
) -> tuple[FirstPromptEntry, ...]:
    """Bind candidates to positions/identifiers for one registered perturbation."""
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique within a slate")
    identifiers: Sequence[str] = alphabetic_identifiers(len(candidates))
    ordered: Sequence[FirstCandidate] = candidates
    if variant == "reverse":
        ordered = tuple(reversed(candidates))
    elif variant == "random_permutation":
        ordered = _stable_permutation(
            candidates,
            namespace="first-candidate-permutation-v1",
            query_id=query_id,
            seed=seed,
        )
    elif variant == "identifier_remap":
        identifiers = _stable_permutation(
            identifiers,
            namespace="first-identifier-remap-v1",
            query_id=query_id,
            seed=seed,
        )
    elif variant != "baseline":
        raise ValueError(f"unknown FIRST perturbation variant: {variant}")
    return tuple(
        FirstPromptEntry(candidate, position, identifier)
        for position, (candidate, identifier) in enumerate(
            zip(ordered, identifiers, strict=True),
            start=1,
        )
    )


def _clean_passage(text: str, max_words: int) -> str:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    normalized = " ".join(text.replace("Title: Content: ", "").split())
    # Match the author's protection against passage text imitating identifiers.
    normalized = _BRACKETED_ALPHA_PATTERN.sub(r"(\1)", normalized)
    return " ".join(normalized.split()[:max_words])


def render_first_user_prompt(
    query: str,
    entries: Sequence[FirstPromptEntry],
    *,
    max_passage_words: int = 300,
) -> str:
    """Render the author's alphabetic RankLLM user prompt."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not entries:
        raise ValueError("FIRST prompt requires candidates")
    size = len(entries)
    expected_positions = list(range(1, size + 1))
    if [entry.input_position for entry in entries] != expected_positions:
        raise ValueError("input positions must be contiguous and one-based")
    if len({entry.identifier for entry in entries}) != size:
        raise ValueError("prompt identifiers must be unique")
    prefix = (
        f"I will provide you with {size} passages, each indicated by a alphabetical "
        f"identifier []. Rank the passages based on their relevance to the search "
        f"query: {query}.\n\n"
    )
    passages = "".join(
        f"[{entry.identifier}] "
        f"{_clean_passage(entry.candidate.text, max_passage_words)}\n"
        for entry in entries
    )
    suffix = (
        f"Search Query: {query}.\n"
        f"Rank the {size} passages above based on their relevance to the search "
        "query. All the passages should be included and listed using identifiers, "
        "in descending order of relevance. The output format should be [] > [], "
        "e.g., [B] > [A], Only respond with the ranking results, do not say any "
        "word or explain."
    )
    return prefix + passages + suffix


def apply_first_chat_template(tokenizer: object, user_prompt: str) -> str:
    """Apply the same system-message fallback used by the author implementation."""
    chat_template = getattr(tokenizer, "chat_template", None)
    supports_system = isinstance(chat_template, str) and "system" in chat_template
    if supports_system:
        messages = [
            {"role": "system", "content": FIRST_SYSTEM_MESSAGE},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": f"{FIRST_SYSTEM_MESSAGE}\n {user_prompt}",
            }
        ]
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def prompt_token_count(tokenizer: object, prompt: str) -> int:
    """Count tokens with the checkpoint tokenizer."""
    return len(tokenizer.encode(prompt))


def fit_first_prompt(
    tokenizer: object,
    query: str,
    entries: Sequence[FirstPromptEntry],
    *,
    context_size: int = 4096,
    generation_reserve: int = 100,
    initial_max_passage_words: int = 300,
) -> tuple[str, int, int]:
    """Fit a prompt deterministically by reducing the per-passage word budget."""
    if context_size <= generation_reserve:
        raise ValueError("context_size must exceed generation_reserve")
    max_words = initial_max_passage_words
    while max_words > 0:
        user_prompt = render_first_user_prompt(
            query,
            entries,
            max_passage_words=max_words,
        )
        prompt = apply_first_chat_template(tokenizer, user_prompt)
        count = prompt_token_count(tokenizer, prompt)
        if count <= context_size - generation_reserve:
            return prompt, count, max_words
        overflow = count - (context_size - generation_reserve)
        reduction = max(1, math.ceil(overflow / max(len(entries) * 3, 1)))
        max_words -= reduction
    raise ValueError("FIRST prompt cannot fit the configured context")


def audit_identifier_tokens(
    tokenizer: object,
    identifiers: Sequence[str],
) -> dict[str, int]:
    """Verify every identifier is exactly one next token after the literal ``[``."""
    prefix_ids = list(tokenizer.encode("[", add_special_tokens=False))
    if not prefix_ids:
        raise ValueError("tokenizer produced no token for '['")
    token_ids: dict[str, int] = {}
    for identifier in identifiers:
        combined = list(
            tokenizer.encode(f"[{identifier}", add_special_tokens=False)
        )
        if combined[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"tokenization of '[{identifier}' does not preserve '[' prefix"
            )
        suffix = combined[len(prefix_ids) :]
        if len(suffix) != 1:
            raise ValueError(
                f"identifier {identifier!r} is not one token after '[': {suffix}"
            )
        if tokenizer.decode(suffix) != identifier:
            raise ValueError(
                f"identifier token for {identifier!r} decodes unexpectedly"
            )
        token_ids[identifier] = int(suffix[0])
    if len(set(token_ids.values())) != len(token_ids):
        raise ValueError("FIRST identifiers do not map to distinct token IDs")
    return token_ids


def audit_prompt_identifier_tokens(
    tokenizer: object,
    first_token_prompt: str,
    identifier_token_ids: Mapping[str, int],
) -> None:
    """Verify the same one-token identities in the fully rendered prompt context."""
    prefix_ids = list(tokenizer.encode(first_token_prompt))
    for identifier, expected_token_id in identifier_token_ids.items():
        combined = list(tokenizer.encode(first_token_prompt + identifier))
        if combined[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"prompt-context tokenization changes before identifier {identifier!r}"
            )
        suffix = combined[len(prefix_ids) :]
        if suffix != [expected_token_id]:
            raise ValueError(
                f"identifier {identifier!r} is not the expected one token in prompt context"
            )


def parse_generated_ranking(
    text: str,
    expected_identifiers: Sequence[str],
) -> tuple[str, ...]:
    """Parse a complete FIRST generation as an exact identifier permutation."""
    expected = tuple(expected_identifiers)
    parsed = tuple(_IDENTIFIER_PATTERN.findall(text.upper()))
    if len(parsed) != len(expected) or set(parsed) != set(expected):
        raise ValueError("generated ranking is not an exact identifier permutation")
    if len(set(parsed)) != len(parsed):
        raise ValueError("generated ranking contains duplicate identifiers")
    return parsed


def rank_identifiers_from_logits(
    logits: Mapping[str, float],
    expected_identifiers: Sequence[str],
) -> tuple[str, ...]:
    """Sort a complete identifier-logit map with a deterministic tie break."""
    expected = tuple(expected_identifiers)
    if set(logits) != set(expected) or len(logits) != len(expected):
        raise ValueError("logits must cover every expected identifier exactly once")
    if not all(math.isfinite(float(logits[key])) for key in expected):
        raise ValueError("identifier logits must be finite")
    original_position = {identifier: index for index, identifier in enumerate(expected)}
    return tuple(
        sorted(
            expected,
            key=lambda identifier: (
                -float(logits[identifier]),
                original_position[identifier],
            ),
        )
    )


def pair_agreement(
    left_ranking: Sequence[str],
    right_ranking: Sequence[str],
) -> float:
    """Return the fraction of unordered pairs ordered identically."""
    if len(left_ranking) < 2:
        raise ValueError("rankings require at least two identifiers")
    if (
        len(left_ranking) != len(right_ranking)
        or set(left_ranking) != set(right_ranking)
        or len(set(left_ranking)) != len(left_ranking)
    ):
        raise ValueError("rankings must be exact permutations of the same identifiers")
    left_position = {value: index for index, value in enumerate(left_ranking)}
    right_position = {value: index for index, value in enumerate(right_ranking)}
    agreements = 0
    pairs = 0
    for left_index, left in enumerate(left_ranking):
        for right in left_ranking[left_index + 1 :]:
            pairs += 1
            agreements += (left_position[left] < left_position[right]) == (
                right_position[left] < right_position[right]
            )
    return agreements / pairs


def sliding_window_ranges(
    candidate_count: int,
    *,
    window_size: int = FIRST_WINDOW_SIZE,
    step: int = FIRST_WINDOW_STEP,
    boundary_offset: int = 0,
) -> tuple[tuple[int, int], ...]:
    """Plan bottom-up FIRST windows, optionally shifting internal boundaries."""
    if not 2 <= window_size <= candidate_count:
        raise ValueError("window_size must be between 2 and candidate_count")
    if not 1 <= step < window_size:
        raise ValueError("step must be positive and smaller than window_size")
    if not 0 <= boundary_offset < step:
        raise ValueError("boundary_offset must be in [0, step)")
    end = candidate_count - boundary_offset
    ranges: list[tuple[int, int]] = []
    while end > 0:
        start = max(0, end - window_size)
        ranges.append((start, end))
        if start == 0:
            break
        end -= step
    # Shifted plans still anchor both edges so every candidate remains covered.
    if ranges[0][1] != candidate_count:
        ranges.insert(0, (candidate_count - window_size, candidate_count))
    if ranges[-1][0] != 0:
        ranges.append((0, window_size))
    return tuple(dict.fromkeys(ranges))


def first_prompt_template_sha256() -> str:
    """Fingerprint the protocol-level prompt contract."""
    return stable_sha256(
        {
            "version": FIRST_PROMPT_VERSION,
            "system": FIRST_SYSTEM_MESSAGE,
            "prefix": (
                "I will provide you with {n} passages, each indicated by a "
                "alphabetical identifier []."
            ),
            "suffix": "Only respond with the ranking results",
            "first_token_prefix": "[",
        }
    )
