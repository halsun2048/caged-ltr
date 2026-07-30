"""Exact Sliding-K replay over a complete deterministic PRP pair cache."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from caged_ltr.reproducibility import sha256_file
from caged_ltr.teachers.prp_real import (
    TRECInputCandidate,
    TRECInputQuery,
    _aggregate_query,
    _load_query_cache,
    _mean,
    _query_cache_path,
    evaluate_complete_rankings,
    load_teacher_inputs,
    ordered_allpair_requests,
)

INITIAL_ORDERS = ("bm25", "reverse_bm25", "random_seed_42")


def _identity_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_overlay(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid Sliding-K overlay line {line_number}"
                ) from error
            key = str(record.get("key", ""))
            response = record.get("response")
            if not key or key in records:
                raise ValueError("Sliding-K overlay keys must be unique")
            if not isinstance(response, dict):
                raise ValueError("Sliding-K overlay response must be an object")
            if response.get("choice") not in {"first", "second", "tie"}:
                raise ValueError("Sliding-K overlay response choice is invalid")
            if bool(response.get("input_truncated", False)):
                raise ValueError("Sliding-K overlay must not contain truncation")
            records[key] = record
    if not records:
        raise ValueError("Sliding-K overlay must not be empty")
    return records


def _chosen_id(record: Mapping[str, object]) -> str | None:
    response = record["response"]
    choice = str(response["choice"])
    if choice == "first":
        return str(record["first_id"])
    if choice == "second":
        return str(record["second_id"])
    return None


def _initial_candidates(
    query: TRECInputQuery,
    *,
    initial_order: str,
    random_seed: int,
) -> list[TRECInputCandidate]:
    candidates = list(query.candidates)
    if initial_order == "bm25":
        return candidates
    if initial_order == "reverse_bm25":
        candidates.reverse()
        return candidates
    if initial_order == "random_seed_42":
        digest = hashlib.sha256(
            f"{random_seed}\0{query.request_id}".encode()
        ).digest()
        stable_seed = int.from_bytes(digest[:8], "big")
        random.Random(stable_seed).shuffle(candidates)
        return candidates
    raise ValueError(f"unsupported initial order: {initial_order}")


def replay_sliding_k_query(
    query: TRECInputQuery,
    records: Mapping[str, Mapping[str, object]],
    *,
    passes: int,
    initial_order: str,
    random_seed: int,
) -> tuple[dict[str, object], set[str]]:
    """Replay one query using the same strict bidirectional swap rule."""
    if passes <= 0:
        raise ValueError("passes must be positive")
    ranking = _initial_candidates(
        query,
        initial_order=initial_order,
        random_seed=random_seed,
    )
    comparisons = 0
    swap_agreements = 0
    conflicts = 0
    swaps = 0
    used_keys: set[str] = set()
    for _ in range(passes):
        for index in range(len(ranking) - 2, -1, -1):
            left = ranking[index]
            right = ranking[index + 1]
            forward_key = (
                f"{query.request_id}\0{left.passage_id}\0{right.passage_id}"
            )
            reverse_key = (
                f"{query.request_id}\0{right.passage_id}\0{left.passage_id}"
            )
            try:
                forward = records[forward_key]
                reverse = records[reverse_key]
            except KeyError as error:
                raise ValueError("Sliding-K cache is missing an ordered pair") from error
            used_keys.update((forward_key, reverse_key))
            forward_choice = _chosen_id(forward)
            reverse_choice = _chosen_id(reverse)
            comparisons += 1
            if forward_choice == reverse_choice and forward_choice is not None:
                swap_agreements += 1
                if forward_choice == right.passage_id:
                    ranking[index], ranking[index + 1] = right, left
                    swaps += 1
            else:
                conflicts += 1
    expected_comparisons = passes * (len(query.candidates) - 1)
    if comparisons != expected_comparisons:
        raise RuntimeError("Sliding-K logical comparison count is invalid")
    return (
        {
            "request_id": query.request_id,
            "year": query.year,
            "query_id": query.query_id,
            "ranking": [candidate.passage_id for candidate in ranking],
            "comparisons": comparisons,
            "logical_ordered_prompts": comparisons * 2,
            "unique_ordered_prompts": len(used_keys),
            "swap_agreement": swap_agreements / comparisons,
            "conflict_ratio": conflicts / comparisons,
            "swaps": swaps,
        },
        used_keys,
    )


def _top10_comparison(
    reference: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    reference_by_request = {
        str(ranking["request_id"]): list(ranking["ranking"])
        for ranking in reference
    }
    overlap = []
    exact_membership = 0
    exact_order = 0
    for ranking in candidate:
        request_id = str(ranking["request_id"])
        reference_top10 = reference_by_request[request_id][:10]
        candidate_top10 = list(ranking["ranking"])[:10]
        overlap.append(
            len(set(reference_top10).intersection(candidate_top10)) / 10
        )
        exact_membership += set(reference_top10) == set(candidate_top10)
        exact_order += reference_top10 == candidate_top10
    return {
        "mean_top10_membership_overlap": _mean(overlap),
        "exact_top10_membership_queries": exact_membership,
        "exact_top10_order_queries": exact_order,
    }


def _gain_retention(
    method_evaluation: Mapping[str, object],
    allpair_evaluation: Mapping[str, object],
) -> dict[str, float]:
    rows = {
        "overall": (
            method_evaluation["overall"],
            allpair_evaluation["overall"],
        )
    }
    for year in sorted(method_evaluation["by_year"]):
        rows[year] = (
            method_evaluation["by_year"][year],
            allpair_evaluation["by_year"][year],
        )
    retention = {}
    for name, (method, allpair) in rows.items():
        method_gain = float(method["absolute_gain"])
        allpair_gain = float(allpair["absolute_gain"])
        retention[name] = method_gain / allpair_gain if allpair_gain else 0.0
    return retention


def run_sliding10_cached_replay(
    *,
    teacher_input_path: Path,
    qrels_path: Path,
    allpair_output_dir: Path,
    truncation_overlay_path: Path,
    output_dir: Path,
    passes: int = 10,
    random_seed: int = 42,
    expected_allpair_manifest_identity: str | None = None,
    expected_overlay_manifest_identity: str | None = None,
) -> dict[str, object]:
    """Fix every ranking from cache, then access qrels exactly once."""
    if passes <= 0:
        raise ValueError("passes must be positive")
    started = time.perf_counter()
    queries = load_teacher_inputs(teacher_input_path)
    allpair_manifest = json.loads(
        (allpair_output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        expected_allpair_manifest_identity is not None
        and allpair_manifest.get("identity_sha256")
        != expected_allpair_manifest_identity
    ):
        raise ValueError("Allpair manifest identity mismatch")
    overlay_manifest = json.loads(
        (truncation_overlay_path.parent / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        expected_overlay_manifest_identity is not None
        and overlay_manifest.get("identity_sha256")
        != expected_overlay_manifest_identity
    ):
        raise ValueError("truncation overlay manifest identity mismatch")
    overlay = _load_overlay(truncation_overlay_path)
    remaining_overlay_keys = set(overlay)
    allpair_rankings: list[dict[str, object]] = []
    method_rankings = {name: [] for name in INITIAL_ORDERS}
    unique_keys = {name: set() for name in INITIAL_ORDERS}
    cache_dir = allpair_output_dir / "ordered_pair_responses"

    for query in queries:
        expected_keys = {
            request.key for request in ordered_allpair_requests(query)
        }
        records = _load_query_cache(
            _query_cache_path(cache_dir, query.request_id),
            request_id=query.request_id,
        )
        if set(records) != expected_keys:
            raise ValueError(f"Allpair cache is incomplete for {query.request_id}")
        query_overlay_keys = set(records).intersection(remaining_overlay_keys)
        for key in query_overlay_keys:
            replacement = dict(records[key])
            replacement["response"] = dict(overlay[key]["response"])
            records[key] = replacement
        remaining_overlay_keys.difference_update(query_overlay_keys)
        allpair_rankings.append(_aggregate_query(query, records))
        for initial_order in INITIAL_ORDERS:
            ranking, used = replay_sliding_k_query(
                query,
                records,
                passes=passes,
                initial_order=initial_order,
                random_seed=random_seed,
            )
            method_rankings[initial_order].append(ranking)
            unique_keys[initial_order].update(used)
    if remaining_overlay_keys:
        raise ValueError("truncation overlay contains keys outside the input")

    rankings_fixed_seconds = time.perf_counter() - started
    allpair_evaluation = evaluate_complete_rankings(
        queries,
        allpair_rankings,
        qrels_path=qrels_path,
    )
    method_evaluations = {
        name: evaluate_complete_rankings(
            queries,
            rankings,
            qrels_path=qrels_path,
        )
        for name, rankings in method_rankings.items()
    }
    logical_prompts_per_order = sum(
        int(ranking["logical_ordered_prompts"])
        for ranking in method_rankings["bm25"]
    )
    allpair_ordered_prompts = sum(
        len(query.candidates) * (len(query.candidates) - 1)
        for query in queries
    )
    methods = {}
    for name in INITIAL_ORDERS:
        rankings = method_rankings[name]
        methods[name] = {
            "evaluation": method_evaluations[name],
            "gain_retention_vs_allpair": _gain_retention(
                method_evaluations[name],
                allpair_evaluation,
            ),
            "top10_vs_allpair": _top10_comparison(
                allpair_rankings,
                rankings,
            ),
            "logical_ordered_prompts": logical_prompts_per_order,
            "unique_ordered_prompts_accessed": len(unique_keys[name]),
            "logical_prompt_ratio_vs_allpair": (
                logical_prompts_per_order / allpair_ordered_prompts
            ),
            "mean_swap_agreement": _mean(
                [float(ranking["swap_agreement"]) for ranking in rankings]
            ),
            "mean_conflict_ratio": _mean(
                [float(ranking["conflict_ratio"]) for ranking in rankings]
            ),
            "total_swaps": sum(int(ranking["swaps"]) for ranking in rankings),
        }
    bm25_evaluation = method_evaluations["bm25"]
    positive_gain_by_year = {
        str(year): float(row["absolute_gain"]) > 0
        for year, row in bm25_evaluation["by_year"].items()
    }
    acceptance = {
        "bm25_order_positive_gain_each_year": all(
            positive_gain_by_year.values()
        ),
        "bm25_order_overall_gain_retention_at_least_0p9": (
            float(methods["bm25"]["gain_retention_vs_allpair"]["overall"]) >= 0.9
        ),
        "logical_prompt_ratio_at_most_0p25": (
            float(methods["bm25"]["logical_prompt_ratio_vs_allpair"]) <= 0.25
        ),
        "reverse_and_random_reported_without_selection": True,
        "zero_new_gpu_calls": True,
        "qrels_delayed_until_all_rankings_complete": True,
    }
    acceptance["all_pre_registered_gates_passed"] = all(acceptance.values())
    summary = {
        "stage": "complete",
        "result_type": "exact cached FLAN-T5 PRP Sliding-10 replay",
        "protocol": {
            "queries": len(queries),
            "passes": passes,
            "initial_orders": list(INITIAL_ORDERS),
            "random_seed": random_seed,
            "allpair_manifest_identity_sha256": allpair_manifest[
                "identity_sha256"
            ],
            "truncation_overlay_manifest_identity_sha256": overlay_manifest[
                "identity_sha256"
            ],
            "truncation_overlay_records": len(overlay),
            "rankings_fixed_before_qrels_access": True,
            "new_gpu_calls": 0,
        },
        "allpair": {
            "evaluation": allpair_evaluation,
            "ordered_prompts": allpair_ordered_prompts,
        },
        "methods": methods,
        "initial_order_sensitivity": {
            "reverse_vs_bm25_top10": _top10_comparison(
                method_rankings["bm25"],
                method_rankings["reverse_bm25"],
            ),
            "random_vs_bm25_top10": _top10_comparison(
                method_rankings["bm25"],
                method_rankings["random_seed_42"],
            ),
            "reverse_overall_ndcg_delta_from_bm25": (
                float(
                    method_evaluations["reverse_bm25"]["overall"][
                        "teacher_trec_eval_ndcg10"
                    ]
                )
                - float(
                    method_evaluations["bm25"]["overall"][
                        "teacher_trec_eval_ndcg10"
                    ]
                )
            ),
            "random_overall_ndcg_delta_from_bm25": (
                float(
                    method_evaluations["random_seed_42"]["overall"][
                        "teacher_trec_eval_ndcg10"
                    ]
                )
                - float(
                    method_evaluations["bm25"]["overall"][
                        "teacher_trec_eval_ndcg10"
                    ]
                )
            ),
        },
        "acceptance": acceptance,
        "acceptance_diagnostics": {
            "bm25_order_positive_gain_by_year": positive_gain_by_year,
        },
        "runtime": {
            "device": "cpu_cache_replay",
            "rankings_fixed_seconds": rankings_fixed_seconds,
            "wall_seconds": time.perf_counter() - started,
        },
        "artifacts": {
            "teacher_input_sha256": sha256_file(teacher_input_path),
            "truncation_overlay_sha256": sha256_file(
                truncation_overlay_path
            ),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
