"""Resumable synthetic smoke test for the PRP teacher protocol."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from caged_ltr.evaluation.metrics import ranking_metrics
from caged_ltr.teachers.prp import (
    DeterministicMockTeacher,
    PRPCandidate,
    PRPQuery,
    PRPRanking,
    allpair_borda,
    pair_diagnostics,
    query_fingerprint,
    sliding_k,
)

ProgressCallback = Callable[[Mapping[str, object]], None]
ORDER_NAMES = ("bm25", "reverse", "random")


def synthetic_prp_queries(
    *,
    seed: int,
    query_count: int = 100,
    candidates_per_query: int = 10,
) -> list[PRPQuery]:
    """Build deterministic graded-relevance queries with imperfect initial rankings."""
    if query_count <= 0:
        raise ValueError("query_count must be positive")
    if candidates_per_query < 3:
        raise ValueError("candidates_per_query must be at least 3")
    generator = np.random.default_rng(seed)
    queries: list[PRPQuery] = []
    for query_index in range(query_count):
        latent = generator.normal(size=candidates_per_query)
        ideal_order = np.argsort(-latent, kind="stable")
        relevance = np.zeros(candidates_per_query, dtype=np.float64)
        relevance[ideal_order[0]] = 3.0
        relevance[ideal_order[1 : min(3, candidates_per_query)]] = 2.0
        relevance[ideal_order[3 : min(6, candidates_per_query)]] = 1.0
        initial_scores = latent + generator.normal(scale=1.15, size=candidates_per_query)
        teacher_scores = latent + generator.normal(scale=0.12, size=candidates_per_query)
        candidates = [
            PRPCandidate(
                candidate_id=f"q{query_index:03d}-d{candidate_index:02d}",
                text=(
                    f"Synthetic passage {candidate_index} for topic {query_index}. "
                    "This text only validates serialization and pair direction."
                ),
                relevance=float(relevance[candidate_index]),
                initial_score=float(initial_scores[candidate_index]),
                teacher_score=float(teacher_scores[candidate_index]),
            )
            for candidate_index in range(candidates_per_query)
        ]
        candidates.sort(key=lambda candidate: (-candidate.initial_score, candidate.candidate_id))
        queries.append(
            PRPQuery(
                query_id=f"q{query_index:03d}",
                text=f"Synthetic information need {query_index}",
                candidates=tuple(candidates),
            )
        )
    return queries


def _ordered_query(query: PRPQuery, order_name: str, *, seed: int) -> PRPQuery:
    candidate_ids = [candidate.candidate_id for candidate in query.candidates]
    if order_name == "bm25":
        ordered_ids = candidate_ids
    elif order_name == "reverse":
        ordered_ids = list(reversed(candidate_ids))
    elif order_name == "random":
        key = hashlib.sha256(f"{seed}\0{query.query_id}".encode()).digest()
        local_seed = int.from_bytes(key[:8], "big")
        generator = np.random.default_rng(local_seed)
        ordered_ids = list(generator.permutation(candidate_ids))
    else:
        raise ValueError(f"unknown initial order: {order_name}")
    return query.reordered(ordered_ids)


def _ranking_cost(result: PRPRanking) -> dict[str, float | int]:
    return {
        "prompts": result.prompt_count,
        "input_tokens": sum(
            comparison.input_tokens for comparison in result.comparisons
        ),
        "output_tokens": sum(
            comparison.output_tokens for comparison in result.comparisons
        ),
        "latency_ms": sum(
            comparison.latency_ms for comparison in result.comparisons
        ),
    }


def _run_one_query(
    query: PRPQuery,
    teacher: DeterministicMockTeacher,
    *,
    sliding_passes: int,
    seed: int,
    prompt_progress: Callable[[str, int, int], None] | None,
) -> dict[str, object]:
    orders: dict[str, object] = {}
    total_cost: dict[str, float] = {
        "prompts": 0.0,
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "latency_ms": 0.0,
    }
    for order_name in ORDER_NAMES:
        ordered_query = _ordered_query(query, order_name, seed=seed)
        allpair_seen = 0

        def allpair_progress(
            stage: str,
            done: int,
            total: int,
            bound_order: str = order_name,
        ) -> None:
            nonlocal allpair_seen
            delta = done - allpair_seen
            allpair_seen = done
            if prompt_progress is not None and delta:
                prompt_progress(f"{bound_order}/{stage}", delta, total)

        allpair = allpair_borda(
            teacher,
            ordered_query,
            progress_callback=allpair_progress,
        )
        sliding_seen = 0

        def sliding_progress(
            stage: str,
            done: int,
            total: int,
            bound_order: str = order_name,
        ) -> None:
            nonlocal sliding_seen
            delta = done - sliding_seen
            sliding_seen = done
            if prompt_progress is not None and delta:
                prompt_progress(f"{bound_order}/{stage}", delta, total)

        sliding = sliding_k(
            teacher,
            ordered_query,
            passes=sliding_passes,
            progress_callback=sliding_progress,
        )
        order_cost: dict[str, float] = {
            "prompts": 0.0,
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "latency_ms": 0.0,
        }
        for result in (allpair, sliding):
            for field, value in _ranking_cost(result).items():
                order_cost[field] += float(value)
                total_cost[field] += float(value)
        orders[order_name] = {
            "initial_ranking": [
                candidate.candidate_id for candidate in ordered_query.candidates
            ],
            "allpair": allpair.payload(),
            "sliding": sliding.payload(),
            "allpair_diagnostics": pair_diagnostics(
                ordered_query,
                allpair.comparisons,
            ),
            "cost": {
                key: int(value) if key != "latency_ms" else value
                for key, value in order_cost.items()
            },
        }
    return {
        "query_id": query.query_id,
        "orders": orders,
        "cost": {
            key: int(value) if key != "latency_ms" else value
            for key, value in total_cost.items()
        },
    }


def _identity_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_cached_records(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid cached JSONL at line {line_number}") from error
        query_id = str(record.get("query_id", ""))
        if not query_id or query_id in records:
            raise ValueError("cached query records must have unique non-empty query IDs")
        records[query_id] = record
    return records


def _ranking_report(
    queries: Sequence[PRPQuery],
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    reports: dict[str, dict[str, float]] = {}
    for order_name in ORDER_NAMES:
        for method in ("initial", "allpair", "sliding"):
            labels: list[float] = []
            scores: list[float] = []
            group_ids: list[str] = []
            for query in queries:
                query_record = records[query.query_id]
                order_record = query_record["orders"][order_name]
                candidate_by_id = {
                    candidate.candidate_id: candidate for candidate in query.candidates
                }
                if method == "initial":
                    initial_ranking = order_record["initial_ranking"]
                    method_scores = {
                        candidate_id: float(len(initial_ranking) - index)
                        for index, candidate_id in enumerate(initial_ranking)
                    }
                else:
                    method_scores = order_record[method]["scores"]
                for candidate_id, candidate in candidate_by_id.items():
                    labels.append(candidate.relevance)
                    scores.append(float(method_scores[candidate_id]))
                    group_ids.append(query.query_id)
            reports[f"{order_name}/{method}"] = ranking_metrics(
                labels,
                scores,
                group_ids,
                cutoffs=(5, 10),
            )
    return reports


def _mean_diagnostics(
    records: Mapping[str, Mapping[str, object]],
    order_name: str,
) -> dict[str, float]:
    fields = (
        "swap_agreement",
        "tie_ratio",
        "pair_coverage",
        "pair_accuracy",
        "pair_accuracy_with_tie_credit",
        "cycle_rate",
    )
    return {
        field: float(
            np.mean(
                [
                    float(record["orders"][order_name]["allpair_diagnostics"][field])
                    for record in records.values()
                ]
            )
        )
        for field in fields
    }


def _stability_report(
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for method in ("allpair", "sliding"):
        bm25_rankings = {
            query_id: tuple(record["orders"]["bm25"][method]["ranking"])
            for query_id, record in records.items()
        }
        for order_name in ("reverse", "random"):
            matches = [
                tuple(record["orders"][order_name][method]["ranking"])
                == bm25_rankings[query_id]
                for query_id, record in records.items()
            ]
            result[f"{method}_{order_name}_exact_match"] = float(np.mean(matches))
    return result


def _build_summary(
    queries: Sequence[PRPQuery],
    records: Mapping[str, Mapping[str, object]],
    *,
    expected_prompts: int,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    ranking = _ranking_report(queries, records)
    stability = _stability_report(records)
    cost = {
        field: sum(
            float(record["cost"][field]) for record in records.values()
        )
        for field in ("prompts", "input_tokens", "output_tokens", "latency_ms")
    }
    cost["prompts"] = int(cost["prompts"])
    cost["input_tokens"] = int(cost["input_tokens"])
    cost["output_tokens"] = int(cost["output_tokens"])
    allpair_prompts = sum(
        int(record["orders"][order_name]["allpair"]["prompt_count"])
        for record in records.values()
        for order_name in ORDER_NAMES
    )
    sliding_prompts = sum(
        int(record["orders"][order_name]["sliding"]["prompt_count"])
        for record in records.values()
        for order_name in ORDER_NAMES
    )
    expected_allpair_prompts = (
        len(queries)
        * len(ORDER_NAMES)
        * len(queries[0].candidates)
        * (len(queries[0].candidates) - 1)
    )
    cost["allpair_prompts"] = allpair_prompts
    cost["sliding_prompts"] = sliding_prompts
    diagnostics = {
        order_name: _mean_diagnostics(records, order_name)
        for order_name in ORDER_NAMES
    }
    acceptance = {
        "all_queries_serialized_once": len(records) == len(queries),
        "total_prompt_count_exact": cost["prompts"] == expected_prompts,
        "allpair_prompt_count_exact": allpair_prompts == expected_allpair_prompts,
        "allpair_reverse_exact_all_queries": (
            stability["allpair_reverse_exact_match"] == 1.0
        ),
        "allpair_random_exact_all_queries": (
            stability["allpair_random_exact_match"] == 1.0
        ),
        "mock_allpair_beats_initial_ndcg10": (
            ranking["bm25/allpair"]["NDCG@10"]
            > ranking["bm25/initial"]["NDCG@10"]
        ),
        "bm25_swap_agreement_at_least_0p9": (
            diagnostics["bm25"]["swap_agreement"] >= 0.9
        ),
        "mock_cycle_rate_zero": diagnostics["bm25"]["cycle_rate"] == 0.0,
    }
    return {
        "stage": "complete",
        "result_type": "synthetic deterministic pipeline smoke; not a PRP reproduction",
        "manifest_identity_sha256": manifest["identity_sha256"],
        "query_count": len(queries),
        "candidates_per_query": len(queries[0].candidates),
        "orders": list(ORDER_NAMES),
        "ranking": ranking,
        "diagnostics": diagnostics,
        "stability": stability,
        "cost": cost,
        "expected_prompt_count": expected_prompts,
        "expected_allpair_prompt_count": expected_allpair_prompts,
        "acceptance": acceptance,
    }


def run_prp_smoke(
    output_dir: Path,
    *,
    seed: int = 42,
    query_count: int = 100,
    candidates_per_query: int = 10,
    sliding_passes: int = 3,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run or resume the deterministic PRP protocol smoke test."""
    if sliding_passes <= 0:
        raise ValueError("sliding_passes must be positive")
    queries = synthetic_prp_queries(
        seed=seed,
        query_count=query_count,
        candidates_per_query=candidates_per_query,
    )
    teacher = DeterministicMockTeacher(
        seed=seed,
        position_bias=0.025,
        noise_scale=0.015,
        tie_margin=0.005,
    )
    config = {
        "seed": seed,
        "query_count": query_count,
        "candidates_per_query": candidates_per_query,
        "sliding_passes": sliding_passes,
        "initial_orders": list(ORDER_NAMES),
        "protocol": "bidirectional strict preference; inconsistency becomes tie",
    }
    identity_payload = {
        "config": config,
        "query_fingerprint": query_fingerprint(queries),
        "teacher": teacher.metadata.payload(),
    }
    manifest = {
        **identity_payload,
        "identity_sha256": _identity_sha256(identity_payload),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached_manifest.get("identity_sha256") != manifest["identity_sha256"]:
            raise ValueError("cached PRP smoke identity mismatch")
    else:
        _write_json(manifest_path, manifest)

    records_path = output_dir / "query_results.jsonl"
    cached = _load_cached_records(records_path)
    expected_query_ids = {query.query_id for query in queries}
    if not set(cached).issubset(expected_query_ids):
        raise ValueError("cached PRP smoke contains unexpected query IDs")

    prompts_per_order = candidates_per_query * (candidates_per_query - 1) + (
        2 * sliding_passes * (candidates_per_query - 1)
    )
    prompts_per_query = len(ORDER_NAMES) * prompts_per_order
    expected_prompts = query_count * prompts_per_query
    completed_prompts = sum(int(record["cost"]["prompts"]) for record in cached.values())

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "resume",
                "query_done": len(cached),
                "query_total": query_count,
                "prompt_done": completed_prompts,
                "prompt_total": expected_prompts,
                "cached_queries": len(cached),
            }
        )

    with records_path.open("a", encoding="utf-8") as output:
        for query_index, query in enumerate(queries, start=1):
            if query.query_id in cached:
                continue
            current_stage = "starting"

            def prompt_progress(
                stage: str,
                delta: int,
                method_total: int,
                bound_query_index: int = query_index,
            ) -> None:
                del method_total
                nonlocal completed_prompts, current_stage
                completed_prompts += delta
                current_stage = stage
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": current_stage,
                            "query_done": bound_query_index - 1,
                            "query_total": query_count,
                            "prompt_done": completed_prompts,
                            "prompt_total": expected_prompts,
                            "cached_queries": len(cached),
                        }
                    )

            record = _run_one_query(
                query,
                teacher,
                sliding_passes=sliding_passes,
                seed=seed,
                prompt_progress=prompt_progress,
            )
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            output.flush()
            os.fsync(output.fileno())
            cached[query.query_id] = record
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "query_complete",
                        "query_done": query_index,
                        "query_total": query_count,
                        "prompt_done": completed_prompts,
                        "prompt_total": expected_prompts,
                        "cached_queries": len(cached) - 1,
                    }
                )

    summary = _build_summary(
        queries,
        cached,
        expected_prompts=expected_prompts,
        manifest=manifest,
    )
    _write_json(output_dir / "summary.json", summary)
    return summary
