"""Resumable real-teacher runner for frozen TREC-DL PRP inputs."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from caged_ltr.reproducibility import sha256_file
from caged_ltr.teachers.flan_t5 import OrderedPairRequest
from caged_ltr.teachers.prp import TeacherMetadata, TeacherResponse

ProgressCallback = Callable[[Mapping[str, object]], None]
CACHE_LAYOUT = "per_query_compact_jsonl_v2"


class BatchedPairwiseTeacher(Protocol):
    @property
    def metadata(self) -> TeacherMetadata: ...

    def compare_many(
        self,
        requests: Sequence[OrderedPairRequest],
    ) -> list[TeacherResponse]: ...


@dataclass(frozen=True, slots=True)
class TRECInputCandidate:
    passage_id: str
    bm25_rank: int
    passage: str


@dataclass(frozen=True, slots=True)
class TRECInputQuery:
    request_id: str
    year: int | None
    query_id: str
    query: str
    candidates: tuple[TRECInputCandidate, ...]


def load_teacher_inputs(path: Path) -> list[TRECInputQuery]:
    """Load the qrels-free JSONL handed to the teacher process."""
    if not path.is_file():
        raise FileNotFoundError(f"teacher input not found: {path}")
    queries: list[TRECInputQuery] = []
    seen_request_ids: set[str] = set()
    forbidden = {"qrels", "relevance", "judged", "graded_relevance"}

    def payload_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                nested
                for child in value.values()
                for nested in payload_keys(child)
            }
        if isinstance(value, list):
            return {
                nested
                for child in value
                for nested in payload_keys(child)
            }
        return set()

    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid teacher input JSONL line {line_number}"
                ) from error
            if payload_keys(payload) & forbidden:
                raise ValueError("teacher input contains an evaluation-only field")
            request_id = str(payload.get("request_id", ""))
            if not request_id or request_id in seen_request_ids:
                raise ValueError("teacher request IDs must be unique and non-empty")
            seen_request_ids.add(request_id)
            candidate_payloads = payload.get("candidates")
            if not isinstance(candidate_payloads, list) or len(candidate_payloads) < 2:
                raise ValueError("each teacher request requires at least two candidates")
            candidates = tuple(
                TRECInputCandidate(
                    passage_id=str(candidate["passage_id"]),
                    bm25_rank=int(candidate["bm25_rank"]),
                    passage=str(candidate["passage"]),
                )
                for candidate in candidate_payloads
            )
            if len({candidate.passage_id for candidate in candidates}) != len(
                candidates
            ):
                raise ValueError("candidate passage IDs must be unique within a query")
            if [candidate.bm25_rank for candidate in candidates] != list(
                range(1, len(candidates) + 1)
            ):
                raise ValueError(
                    "teacher candidates must be in contiguous BM25 rank order"
                )
            queries.append(
                TRECInputQuery(
                    request_id=request_id,
                    year=(
                        int(payload["year"])
                        if payload.get("year") is not None
                        else None
                    ),
                    query_id=str(payload["query_id"]),
                    query=str(payload["query"]),
                    candidates=candidates,
                )
            )
    if not queries:
        raise ValueError("teacher input must contain at least one query")
    return queries


def ordered_allpair_requests(
    query: TRECInputQuery,
) -> list[OrderedPairRequest]:
    """Enumerate both A/B orders for every unordered candidate pair."""
    requests: list[OrderedPairRequest] = []
    for left, right in itertools.combinations(query.candidates, 2):
        requests.extend(
            (
                OrderedPairRequest(
                    request_id=query.request_id,
                    query_id=query.query_id,
                    query=query.query,
                    first_id=left.passage_id,
                    first=left.passage,
                    second_id=right.passage_id,
                    second=right.passage,
                ),
                OrderedPairRequest(
                    request_id=query.request_id,
                    query_id=query.query_id,
                    query=query.query,
                    first_id=right.passage_id,
                    first=right.passage,
                    second_id=left.passage_id,
                    second=left.passage,
                ),
            )
        )
    return requests


def _identity_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _query_cache_path(cache_dir: Path, request_id: str) -> Path:
    digest = hashlib.sha256(request_id.encode()).hexdigest()
    return cache_dir / f"{digest}.jsonl"


def _load_query_cache(
    path: Path,
    *,
    request_id: str,
) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid response cache JSONL line {line_number}"
                ) from error
            key = str(record.get("key", ""))
            if not key or key in records:
                raise ValueError("cached ordered-pair keys must be unique and non-empty")
            if str(record.get("request_id", "")) != request_id:
                raise ValueError("query cache contains a mismatched request ID")
            response = record.get("response")
            if not isinstance(response, dict):
                raise ValueError("query cache response must be an object")
            if response.get("choice") not in {"first", "second", "tie"}:
                raise ValueError("query cache contains an invalid pair choice")
            records[key] = record
    return records


def _response_record(
    request: OrderedPairRequest,
    response: TeacherResponse,
) -> dict[str, object]:
    return {
        "key": request.key,
        "request_id": request.request_id,
        "query_id": request.query_id,
        "first_id": request.first_id,
        "second_id": request.second_id,
        "response": asdict(response),
    }


def _chosen_id(record: Mapping[str, object]) -> str | None:
    response = record["response"]
    choice = str(response["choice"])
    if choice == "first":
        return str(record["first_id"])
    if choice == "second":
        return str(record["second_id"])
    return None


def _aggregate_query(
    query: TRECInputQuery,
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    scores = {candidate.passage_id: 0.0 for candidate in query.candidates}
    swap_agreements = 0
    ties = 0
    margins: list[float] = []
    comparisons = 0
    for left, right in itertools.combinations(query.candidates, 2):
        forward = records[
            f"{query.request_id}\0{left.passage_id}\0{right.passage_id}"
        ]
        reverse = records[
            f"{query.request_id}\0{right.passage_id}\0{left.passage_id}"
        ]
        forward_choice = _chosen_id(forward)
        reverse_choice = _chosen_id(reverse)
        comparisons += 1
        if forward_choice == reverse_choice and forward_choice is not None:
            swap_agreements += 1
            scores[forward_choice] += 1.0
        else:
            ties += 1
            scores[left.passage_id] += 0.5
            scores[right.passage_id] += 0.5
        for record in (forward, reverse):
            response = record["response"]
            first_score = response.get("score_first")
            second_score = response.get("score_second")
            if first_score is not None and second_score is not None:
                margins.append(abs(float(first_score) - float(second_score)))
    bm25_ranks = {
        candidate.passage_id: candidate.bm25_rank
        for candidate in query.candidates
    }
    ranking = sorted(
        scores,
        key=lambda passage_id: (
            -scores[passage_id],
            bm25_ranks[passage_id],
            passage_id,
        ),
    )
    return {
        "request_id": query.request_id,
        "year": query.year,
        "query_id": query.query_id,
        "ranking": ranking,
        "scores": scores,
        "comparisons": comparisons,
        "swap_agreement": swap_agreements / comparisons,
        "tie_ratio": ties / comparisons,
        "mean_absolute_likelihood_margin": (
            sum(margins) / len(margins) if margins else None
        ),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ndcg_linear(
    ranked_labels: Sequence[int],
    ideal_labels: Sequence[int],
    cutoff: int,
) -> float:
    dcg = sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(ranked_labels[:cutoff], 1)
    )
    ideal = sorted(ideal_labels, reverse=True)
    idcg = sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal[:cutoff], 1)
    )
    return dcg / idcg if idcg else 0.0


def evaluate_complete_rankings(
    queries: Sequence[TRECInputQuery],
    rankings: Sequence[Mapping[str, object]],
    *,
    qrels_path: Path,
) -> dict[str, object]:
    """Evaluate only after teacher inference has completed, never inside it."""
    if any(query.year is None for query in queries):
        raise ValueError("evaluation requires a year for every query")
    qrels = pd.read_parquet(qrels_path)
    relevance = {
        (str(row.request_id), str(row.passage_id)): int(row.graded_relevance)
        for row in qrels.itertuples()
    }
    ideal_by_request = {
        str(request_id): group["graded_relevance"].astype(int).tolist()
        for request_id, group in qrels.groupby("request_id", sort=False)
    }
    ranking_by_request = {
        str(ranking["request_id"]): [str(value) for value in ranking["ranking"]]
        for ranking in rankings
    }
    per_query: list[dict[str, object]] = []
    for query in queries:
        teacher_order = ranking_by_request[query.request_id]
        bm25_order = [candidate.passage_id for candidate in query.candidates]
        ideal = ideal_by_request[query.request_id]
        teacher_labels = [
            relevance.get((query.request_id, passage_id), 0)
            for passage_id in teacher_order
        ]
        bm25_labels = [
            relevance.get((query.request_id, passage_id), 0)
            for passage_id in bm25_order
        ]
        per_query.append(
            {
                "request_id": query.request_id,
                "year": query.year,
                "teacher_ndcg10": _ndcg_linear(teacher_labels, ideal, 10),
                "bm25_ndcg10": _ndcg_linear(bm25_labels, ideal, 10),
            }
        )

    def summary(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
        teacher = _mean([float(row["teacher_ndcg10"]) for row in rows])
        bm25 = _mean([float(row["bm25_ndcg10"]) for row in rows])
        return {
            "queries": float(len(rows)),
            "teacher_trec_eval_ndcg10": teacher,
            "bm25_trec_eval_ndcg10": bm25,
            "absolute_gain": teacher - bm25,
        }

    years = sorted({int(query.year) for query in queries if query.year is not None})
    return {
        "metric": "linear graded NDCG@10 over complete official qrels",
        "overall": summary(per_query),
        "by_year": {
            str(year): summary(
                [row for row in per_query if int(row["year"]) == year]
            )
            for year in years
        },
        "per_query": per_query,
    }


def run_prp_r3_1b(
    teacher: BatchedPairwiseTeacher,
    *,
    teacher_input_path: Path,
    output_dir: Path,
    qrels_path: Path,
    query_limit: int,
    batch_size: int,
    batch_order: str = "input",
    evaluate_when_complete: bool = True,
    max_ordered_prompts: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run or resume bounded-memory inference, then evaluate only when complete."""
    if query_limit <= 0:
        raise ValueError("query_limit must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_order not in {"input", "length"}:
        raise ValueError("batch_order must be 'input' or 'length'")
    if max_ordered_prompts is not None and max_ordered_prompts <= 0:
        raise ValueError("max_ordered_prompts must be positive when provided")
    queries = load_teacher_inputs(teacher_input_path)[:query_limit]
    expected_by_request = {
        query.request_id: len(query.candidates) * (len(query.candidates) - 1)
        for query in queries
    }
    expected_ordered_prompts = sum(expected_by_request.values())
    identity_payload = {
        "teacher_input_sha256": sha256_file(teacher_input_path),
        "selected_request_ids": [query.request_id for query in queries],
        "teacher": teacher.metadata.payload(),
        "protocol": "all unordered pairs in both A/B orders; conflicts become ties",
        "cache_layout": CACHE_LAYOUT,
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
            raise ValueError("cached R3.1b identity mismatch")
    else:
        _write_json(manifest_path, manifest)

    cache_dir = output_dir / "ordered_pair_responses"
    cache_dir.mkdir(exist_ok=True)
    expected_cache_paths = {
        _query_cache_path(cache_dir, query.request_id)
        for query in queries
    }
    unexpected_cache_paths = set(cache_dir.glob("*.jsonl")) - expected_cache_paths
    if unexpected_cache_paths:
        raise ValueError("response cache contains unexpected query shards")

    cached_ordered_prompts = 0
    for query in queries:
        requests = ordered_allpair_requests(query)
        expected_keys = {request.key for request in requests}
        cached = _load_query_cache(
            _query_cache_path(cache_dir, query.request_id),
            request_id=query.request_id,
        )
        if not set(cached).issubset(expected_keys):
            raise ValueError("response cache contains unexpected ordered pairs")
        cached_ordered_prompts += len(cached)

    prompt_total = (
        min(max_ordered_prompts, expected_ordered_prompts)
        if max_ordered_prompts is not None
        else expected_ordered_prompts
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "resume",
                "prompt_done": min(cached_ordered_prompts, prompt_total),
                "prompt_total": prompt_total,
                "cached_prompts": cached_ordered_prompts,
                "batch_size": batch_size,
            }
        )

    started = time.perf_counter()
    newly_scored = 0
    remaining_budget = max(prompt_total - cached_ordered_prompts, 0)
    for query in queries:
        if remaining_budget <= 0:
            break
        requests = ordered_allpair_requests(query)
        request_by_key = {request.key: request for request in requests}
        cache_path = _query_cache_path(cache_dir, query.request_id)
        cached = _load_query_cache(
            cache_path,
            request_id=query.request_id,
        )
        if not set(cached).issubset(request_by_key):
            raise ValueError("response cache contains unexpected ordered pairs")
        pending = [
            request for request in requests if request.key not in cached
        ][:remaining_budget]
        if batch_order == "length":
            pending.sort(
                key=lambda request: (
                    len(request.query)
                    + len(request.first)
                    + len(request.second),
                    request.key,
                )
            )
        with cache_path.open("a", encoding="utf-8") as output:
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                responses = teacher.compare_many(batch)
                if len(responses) != len(batch):
                    raise RuntimeError(
                        "teacher response count does not match batch size"
                    )
                for request, response in zip(batch, responses, strict=True):
                    record = _response_record(request, response)
                    output.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    cached[request.key] = record
                output.flush()
                os.fsync(output.fileno())
                newly_scored += len(batch)
                cached_ordered_prompts += len(batch)
                remaining_budget -= len(batch)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "inference",
                            "prompt_done": min(
                                cached_ordered_prompts,
                                prompt_total,
                            ),
                            "prompt_total": prompt_total,
                            "cached_prompts": (
                                cached_ordered_prompts - newly_scored
                            ),
                            "batch_size": len(batch),
                        }
                    )
    wall_seconds = time.perf_counter() - started
    complete = cached_ordered_prompts == expected_ordered_prompts
    rankings: list[dict[str, object]] = []
    truncated = 0
    exact_likelihood_ties = 0
    for query in queries:
        cached = _load_query_cache(
            _query_cache_path(cache_dir, query.request_id),
            request_id=query.request_id,
        )
        truncated += sum(
            bool(record["response"].get("input_truncated", False))
            for record in cached.values()
        )
        exact_likelihood_ties += sum(
            str(record["response"]["choice"]) == "tie"
            for record in cached.values()
        )
        if complete:
            if len(cached) != expected_by_request[query.request_id]:
                raise ValueError("complete query cache has an invalid pair count")
            rankings.append(_aggregate_query(query, cached))
    summary: dict[str, object] = {
        "stage": (
            "complete"
            if complete and evaluate_when_complete
            else "inference_complete"
            if complete
            else "admission_complete"
        ),
        "result_type": "real FLAN-T5 PRP teacher inference",
        "manifest_identity_sha256": manifest["identity_sha256"],
        "query_count": len(queries),
        "expected_ordered_prompts": expected_ordered_prompts,
        "cached_ordered_prompts": cached_ordered_prompts,
        "newly_scored_ordered_prompts": newly_scored,
        "cache_layout": CACHE_LAYOUT,
        "cache_shards": sum(path.is_file() for path in expected_cache_paths),
        "batch_size": batch_size,
        "batch_order": batch_order,
        "wall_seconds_this_run": wall_seconds,
        "new_prompt_throughput_per_second": (
            newly_scored / wall_seconds if wall_seconds > 0 else 0.0
        ),
        "truncated_inputs": truncated,
        "exact_likelihood_ties": exact_likelihood_ties,
        "invalid_outputs": 0,
        "rankings": rankings,
        "teacher": teacher.metadata.payload(),
        "qrels_accessed": complete and evaluate_when_complete,
    }
    runtime_diagnostics = getattr(teacher, "runtime_diagnostics", None)
    if callable(runtime_diagnostics):
        summary["runtime"] = runtime_diagnostics()
    if complete and evaluate_when_complete:
        summary["evaluation"] = evaluate_complete_rankings(
            queries,
            rankings,
            qrels_path=qrels_path,
        )
    if complete:
        summary["diagnostics"] = {
            "mean_swap_agreement": _mean(
                [float(ranking["swap_agreement"]) for ranking in rankings]
            ),
            "mean_tie_ratio": _mean(
                [float(ranking["tie_ratio"]) for ranking in rankings]
            ),
        }
    _write_json(output_dir / "summary.json", summary)
    return summary
