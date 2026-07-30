"""Isolated sensitivity audit for truncated PRP Allpair prompts."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from caged_ltr.reproducibility import sha256_file
from caged_ltr.teachers.flan_t5 import OrderedPairRequest
from caged_ltr.teachers.prp import TeacherMetadata, TeacherResponse
from caged_ltr.teachers.prp_real import (
    TRECInputQuery,
    _aggregate_query,
    _load_query_cache,
    _mean,
    _query_cache_path,
    evaluate_complete_rankings,
    load_teacher_inputs,
    ordered_allpair_requests,
)

ProgressCallback = Callable[[Mapping[str, object]], None]


class BatchedPairwiseTeacher(Protocol):
    @property
    def metadata(self) -> TeacherMetadata: ...

    def compare_many(
        self,
        requests: Sequence[OrderedPairRequest],
    ) -> list[TeacherResponse]: ...


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


def _load_audit_cache(
    path: Path,
    *,
    expected_keys: set[str],
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
                    f"invalid truncation audit cache line {line_number}"
                ) from error
            key = str(record.get("key", ""))
            if key not in expected_keys or key in records:
                raise ValueError("truncation audit cache key is unexpected or duplicate")
            response = record.get("response")
            if not isinstance(response, dict):
                raise ValueError("truncation audit response must be an object")
            if response.get("choice") not in {"first", "second", "tie"}:
                raise ValueError("truncation audit response choice is invalid")
            records[key] = record
    return records


def discover_truncated_requests(
    *,
    teacher_input_path: Path,
    baseline_output_dir: Path,
) -> tuple[
    list[TRECInputQuery],
    list[tuple[OrderedPairRequest, dict[str, object]]],
]:
    """Find the exact ordered prompts marked truncated in a complete baseline."""
    queries = load_teacher_inputs(teacher_input_path)
    baseline_cache_dir = baseline_output_dir / "ordered_pair_responses"
    targets: list[tuple[OrderedPairRequest, dict[str, object]]] = []
    for query in queries:
        requests = ordered_allpair_requests(query)
        request_by_key = {request.key: request for request in requests}
        records = _load_query_cache(
            _query_cache_path(baseline_cache_dir, query.request_id),
            request_id=query.request_id,
        )
        if set(records) != set(request_by_key):
            raise ValueError(
                f"baseline cache is incomplete for {query.request_id}"
            )
        targets.extend(
            (request_by_key[key], record)
            for key, record in records.items()
            if bool(record["response"].get("input_truncated", False))
        )
    targets.sort(key=lambda item: item[0].key)
    if not targets:
        raise ValueError("baseline contains no truncated prompts to audit")
    return queries, targets


def _validate_teacher_identity(
    baseline: Mapping[str, object],
    audit: Mapping[str, object],
) -> None:
    fields = (
        "backend",
        "model_name",
        "model_revision",
        "tokenizer_name",
        "tokenizer_revision",
        "quantization",
        "prompt_name",
        "prompt_version",
        "prompt_sha256",
    )
    if any(baseline.get(field) != audit.get(field) for field in fields):
        raise ValueError("audit teacher identity differs from the baseline")
    baseline_parameters = dict(baseline["generation_parameters"])
    audit_parameters = dict(audit["generation_parameters"])
    baseline_limit = int(baseline_parameters.pop("max_input_tokens"))
    audit_limit = int(audit_parameters.pop("max_input_tokens"))
    if baseline_parameters != audit_parameters:
        raise ValueError("audit generation parameters differ from the baseline")
    if audit_limit <= baseline_limit:
        raise ValueError("audit input limit must be larger than the baseline")


def _replace_response(
    baseline_record: Mapping[str, object],
    audit_record: Mapping[str, object],
) -> dict[str, object]:
    combined = dict(baseline_record)
    combined["response"] = dict(audit_record["response"])
    return combined


def _ranking_diagnostics(
    baseline_rankings: Sequence[Mapping[str, object]],
    audited_rankings: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    baseline_by_request = {
        str(ranking["request_id"]): list(ranking["ranking"])
        for ranking in baseline_rankings
    }
    audited_by_request = {
        str(ranking["request_id"]): list(ranking["ranking"])
        for ranking in audited_rankings
    }
    ranking_changed = 0
    top10_order_changed = 0
    top10_membership_changed = 0
    maximum_rank_shift = 0
    for request_id, baseline in baseline_by_request.items():
        audited = audited_by_request[request_id]
        ranking_changed += baseline != audited
        top10_order_changed += baseline[:10] != audited[:10]
        top10_membership_changed += set(baseline[:10]) != set(audited[:10])
        baseline_positions = {
            passage_id: index for index, passage_id in enumerate(baseline)
        }
        maximum_rank_shift = max(
            maximum_rank_shift,
            max(
                abs(index - baseline_positions[passage_id])
                for index, passage_id in enumerate(audited)
            ),
        )
    return {
        "ranking_changed_queries": ranking_changed,
        "top10_order_changed_queries": top10_order_changed,
        "top10_membership_changed_queries": top10_membership_changed,
        "maximum_candidate_rank_shift": maximum_rank_shift,
    }


def run_truncation_sensitivity_audit(
    teacher: BatchedPairwiseTeacher,
    *,
    teacher_input_path: Path,
    baseline_output_dir: Path,
    output_dir: Path,
    qrels_path: Path,
    batch_size: int,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """Rescore only truncated prompts and evaluate an isolated cache overlay."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    queries, targets = discover_truncated_requests(
        teacher_input_path=teacher_input_path,
        baseline_output_dir=baseline_output_dir,
    )
    baseline_manifest = json.loads(
        (baseline_output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    audit_teacher = teacher.metadata.payload()
    _validate_teacher_identity(baseline_manifest["teacher"], audit_teacher)
    target_payload = [
        {
            "key": request.key,
            "baseline_response": baseline_record["response"],
        }
        for request, baseline_record in targets
    ]
    identity_payload = {
        "protocol": "isolated rescore of baseline input_truncated=true prompts",
        "teacher_input_sha256": sha256_file(teacher_input_path),
        "baseline_manifest_identity_sha256": baseline_manifest["identity_sha256"],
        "target_records_sha256": _identity_sha256(target_payload),
        "target_count": len(targets),
        "audit_teacher": audit_teacher,
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
            raise ValueError("cached truncation audit identity mismatch")
    else:
        _write_json(manifest_path, manifest)

    cache_path = output_dir / "rescored_truncated_responses.jsonl"
    expected_keys = {request.key for request, _ in targets}
    cached = _load_audit_cache(cache_path, expected_keys=expected_keys)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "resume",
                "prompt_done": len(cached),
                "prompt_total": len(targets),
                "batch_size": batch_size,
            }
        )
    pending = [
        request for request, _ in targets if request.key not in cached
    ]
    pending.sort(
        key=lambda request: (
            len(request.query) + len(request.first) + len(request.second),
            request.key,
        )
    )
    started = time.perf_counter()
    newly_scored = 0
    with cache_path.open("a", encoding="utf-8") as output:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            responses = teacher.compare_many(batch)
            if len(responses) != len(batch):
                raise RuntimeError("teacher response count does not match batch size")
            for request, response in zip(batch, responses, strict=True):
                record = {
                    "key": request.key,
                    "request_id": request.request_id,
                    "query_id": request.query_id,
                    "first_id": request.first_id,
                    "second_id": request.second_id,
                    "response": asdict(response),
                }
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
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "inference",
                        "prompt_done": len(cached),
                        "prompt_total": len(targets),
                        "batch_size": len(batch),
                    }
                )
    wall_seconds = time.perf_counter() - started
    cached = _load_audit_cache(cache_path, expected_keys=expected_keys)
    complete = len(cached) == len(targets)
    remaining_truncated = sum(
        bool(record["response"].get("input_truncated", False))
        for record in cached.values()
    )
    summary: dict[str, object] = {
        "stage": (
            "complete"
            if complete and remaining_truncated == 0
            else "inference_complete_still_truncated"
            if complete
            else "incomplete"
        ),
        "result_type": "isolated PRP input-truncation sensitivity audit",
        "manifest_identity_sha256": manifest["identity_sha256"],
        "baseline_manifest_identity_sha256": baseline_manifest["identity_sha256"],
        "target_ordered_prompts": len(targets),
        "cached_ordered_prompts": len(cached),
        "newly_scored_ordered_prompts": newly_scored,
        "remaining_truncated_inputs": remaining_truncated,
        "wall_seconds_this_run": wall_seconds,
        "new_prompt_throughput_per_second": (
            newly_scored / wall_seconds if wall_seconds > 0 else 0.0
        ),
        "qrels_accessed": False,
        "teacher": audit_teacher,
    }
    runtime_diagnostics = getattr(teacher, "runtime_diagnostics", None)
    if callable(runtime_diagnostics):
        summary["runtime"] = runtime_diagnostics()
    if not complete or remaining_truncated:
        _write_json(output_dir / "summary.json", summary)
        return summary

    baseline_cache_dir = baseline_output_dir / "ordered_pair_responses"
    baseline_rankings: list[dict[str, object]] = []
    audited_rankings: list[dict[str, object]] = []
    choice_changes = 0
    for query in queries:
        baseline_records = _load_query_cache(
            _query_cache_path(baseline_cache_dir, query.request_id),
            request_id=query.request_id,
        )
        audited_records = dict(baseline_records)
        for key in set(baseline_records).intersection(cached):
            choice_changes += (
                baseline_records[key]["response"]["choice"]
                != cached[key]["response"]["choice"]
            )
            audited_records[key] = _replace_response(
                baseline_records[key],
                cached[key],
            )
        baseline_rankings.append(_aggregate_query(query, baseline_records))
        audited_rankings.append(_aggregate_query(query, audited_records))

    baseline_evaluation = evaluate_complete_rankings(
        queries,
        baseline_rankings,
        qrels_path=qrels_path,
    )
    audited_evaluation = evaluate_complete_rankings(
        queries,
        audited_rankings,
        qrels_path=qrels_path,
    )
    baseline_overall = baseline_evaluation["overall"]
    audited_overall = audited_evaluation["overall"]
    summary.update(
        {
            "qrels_accessed": True,
            "choice_changes": choice_changes,
            "ranking_diagnostics": _ranking_diagnostics(
                baseline_rankings,
                audited_rankings,
            ),
            "baseline_evaluation": baseline_evaluation,
            "audited_evaluation": audited_evaluation,
            "overall_ndcg_at_10_delta": (
                float(audited_overall["teacher_trec_eval_ndcg10"])
                - float(baseline_overall["teacher_trec_eval_ndcg10"])
            ),
            "audited_diagnostics": {
                "mean_swap_agreement": _mean(
                    [
                        float(ranking["swap_agreement"])
                        for ranking in audited_rankings
                    ]
                ),
                "mean_tie_ratio": _mean(
                    [float(ranking["tie_ratio"]) for ranking in audited_rankings]
                ),
            },
        }
    )
    _write_json(output_dir / "summary.json", summary)
    return summary
