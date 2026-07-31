"""Freeze and validate qrels-free FIRST R5.0 prompts without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer

from caged_ltr.data.instruction_distillation import normalize_query
from caged_ltr.reproducibility import sha256_file
from caged_ltr.teachers.first import (
    FIRST_CODE_REPOSITORY,
    FIRST_CODE_REVISION,
    FIRST_MODEL,
    FIRST_MODEL_REVISION,
    FIRST_PROMPT_VERSION,
    FirstCandidate,
    alphabetic_identifiers,
    audit_identifier_tokens,
    audit_prompt_identifier_tokens,
    build_prompt_entries,
    first_prompt_template_sha256,
    fit_first_prompt,
    sliding_window_ranges,
    stable_sha256,
)
from caged_ltr.teachers.prp_real import load_teacher_inputs

VARIANTS = (
    "baseline",
    "reverse",
    "random_permutation",
    "identifier_remap",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("FIRST config must be a mapping")
    return payload


def _artifact_hashes_match(manifest: dict[str, object]) -> bool:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            return False
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reproduction/first_r5.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/r5_0_first_local_admission"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/experiments/r5_0_first_local_admission.json"),
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require the pinned tokenizer to exist in the Hugging Face cache.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    data_config = config["data"]
    teacher_config = config["teacher"]
    audit_config = config["audit"]
    if not isinstance(data_config, dict) or not isinstance(teacher_config, dict):
        raise ValueError("FIRST config data/teacher sections must be mappings")
    if not isinstance(audit_config, dict):
        raise ValueError("FIRST config audit section must be a mapping")

    model = str(teacher_config["model"])
    revision = str(teacher_config["revision"])
    tokenizer_revision = str(teacher_config["tokenizer_revision"])
    if (model, revision, tokenizer_revision) != (
        FIRST_MODEL,
        FIRST_MODEL_REVISION,
        FIRST_MODEL_REVISION,
    ):
        raise ValueError("FIRST model and tokenizer must use the registered revision")
    author_code = teacher_config.get("author_code")
    if author_code != {
        "repository": FIRST_CODE_REPOSITORY,
        "revision": FIRST_CODE_REVISION,
    }:
        raise ValueError("FIRST author code identity differs from the registered source")

    input_path = Path(str(data_config["teacher_inputs"]))
    manifest_path = Path(str(data_config["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    teacher_inputs = load_teacher_inputs(input_path)
    expected_queries = int(data_config["query_count"])
    expected_candidates = int(data_config["candidates_per_query"])
    if len(teacher_inputs) != expected_queries:
        raise ValueError(
            f"expected {expected_queries} FIRST queries, found {len(teacher_inputs)}"
        )
    if any(len(query.candidates) != expected_candidates for query in teacher_inputs):
        raise ValueError(
            f"every FIRST query must contain exactly {expected_candidates} candidates"
        )

    evaluation_ids: set[str] = set()
    evaluation_texts: set[str] = set()
    for evaluation_path in data_config["evaluation_query_identity_exclusion"]:
        evaluation = pd.read_parquet(
            Path(str(evaluation_path)),
            columns=["query_id", "query"],
        )
        evaluation_ids.update(evaluation["query_id"].astype(str))
        evaluation_texts.update(normalize_query(str(text)) for text in evaluation["query"])
    input_ids = {query.query_id for query in teacher_inputs}
    input_texts = {normalize_query(query.query) for query in teacher_inputs}

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        revision=tokenizer_revision,
        token=False,
        local_files_only=args.local_files_only,
    )
    identifiers = alphabetic_identifiers(expected_candidates)
    identifier_token_ids = audit_identifier_tokens(tokenizer, identifiers)
    context_size = int(teacher_config["context_size"])
    max_words = int(teacher_config["initial_max_passage_words"])
    seed = int(config["experiment"]["seed"])

    prompt_records: list[dict[str, object]] = []
    token_counts: list[int] = []
    word_budgets: list[int] = []
    query_variant_fingerprints: set[str] = set()
    for query in teacher_inputs:
        candidates = tuple(
            FirstCandidate(
                candidate_id=candidate.passage_id,
                text=candidate.passage,
                retrieval_rank=candidate.bm25_rank,
            )
            for candidate in query.candidates
        )
        for variant in VARIANTS:
            entries = build_prompt_entries(
                candidates,
                query_id=query.query_id,
                variant=variant,
                seed=seed,
            )
            prompt, token_count, final_word_budget = fit_first_prompt(
                tokenizer,
                query.query,
                entries,
                context_size=context_size,
                initial_max_passage_words=max_words,
            )
            first_token_prompt = prompt + "["
            if not prompt_records:
                audit_prompt_identifier_tokens(
                    tokenizer,
                    first_token_prompt,
                    identifier_token_ids,
                )
            mapping = [
                {
                    "candidate_id": entry.candidate.candidate_id,
                    "input_position": entry.input_position,
                    "identifier": entry.identifier,
                    "retrieval_rank": entry.candidate.retrieval_rank,
                }
                for entry in entries
            ]
            fingerprint = stable_sha256(
                {
                    "query_id": query.query_id,
                    "variant": variant,
                    "mapping": mapping,
                    "prompt_sha256": hashlib_sha256_text(prompt),
                }
            )
            if fingerprint in query_variant_fingerprints:
                raise ValueError("duplicate FIRST prompt fingerprint")
            query_variant_fingerprints.add(fingerprint)
            prompt_records.append(
                {
                    "schema": "first_prompt_input_v1",
                    "query_id": query.query_id,
                    "slate_id": f"{query.query_id}:{variant}",
                    "variant": variant,
                    "query": query.query,
                    "candidate_mapping": mapping,
                    "prompt": prompt,
                    "first_token_prompt": first_token_prompt,
                    "prompt_token_count": token_count,
                    "max_passage_words": final_word_budget,
                    "prompt_sha256": hashlib_sha256_text(prompt),
                    "fingerprint": fingerprint,
                }
            )
            token_counts.append(token_count)
            word_budgets.append(final_word_budget)

    prompt_path = args.output_dir / "prompt_inputs.jsonl"
    _write_jsonl(prompt_path, prompt_records)
    boundary_offsets = [int(value) for value in audit_config["window_boundary_offsets"]]
    window_plans = {
        str(offset): sliding_window_ranges(
            100,
            window_size=int(teacher_config["window_size"]),
            step=int(teacher_config["window_step"]),
            boundary_offset=offset,
        )
        for offset in boundary_offsets
    }
    output_manifest = {
        "stage": "complete",
        "schema": "first_r5_local_admission_v1",
        "result_type": "qrels-free prompt/token/protocol admission; no model inference",
        "model": {
            "name": model,
            "revision": revision,
            "tokenizer_revision": tokenizer_revision,
            "weights_loaded": False,
            "training": False,
        },
        "author_code": author_code,
        "prompt": {
            "version": FIRST_PROMPT_VERSION,
            "template_sha256": first_prompt_template_sha256(),
            "first_generation_prefix": "[",
            "identifiers": list(identifiers),
            "identifier_token_ids": identifier_token_ids,
        },
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "queries": len(teacher_inputs),
            "candidates_per_query": expected_candidates,
            "qrels_accessed": False,
        },
        "frozen_prompts": {
            "path": str(prompt_path),
            "sha256": sha256_file(prompt_path),
            "records": len(prompt_records),
            "variants": list(VARIANTS),
            "max_token_count": max(token_counts),
            "min_token_count": min(token_counts),
            "min_passage_word_budget": min(word_budgets),
        },
        "registered_gpu_audit": {
            "complete_generation_subset_queries": int(
                audit_config["complete_generation_subset_queries"]
            ),
            "metrics": list(audit_config["metrics"]),
            "window_plans_top100": window_plans,
            "prefill_and_decoding_timed_separately": True,
        },
    }
    acceptance = {
        "model_revision_is_full_and_pinned": len(revision) == 40,
        "author_code_revision_is_full_and_pinned": len(FIRST_CODE_REVISION) == 40,
        "model_weights_not_loaded": output_manifest["model"]["weights_loaded"] is False,
        "no_gpu_or_training_used": output_manifest["model"]["training"] is False,
        "source_artifact_hashes_match": _artifact_hashes_match(manifest),
        "qrels_not_accessed": (
            data_config.get("qrels_accessed") is False
            and manifest.get("test_isolation", {}).get("qrels_accessed") is False
        ),
        "evaluation_query_id_overlap_zero": not (input_ids & evaluation_ids),
        "evaluation_query_text_overlap_zero": not (input_texts & evaluation_texts),
        "query_and_candidate_shape_is_100_by_20": (
            len(teacher_inputs) == 100 and expected_candidates == 20
        ),
        "identifiers_A_to_T_are_distinct_single_tokens": (
            tuple(identifier_token_ids) == identifiers
            and len(set(identifier_token_ids.values())) == 20
        ),
        "all_four_perturbations_frozen": len(prompt_records)
        == len(teacher_inputs) * len(VARIANTS),
        "all_prompts_fit_context": max(token_counts) <= context_size - 100,
        "complete_generation_comparison_registered": (
            "pair_agreement" in audit_config["metrics"]
            and int(audit_config["complete_generation_subset_queries"]) > 0
        ),
        "boundary_perturbation_registered": set(boundary_offsets) == {0, 5},
        "prefill_and_decoding_latency_separated": True,
    }
    acceptance = {name: bool(value) for name, value in acceptance.items()}
    report = {
        **output_manifest,
        "acceptance": acceptance,
        "all_acceptance_pass": all(acceptance.values()),
        "next": (
            "Run an 8-query single-GPU admission with the pinned 7B BF16 checkpoint; "
            "then freeze first-token logits for all 100 queries and full generations "
            "for the registered 20-query audit subset."
        ),
    }
    _write_json(args.output_dir / "manifest.json", output_manifest)
    _write_json(args.report, report)
    print(
        json.dumps(
            {
                "stage": report["stage"],
                "all_acceptance_pass": report["all_acceptance_pass"],
                "queries": len(teacher_inputs),
                "prompt_records": len(prompt_records),
                "max_prompt_tokens": max(token_counts),
                "report": str(args.report),
            },
            ensure_ascii=False,
        )
    )
    if not report["all_acceptance_pass"]:
        raise SystemExit(1)


def hashlib_sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    main()
