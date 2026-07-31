"""Run the bounded R5.1 FIRST inference admission with progress and resume."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from caged_ltr.reproducibility import sha256_file
from caged_ltr.teachers.first import (
    FIRST_MODEL,
    FIRST_MODEL_REVISION,
    JsonlResultCache,
    normalized_entropy,
    pair_agreement,
    parse_generated_ranking,
    rank_identifiers_from_logits,
    stable_sha256,
    top1_top2_margin,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_inputs(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            record = json.loads(line)
            key = str(record.get("fingerprint", ""))
            if not key or key in seen:
                raise ValueError(f"invalid or duplicate frozen prompt at line {line_number}")
            if record.get("schema") != "first_prompt_input_v1":
                raise ValueError("unexpected frozen FIRST prompt schema")
            mapping = record.get("candidate_mapping")
            if not isinstance(mapping, list) or not 2 <= len(mapping) <= 26:
                raise ValueError("FIRST inference requires 2-26 candidates")
            if not isinstance(record.get("prompt"), str):
                raise ValueError("frozen prompt is missing prompt text")
            if not isinstance(record.get("first_token_prompt"), str):
                raise ValueError("frozen prompt is missing first-token prompt text")
            seen.add(key)
            records.append(record)
    if not records:
        raise ValueError("frozen FIRST prompt input is empty")
    return records


def _progress(done: int, total: int, started: float, *, cached: bool = False) -> None:
    elapsed = time.monotonic() - started
    rate = done / elapsed if elapsed else 0.0
    eta = (total - done) / rate if rate else 0.0
    width = 24
    filled = round(width * done / max(total, 1))
    status = "cached" if cached else "done"
    print(
        f"\r[{('#' * filled):<{width}}] {status:<6} {done:>3}/{total:<3} "
        f"elapsed={elapsed:6.1f}s ETA={eta:6.1f}s",
        end="",
        flush=True,
    )
    if done == total:
        print()


class _DryRunBackend:
    """Deterministic fake backend used to test cache and output contracts on CPU."""

    def infer(self, record: dict[str, object], *, full_generation: bool) -> dict[str, object]:
        mapping = record["candidate_mapping"]
        assert isinstance(mapping, list)
        identifiers = tuple(str(item["identifier"]) for item in mapping)
        logits = {
            identifier: float(len(identifiers) - index)
            for index, identifier in enumerate(identifiers)
        }
        ranking = rank_identifiers_from_logits(logits, identifiers)
        payload: dict[str, object] = {
            "status": "complete",
            "model_inference": False,
            "prefill_seconds": 0.0,
            "decoding_seconds": 0.0,
            "identifier_logits": logits,
            "first_token_ranking": list(ranking),
            "normalized_entropy": normalized_entropy(logits),
            "top1_top2_margin": top1_top2_margin(logits),
        }
        if full_generation:
            payload["generated_text"] = " > ".join(f"[{identifier}]" for identifier in ranking)
            payload["generated_ranking"] = list(
                parse_generated_ranking(str(payload["generated_text"]), identifiers)
            )
            payload["pair_agreement"] = pair_agreement(ranking, payload["generated_ranking"])
        return payload


class _TransformersBackend:
    def __init__(self, *, model_name: str, revision: str, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("R5.1 requires CUDA; use --dry-run for local validation")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            token=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device,
            token=False,
        )
        self.model.eval()

    def infer(self, record: dict[str, object], *, full_generation: bool) -> dict[str, object]:
        torch = self.torch
        mapping = record["candidate_mapping"]
        assert isinstance(mapping, list)
        identifiers = tuple(str(item["identifier"]) for item in mapping)
        token_ids = {
            identifier: int(self.tokenizer.encode(f"[{identifier}", add_special_tokens=False)[-1])
            for identifier in identifiers
        }
        first_prompt = str(record["first_token_prompt"])
        model_inputs = self.tokenizer(first_prompt, return_tensors="pt").to(self.model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**model_inputs, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - prefill_start
        next_logits = outputs.logits[0, -1]
        logits = {
            identifier: float(next_logits[token_id].float().cpu())
            for identifier, token_id in token_ids.items()
        }
        ranking = rank_identifiers_from_logits(logits, identifiers)
        payload: dict[str, object] = {
            "status": "complete",
            "model_inference": True,
            "prefill_seconds": prefill_seconds,
            "decoding_seconds": 0.0,
            "identifier_logits": logits,
            "first_token_ranking": list(ranking),
            "normalized_entropy": normalized_entropy(logits),
            "top1_top2_margin": top1_top2_margin(logits),
        }
        if full_generation:
            generation_inputs = self.tokenizer(
                str(record["prompt"]), return_tensors="pt"
            ).to(self.model.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_start = time.perf_counter()
            with torch.inference_mode():
                generated = self.model.generate(
                    **generation_inputs,
                    do_sample=False,
                    max_new_tokens=128,
                    min_new_tokens=1,
                    use_cache=True,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_generation_seconds = time.perf_counter() - generation_start
            generated_text = self.tokenizer.decode(
                generated[0, generation_inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            generated_ranking = parse_generated_ranking(generated_text, identifiers)
            payload["generated_text"] = generated_text
            payload["generated_ranking"] = list(generated_ranking)
            payload["pair_agreement"] = pair_agreement(ranking, generated_ranking)
            payload["decoding_seconds"] = max(0.0, total_generation_seconds - prefill_seconds)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-inputs",
        type=Path,
        default=Path("runs/r5_0_first_local_admission/prompt_inputs.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/r5_1_first_gpu_admission"))
    parser.add_argument("--model", default=FIRST_MODEL)
    parser.add_argument("--revision", default=FIRST_MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-limit", type=int, default=8)
    parser.add_argument("--variant", default="baseline", choices=("baseline", "all"))
    parser.add_argument("--full-generation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate cache/output flow without CUDA or model weights.",
    )
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.model != FIRST_MODEL or args.revision != FIRST_MODEL_REVISION:
        raise ValueError("R5.1 admission is locked to the registered FIRST checkpoint")
    if args.query_limit <= 0:
        raise ValueError("query-limit must be positive")
    records = _load_inputs(args.prompt_inputs)
    if args.variant == "baseline":
        records = [record for record in records if record["variant"] == "baseline"]
    records = records[: args.query_limit]
    if not records:
        raise ValueError("no frozen records selected")
    protocol_fingerprint = stable_sha256(
        {
            "model": args.model,
            "revision": args.revision,
            "input_sha256": sha256_file(args.prompt_inputs),
            "variant": args.variant,
            "full_generation": args.full_generation,
        }
    )
    cache = JsonlResultCache(
        args.output_dir / "results.jsonl",
        protocol_fingerprint=protocol_fingerprint,
    )
    backend: Any = _DryRunBackend() if args.dry_run else _TransformersBackend(
        model_name=args.model,
        revision=args.revision,
        device=args.device,
    )
    started = time.monotonic()
    cached_count = 0
    for index, record in enumerate(records, start=1):
        key = str(record["fingerprint"])
        if key in cache.records:
            cached_count += 1
            if args.progress:
                _progress(index, len(records), started, cached=True)
            continue
        payload = backend.infer(record, full_generation=args.full_generation)
        payload["query_id"] = record["query_id"]
        payload["slate_id"] = record["slate_id"]
        payload["variant"] = record["variant"]
        payload["prompt_sha256"] = record["prompt_sha256"]
        cache.append(key, payload)
        if args.progress:
            _progress(index, len(records), started)
    report = {
        "stage": "complete",
        "result_type": "FIRST R5.1 bounded inference admission",
        "model": args.model,
        "revision": args.revision,
        "dry_run": args.dry_run,
        "cuda_used": not args.dry_run,
        "training": False,
        "query_records": len(records),
        "cached_records": cached_count,
        "completed_records": len(cache.records),
        "full_generation": args.full_generation,
        "input_sha256": sha256_file(args.prompt_inputs),
        "protocol_fingerprint": protocol_fingerprint,
        "cache": str(args.output_dir / "results.jsonl"),
        "acceptance": {
            "model_identity_locked": True,
            "training_not_used": True,
            "all_selected_records_completed": all(
                str(record["fingerprint"]) in cache.records for record in records
            ),
            "cache_is_resumable": True,
            "full_generation_requested": args.full_generation,
            "cuda_used": not args.dry_run,
        },
    }
    # Full generation is an optional protocol branch in R5.2.  Its absence
    # must not make an otherwise complete first-token run fail admission.
    report["local_validation_pass"] = all(
        value
        for name, value in report["acceptance"].items()
        if name not in {"cuda_used", "full_generation_requested"}
    )
    report["gpu_admission_complete"] = bool(
        report["local_validation_pass"] and not args.dry_run
    )
    report["all_acceptance_pass"] = bool(
        report["local_validation_pass"] and (args.dry_run or report["cuda_used"])
    )
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    if not report["local_validation_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
