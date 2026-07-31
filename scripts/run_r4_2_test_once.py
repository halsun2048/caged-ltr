"""Run qrels-free R4 scoring, freeze predictions, then evaluate exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoTokenizer

from caged_ltr.evaluation.r4_test_once import (
    R4_CONTROLS,
    R4_PREDICTION_COLUMNS,
    evaluate_r4_test_once,
    merge_r4_prediction_shards,
)
from caged_ltr.models import (
    DEFAULT_DEBERTA_V3_BASE,
    DEFAULT_DEBERTA_V3_BASE_REVISION,
    PointwiseCrossEncoder,
    tokenize_query_passages,
)
from caged_ltr.reproducibility import seed_everything, sha256_file
from caged_ltr.teachers.prp_real import TRECInputQuery, load_teacher_inputs


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _identity_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    return f"{minutes:02d}m{seconds:02d}s"


def _autocast(device: torch.device, precision: str) -> torch.autocast:
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    enabled = device.type == "cuda" and precision in dtypes
    return torch.autocast(
        device_type=device.type,
        dtype=dtypes.get(precision, torch.float32),
        enabled=enabled,
    )


def _checkpoint_sha256(control: str, checkpoint: Path | None) -> str | None:
    if control == "vanilla":
        if checkpoint is not None:
            raise ValueError("vanilla control must not receive a trained checkpoint")
        return None
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(f"{control} checkpoint is required: {checkpoint}")
    return sha256_file(checkpoint)


def _score_identity(
    args: argparse.Namespace,
    *,
    queries: list[TRECInputQuery],
    checkpoint_sha256: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "r4_test_once_shard_v1",
        "control": args.control,
        "teacher_inputs_sha256": sha256_file(args.teacher_inputs),
        "checkpoint_sha256": checkpoint_sha256,
        "vanilla_initialization": (
            {
                "model": args.model,
                "revision": args.revision,
                "seed": args.seed,
                "trained": False,
            }
            if args.control == "vanilla"
            else None
        ),
        "model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "request_ids": [query.request_id for query in queries],
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "precision": args.precision,
    }
    return {**payload, "identity_sha256": _identity_sha256(payload)}


def _load_student(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> PointwiseCrossEncoder:
    seed_everything(args.seed, deterministic_algorithms=False)
    model = PointwiseCrossEncoder.from_pretrained(
        model_name=args.model,
        revision=args.revision,
        cache_dir=str(args.cache_dir),
    )
    if args.control != "vanilla":
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if "model" not in checkpoint:
            raise ValueError("R4 checkpoint does not contain model state")
        model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def _score_query(
    query: TRECInputQuery,
    *,
    tokenizer: Any,
    model: PointwiseCrossEncoder,
    device: torch.device,
    precision: str,
    batch_size: int,
    max_length: int,
) -> tuple[list[float], list[float], float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    scores: list[torch.Tensor] = []
    for start in range(0, len(query.candidates), batch_size):
        candidates = query.candidates[start : start + batch_size]
        encoded = tokenize_query_passages(
            tokenizer,
            [query.query] * len(candidates),
            [candidate.passage for candidate in candidates],
            max_length=max_length,
        )
        encoded = {
            name: tensor.to(device, non_blocking=True)
            for name, tensor in encoded.items()
        }
        with _autocast(device, precision):
            batch_scores = model(**encoded)
        scores.append(batch_scores.float().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    raw = torch.cat(scores)
    return raw.tolist(), torch.sigmoid(raw).tolist(), elapsed


def _score(args: argparse.Namespace) -> None:
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid R4 shard index/count")
    if min(args.batch_size, args.max_length) <= 0 or args.seed < 0:
        raise ValueError("invalid R4 batch, length, or seed")
    all_queries = load_teacher_inputs(args.teacher_inputs)
    queries = all_queries[args.shard_index :: args.num_shards]
    if not queries or any(query.year is None for query in queries):
        raise ValueError("R4 test shards require non-empty, year-tagged queries")
    checkpoint_sha = _checkpoint_sha256(args.control, args.checkpoint)
    identity = _score_identity(
        args,
        queries=queries,
        checkpoint_sha256=checkpoint_sha,
    )
    shard_dir = args.output_root / "shards" / args.control / f"shard_{args.shard_index:02d}"
    prediction_path = shard_dir / "predictions.parquet"
    manifest_path = shard_dir / "manifest.json"
    progress_path = shard_dir / "progress.json"
    if manifest_path.is_file() and prediction_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("identity_sha256") == identity["identity_sha256"]
            and manifest.get("predictions_sha256") == sha256_file(prediction_path)
            and manifest.get("qrels_accessed") is False
        ):
            print(
                json.dumps(
                    {
                        "stage": "cached",
                        "control": args.control,
                        "shard": args.shard_index,
                        "report": str(manifest_path),
                    }
                ),
                flush=True,
            )
            return
        raise ValueError("existing R4 test shard has a different or corrupt identity")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"[model] {args.control} shard={args.shard_index}/{args.num_shards} "
        f"device={device} queries={len(queries)}",
        file=sys.stderr,
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    model = _load_student(args, device=device)
    first = queries[0]
    warmup = tokenize_query_passages(
        tokenizer,
        [first.query],
        [first.candidates[0].passage],
        max_length=args.max_length,
    )
    warmup = {name: tensor.to(device) for name, tensor in warmup.items()}
    with torch.inference_mode(), _autocast(device, args.precision):
        model(**warmup)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    started = time.monotonic()
    rows: list[dict[str, object]] = []
    for index, query in enumerate(queries, start=1):
        raw_scores, probabilities, elapsed = _score_query(
            query,
            tokenizer=tokenizer,
            model=model,
            device=device,
            precision=args.precision,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        for candidate, raw_score, probability in zip(
            query.candidates,
            raw_scores,
            probabilities,
            strict=True,
        ):
            rows.append(
                {
                    "control": args.control,
                    "request_id": query.request_id,
                    "year": int(query.year),
                    "query_id": query.query_id,
                    "passage_id": candidate.passage_id,
                    "bm25_rank": candidate.bm25_rank,
                    "raw_score": float(raw_score),
                    "probability": float(probability),
                    "query_elapsed_seconds": elapsed,
                }
            )
        _write_json(
            progress_path,
            {
                "stage": "scoring",
                "control": args.control,
                "shard": args.shard_index,
                "completed_queries": index,
                "total_queries": len(queries),
                "elapsed_seconds": time.monotonic() - started,
                "qrels_accessed": False,
            },
        )
        width = 24
        filled = round(width * index / len(queries))
        sys.stderr.write("\r\033[2K")
        sys.stderr.write(
            f"[{args.control}:{args.shard_index}] "
            f"[{'#' * filled}{'-' * (width - filled)}] "
            f"queries={index}/{len(queries)} elapsed={_duration(time.monotonic() - started)}"
        )
        sys.stderr.flush()
    sys.stderr.write("\n")

    predictions = pd.DataFrame(rows, columns=R4_PREDICTION_COLUMNS)
    _write_parquet(prediction_path, predictions)
    wall_seconds = time.monotonic() - started
    manifest = {
        "stage": "complete",
        **identity,
        "queries": len(queries),
        "candidates": len(predictions),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "wall_seconds": wall_seconds,
        "mean_seconds_per_query": float(
            predictions[["request_id", "query_elapsed_seconds"]]
            .drop_duplicates()["query_elapsed_seconds"]
            .mean()
        ),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "prediction_columns": list(predictions.columns),
        "qrels_accessed": False,
        "test_labels_accessed": False,
    }
    _write_json(manifest_path, manifest)
    _write_json(
        progress_path,
        {
            "stage": "complete",
            "control": args.control,
            "shard": args.shard_index,
            "completed_queries": len(queries),
            "total_queries": len(queries),
            "elapsed_seconds": wall_seconds,
            "qrels_accessed": False,
        },
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "control": args.control,
                "shard": args.shard_index,
                "queries": len(queries),
                "qrels_accessed": False,
                "report": str(manifest_path),
            }
        ),
        flush=True,
    )


def _launch(args: argparse.Namespace) -> None:
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    expected_workers = len(R4_CONTROLS) * args.num_shards
    if len(gpus) != expected_workers or len(set(gpus)) != len(gpus):
        raise ValueError(
            f"launch requires {expected_workers} distinct GPUs, got {gpus}"
        )
    checkpoint_by_control = {
        "vanilla": None,
        "bm25": args.bm25_checkpoint,
        "random": args.random_checkpoint,
        "prp": args.prp_checkpoint,
    }
    jobs: list[dict[str, object]] = []
    worker = 0
    started = time.monotonic()
    for control in R4_CONTROLS:
        for shard_index in range(args.num_shards):
            gpu = gpus[worker]
            worker += 1
            log_path = (
                args.output_root
                / "logs"
                / f"{control}_shard_{shard_index:02d}_gpu_{gpu}.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "score",
                "--control",
                control,
                "--teacher-inputs",
                str(args.teacher_inputs),
                "--output-root",
                str(args.output_root),
                "--shard-index",
                str(shard_index),
                "--num-shards",
                str(args.num_shards),
                "--device",
                f"cuda:{gpu}",
                "--precision",
                args.precision,
                "--batch-size",
                str(args.batch_size),
                "--max-length",
                str(args.max_length),
                "--seed",
                str(args.seed),
                "--model",
                args.model,
                "--revision",
                args.revision,
                "--cache-dir",
                str(args.cache_dir),
            ]
            checkpoint = checkpoint_by_control[control]
            if checkpoint is not None:
                command.extend(["--checkpoint", str(checkpoint)])
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            jobs.append(
                {
                    "control": control,
                    "shard": shard_index,
                    "gpu": gpu,
                    "process": process,
                    "log_handle": log_handle,
                    "log": log_path,
                }
            )
    try:
        while any(job["process"].poll() is None for job in jobs):
            parts = []
            for job in jobs:
                progress_path = (
                    args.output_root
                    / "shards"
                    / str(job["control"])
                    / f"shard_{int(job['shard']):02d}"
                    / "progress.json"
                )
                completed = 0
                total = "?"
                if progress_path.is_file():
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    completed = int(progress.get("completed_queries", 0))
                    total = str(progress.get("total_queries", "?"))
                status = (
                    "done"
                    if job["process"].poll() == 0
                    else "run"
                    if job["process"].poll() is None
                    else "FAIL"
                )
                parts.append(
                    f"{job['control']}:{job['shard']}={completed}/{total}({status})"
                )
            sys.stderr.write("\r\033[2K")
            sys.stderr.write(
                f"[8-GPU {_duration(time.monotonic() - started)}] " + " | ".join(parts)
            )
            sys.stderr.flush()
            time.sleep(args.progress_interval)
    except KeyboardInterrupt:
        for job in jobs:
            process = job["process"]
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        sys.stderr.write("\n")
        for job in jobs:
            job["log_handle"].close()
    failures = [
        {
            "control": job["control"],
            "shard": job["shard"],
            "gpu": job["gpu"],
            "exit_code": job["process"].returncode,
            "log": str(job["log"]),
        }
        for job in jobs
        if job["process"].returncode != 0
    ]
    summary = {
        "stage": "complete" if not failures else "failed",
        "workers": expected_workers,
        "gpus": gpus,
        "num_shards_per_control": args.num_shards,
        "wall_seconds": time.monotonic() - started,
        "failures": failures,
        "qrels_accessed": False,
    }
    _write_json(args.output_root / "launch_summary.json", summary)
    if failures:
        raise RuntimeError(f"R4 scoring workers failed: {failures}")
    print(json.dumps(summary), flush=True)


def _merge(args: argparse.Namespace) -> None:
    queries = load_teacher_inputs(args.teacher_inputs)
    expected_request_ids = {query.request_id for query in queries}
    source_sha = sha256_file(args.teacher_inputs)
    frozen_dir = args.output_root / "frozen"
    artifacts: dict[str, object] = {}
    for control in R4_CONTROLS:
        frames = []
        shard_manifests = []
        for shard_index in range(args.num_shards):
            shard_dir = (
                args.output_root / "shards" / control / f"shard_{shard_index:02d}"
            )
            manifest_path = shard_dir / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"missing R4 shard manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prediction_path = Path(str(manifest["predictions"]))
            if (
                manifest.get("stage") != "complete"
                or manifest.get("control") != control
                or manifest.get("teacher_inputs_sha256") != source_sha
                or manifest.get("qrels_accessed") is not False
                or manifest.get("test_labels_accessed") is not False
                or manifest.get("predictions_sha256") != sha256_file(prediction_path)
            ):
                raise ValueError(f"invalid or unsafe R4 shard: {manifest_path}")
            frames.append(pd.read_parquet(prediction_path))
            shard_manifests.append(
                {
                    "path": str(manifest_path),
                    "identity_sha256": manifest["identity_sha256"],
                    "predictions_sha256": manifest["predictions_sha256"],
                }
            )
        merged = merge_r4_prediction_shards(
            frames,
            control=control,
            expected_request_ids=expected_request_ids,
        )
        output_path = frozen_dir / f"{control}.parquet"
        _write_parquet(output_path, merged)
        artifacts[control] = {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "rows": len(merged),
            "queries": merged["request_id"].nunique(),
            "shards": shard_manifests,
        }
    manifest_payload: dict[str, object] = {
        "schema": "r4_test_once_frozen_predictions_v1",
        "stage": "predictions_frozen",
        "teacher_inputs": str(args.teacher_inputs),
        "teacher_inputs_sha256": source_sha,
        "controls": artifacts,
        "query_count": len(expected_request_ids),
        "candidate_count_per_control": len(queries) * 100,
        "prediction_columns": list(R4_PREDICTION_COLUMNS),
        "qrels_accessed": False,
        "test_labels_accessed": False,
    }
    manifest_payload["identity_sha256"] = _identity_sha256(manifest_payload)
    manifest_path = frozen_dir / "manifest.json"
    _write_json(manifest_path, manifest_payload)
    print(
        json.dumps(
            {
                "stage": "predictions_frozen",
                "queries": len(expected_request_ids),
                "controls": len(artifacts),
                "qrels_accessed": False,
                "report": str(manifest_path),
            }
        ),
        flush=True,
    )


def _load_frozen_predictions(
    manifest_path: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("stage") != "predictions_frozen"
        or manifest.get("qrels_accessed") is not False
        or manifest.get("test_labels_accessed") is not False
        or set(manifest.get("controls", {})) != set(R4_CONTROLS)
    ):
        raise ValueError("R4 frozen prediction manifest is incomplete or unsafe")
    predictions: dict[str, pd.DataFrame] = {}
    for control in R4_CONTROLS:
        artifact = manifest["controls"][control]
        path = Path(str(artifact["path"]))
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"frozen R4 prediction hash mismatch: {control}")
        predictions[control] = pd.read_parquet(path)
    return predictions, manifest


def _evaluate(args: argparse.Namespace) -> None:
    report_path = args.report
    receipt_path = report_path.with_name(report_path.stem + "_access_receipt.json")
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("stage") == "complete"
            and report_path.is_file()
            and receipt.get("report_sha256") == sha256_file(report_path)
        ):
            print(
                json.dumps(
                    {
                        "stage": "cached_complete",
                        "qrels_reopened": False,
                        "report": str(report_path),
                    }
                )
            )
            return
        raise RuntimeError("R4 test access already started but did not complete cleanly")
    if not args.allow_test_access:
        raise PermissionError(
            "refusing to read qrels without explicit --allow-test-access"
        )
    predictions, frozen_manifest = _load_frozen_predictions(args.frozen_manifest)
    frozen_manifest_sha = sha256_file(args.frozen_manifest)
    _write_json(
        receipt_path,
        {
            "stage": "authorized",
            "authorized_at_utc": datetime.now(UTC).isoformat(),
            "frozen_manifest_sha256": frozen_manifest_sha,
            "qrels_access_count": 0,
        },
    )
    qrels = pd.read_parquet(args.qrels)
    qrels_sha = sha256_file(args.qrels)
    metrics = evaluate_r4_test_once(
        predictions,
        qrels,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    teacher_report = json.loads(args.teacher_report.read_text(encoding="utf-8"))
    teacher = teacher_report["evaluation"]
    controls = metrics["controls"]
    prp_overall = controls["prp"]["overall"]["trec_eval_ndcg_at_10"]
    bm25_initial = controls["bm25_initial"]["overall"]["trec_eval_ndcg_at_10"]
    teacher_overall = float(teacher["overall"]["teacher_ndcg_at_10"])
    denominator = teacher_overall - bm25_initial
    retained_gain = (
        (prp_overall - bm25_initial) / denominator if denominator != 0 else None
    )
    prp_latency = controls["prp"]["efficiency"]["mean_seconds_per_query"]
    teacher_mean_seconds = args.teacher_mean_single_gpu_seconds_per_query
    acceptance = {
        "predictions_frozen_before_qrels": True,
        "qrels_accessed_exactly_once": True,
        "prp_beats_vanilla_overall": (
            prp_overall
            > controls["vanilla"]["overall"]["trec_eval_ndcg_at_10"]
        ),
        "prp_beats_random_overall": (
            prp_overall
            > controls["random"]["overall"]["trec_eval_ndcg_at_10"]
        ),
        "prp_beats_bm25_ranknet_overall": (
            prp_overall > controls["bm25"]["overall"]["trec_eval_ndcg_at_10"]
        ),
        "prp_direction_positive_vs_all_controls_both_years": all(
            controls["prp"]["by_year"][year]["trec_eval_ndcg_at_10"]
            > controls[control]["by_year"][year]["trec_eval_ndcg_at_10"]
            for year in ("2019", "2020")
            for control in ("vanilla", "bm25", "random")
        ),
        "prp_at_least_10x_faster_than_allpair_single_gpu_reference": (
            teacher_mean_seconds / prp_latency >= 10.0
        ),
    }
    launch_summary_path = args.frozen_manifest.parent.parent / "launch_summary.json"
    launch_summary = (
        json.loads(launch_summary_path.read_text(encoding="utf-8"))
        if launch_summary_path.is_file()
        else None
    )
    report = {
        "stage": "complete",
        "result_type": "R4.2 locked TREC-DL19/20 test-once evaluation",
        "frozen_predictions": {
            "manifest": str(args.frozen_manifest),
            "manifest_sha256": frozen_manifest_sha,
            "identity_sha256": frozen_manifest["identity_sha256"],
        },
        "test_access": {
            "qrels": str(args.qrels),
            "qrels_sha256": qrels_sha,
            "qrels_access_count": 1,
            "predictions_frozen_before_access": True,
            "no_test_set_model_or_hyperparameter_selection": True,
        },
        "metrics": metrics,
        "teacher_reference": {
            "report": str(args.teacher_report),
            "overall_trec_eval_ndcg_at_10": teacher_overall,
            "dl2019_trec_eval_ndcg_at_10": teacher["dl2019"][
                "teacher_ndcg_at_10"
            ],
            "dl2020_trec_eval_ndcg_at_10": teacher["dl2020"][
                "teacher_ndcg_at_10"
            ],
            "mean_single_gpu_seconds_per_query": teacher_mean_seconds,
            "timing_note": (
                "observed mean over 91 non-admission R3.1c Allpair worker queries"
            ),
        },
        "comparison": {
            "prp_student_teacher_gain_retention": retained_gain,
            "prp_student_vs_teacher_absolute_ndcg": prp_overall - teacher_overall,
            "prp_student_single_gpu_speedup_vs_allpair_reference": (
                teacher_mean_seconds / prp_latency
            ),
            "logical_scoring_calls_per_query": {
                "pointwise_student": 100,
                "bidirectional_allpair_teacher": 9900,
                "reduction_factor": 99.0,
                "complexity": "O(N^2) to O(N)",
            },
        },
        "eight_gpu_launch": launch_summary,
        "acceptance": acceptance,
        "qrels_accessed": True,
        "test_accessed": True,
    }
    _write_json(report_path, report)
    _write_json(
        receipt_path,
        {
            "stage": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "frozen_manifest_sha256": frozen_manifest_sha,
            "qrels_sha256": qrels_sha,
            "qrels_access_count": 1,
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
        },
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "qrels_access_count": 1,
                "acceptance": acceptance,
                "report": str(report_path),
            }
        ),
        flush=True,
    )


def _add_common_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_DEBERTA_V3_BASE)
    parser.add_argument("--revision", default=DEFAULT_DEBERTA_V3_BASE_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf-cache"))
    parser.add_argument(
        "--precision",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="score one qrels-free shard")
    score_parser.add_argument("--control", choices=R4_CONTROLS, required=True)
    score_parser.add_argument("--checkpoint", type=Path)
    score_parser.add_argument("--teacher-inputs", type=Path, required=True)
    score_parser.add_argument("--output-root", type=Path, required=True)
    score_parser.add_argument("--shard-index", type=int, required=True)
    score_parser.add_argument("--num-shards", type=int, required=True)
    score_parser.add_argument("--device", default="cuda")
    _add_common_model_arguments(score_parser)
    score_parser.set_defaults(func=_score)

    launch_parser = subparsers.add_parser("launch", help="launch all eight GPU shards")
    launch_parser.add_argument("--teacher-inputs", type=Path, required=True)
    launch_parser.add_argument("--output-root", type=Path, required=True)
    launch_parser.add_argument("--bm25-checkpoint", type=Path, required=True)
    launch_parser.add_argument("--random-checkpoint", type=Path, required=True)
    launch_parser.add_argument("--prp-checkpoint", type=Path, required=True)
    launch_parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    launch_parser.add_argument("--num-shards", type=int, default=2)
    launch_parser.add_argument("--progress-interval", type=float, default=2.0)
    _add_common_model_arguments(launch_parser)
    launch_parser.set_defaults(func=_launch)

    merge_parser = subparsers.add_parser("merge", help="freeze qrels-free predictions")
    merge_parser.add_argument("--teacher-inputs", type=Path, required=True)
    merge_parser.add_argument("--output-root", type=Path, required=True)
    merge_parser.add_argument("--num-shards", type=int, default=2)
    merge_parser.set_defaults(func=_merge)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="open qrels once after predictions are frozen",
    )
    evaluate_parser.add_argument("--frozen-manifest", type=Path, required=True)
    evaluate_parser.add_argument("--qrels", type=Path, required=True)
    evaluate_parser.add_argument("--teacher-report", type=Path, required=True)
    evaluate_parser.add_argument("--report", type=Path, required=True)
    evaluate_parser.add_argument("--allow-test-access", action="store_true")
    evaluate_parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument(
        "--teacher-mean-single-gpu-seconds-per-query",
        type=float,
        default=165.07049662510323,
    )
    evaluate_parser.set_defaults(func=_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
