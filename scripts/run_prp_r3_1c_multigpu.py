"""Prepare, launch, merge, and finalize disjoint PRP inference shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

from caged_ltr.teachers.prp import TeacherMetadata
from caged_ltr.teachers.prp_real import run_prp_r3_1b


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _cache_path(cache_dir: Path, request_id: str) -> Path:
    digest = hashlib.sha256(request_id.encode()).hexdigest()
    return cache_dir / f"{digest}.jsonl"


def _load_records(path: Path, request_id: str) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid cache line {path}:{line_number}") from error
            key = str(record.get("key", ""))
            if not key or key in records:
                raise ValueError(f"duplicate or empty cache key in {path}")
            if str(record.get("request_id", "")) != request_id:
                raise ValueError(f"mismatched request ID in {path}")
            records[key] = record
    return records


def _teacher_rows(path: Path) -> list[tuple[str, dict[str, object]]]:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            payload = json.loads(line)
            rows.append((line.rstrip("\n"), payload))
    if not rows:
        raise ValueError("teacher input is empty")
    return rows


def _same_file(left: Path, right: Path) -> bool:
    return left.read_bytes() == right.read_bytes()


def prepare(args: argparse.Namespace) -> None:
    teacher_input = args.teacher_input.resolve()
    source_output = args.source_output.resolve()
    work_dir = args.work_dir.resolve()
    source_cache = source_output / "ordered_pair_responses"
    rows = _teacher_rows(teacher_input)
    tasks = []
    completed_prompts = 0
    expected_prompts = 0
    for index, (line, payload) in enumerate(rows):
        request_id = str(payload["request_id"])
        candidates = payload["candidates"]
        expected = len(candidates) * (len(candidates) - 1)
        cached = len(_load_records(_cache_path(source_cache, request_id), request_id))
        if cached > expected:
            raise ValueError(f"too many cached prompts for {request_id}")
        expected_prompts += expected
        completed_prompts += cached
        if cached < expected:
            tasks.append(
                {
                    "index": index,
                    "line": line,
                    "request_id": request_id,
                    "expected_prompts": expected,
                    "cached_prompts": cached,
                    "remaining_prompts": expected - cached,
                }
            )

    shard_tasks: list[list[dict[str, object]]] = [[] for _ in range(args.shards)]
    shard_loads = [0] * args.shards
    for task in sorted(
        tasks,
        key=lambda row: (-int(row["remaining_prompts"]), int(row["index"])),
    ):
        shard_index = min(range(args.shards), key=lambda index: shard_loads[index])
        shard_tasks[shard_index].append(task)
        shard_loads[shard_index] += int(task["remaining_prompts"])

    shards = []
    for shard_index, assigned in enumerate(shard_tasks):
        assigned.sort(key=lambda row: int(row["index"]))
        shard_dir = work_dir / "shards" / f"{shard_index:02d}"
        shard_input = shard_dir / "teacher_inputs.jsonl"
        output_dir = shard_dir / "output"
        output_cache = output_dir / "ordered_pair_responses"
        output_cache.mkdir(parents=True, exist_ok=True)
        content = "".join(f"{task['line']}\n" for task in assigned)
        if shard_input.is_file():
            if shard_input.read_text(encoding="utf-8") != content:
                raise ValueError(f"existing shard input differs: {shard_input}")
        else:
            shard_input.parent.mkdir(parents=True, exist_ok=True)
            shard_input.write_text(content, encoding="utf-8")
        for task in assigned:
            request_id = str(task["request_id"])
            source = _cache_path(source_cache, request_id)
            target = _cache_path(output_cache, request_id)
            if not source.is_file():
                continue
            if target.is_file() and not _same_file(source, target):
                target_records = _load_records(target, request_id)
                source_records = _load_records(source, request_id)
                if not set(source_records).issubset(target_records):
                    raise ValueError(f"existing shard cache conflicts: {target}")
            elif not target.is_file():
                shutil.copy2(source, target)
        shards.append(
            {
                "gpu": shard_index,
                "input": str(shard_input),
                "output": str(output_dir),
                "query_count": len(assigned),
                "request_ids": [task["request_id"] for task in assigned],
                "expected_prompts": sum(
                    int(task["expected_prompts"]) for task in assigned
                ),
                "cached_prompts": sum(
                    int(task["cached_prompts"]) for task in assigned
                ),
                "remaining_prompts": shard_loads[shard_index],
            }
        )

    plan = {
        "schema": "prp_r3_1c_multigpu_v1",
        "teacher_input": str(teacher_input),
        "qrels": str(args.qrels.resolve()),
        "source_output": str(source_output),
        "work_dir": str(work_dir),
        "shard_count": args.shards,
        "expected_prompts": expected_prompts,
        "completed_prompts_at_prepare": completed_prompts,
        "remaining_prompts_at_prepare": expected_prompts - completed_prompts,
        "shards": shards,
    }
    _write_json(work_dir / "plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False))


def _load_plan(work_dir: Path) -> dict[str, object]:
    return json.loads((work_dir.resolve() / "plan.json").read_text(encoding="utf-8"))


def launch(args: argparse.Namespace) -> None:
    work_dir = args.work_dir.resolve()
    plan = _load_plan(work_dir)
    processes = []
    launch_rows = []
    for shard in plan["shards"]:
        if int(shard["query_count"]) == 0:
            continue
        output_dir = Path(str(shard["output"]))
        summary_path = output_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("stage") == "inference_complete":
                continue
        command = [
            sys.executable,
            str(args.runner.resolve()),
            "--teacher-input",
            str(shard["input"]),
            "--qrels",
            str(work_dir / "QRELS_MUST_NOT_BE_READ.parquet"),
            "--queries",
            str(shard["query_count"]),
            "--batch-size",
            str(args.batch_size),
            "--batch-order",
            "length",
            "--max-ordered-prompts",
            "0",
            "--scoring-mode",
            "likelihood",
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(args.cache_dir.resolve()),
            "--defer-evaluation",
            "--progress",
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(shard["gpu"]),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONPATH": str(args.source_dir.resolve()),
                "PYTHONUNBUFFERED": "1",
            }
        )
        log_path = work_dir / f"gpu_{int(shard['gpu']):02d}.log"
        log_file = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((process, log_file, shard, log_path))
        launch_rows.append(
            {
                "gpu": shard["gpu"],
                "pid": process.pid,
                "log": str(log_path),
                "remaining_prompts": shard["remaining_prompts"],
            }
        )
    _write_json(work_dir / "launch.json", {"workers": launch_rows})
    print(json.dumps({"stage": "launched", "workers": launch_rows}, ensure_ascii=False))
    failures = []
    for process, log_file, shard, log_path in processes:
        return_code = process.wait()
        log_file.close()
        if return_code:
            failures.append(
                {
                    "gpu": shard["gpu"],
                    "return_code": return_code,
                    "log": str(log_path),
                }
            )
    result = {"stage": "workers_finished", "failures": failures}
    _write_json(work_dir / "workers_finished.json", result)
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


def status(args: argparse.Namespace) -> None:
    plan = _load_plan(args.work_dir)
    rows = []
    total_cached = 0
    total_expected = 0
    for shard in plan["shards"]:
        cached = 0
        for request_id in shard["request_ids"]:
            cached += len(
                _load_records(
                    _cache_path(
                        Path(str(shard["output"])) / "ordered_pair_responses",
                        str(request_id),
                    ),
                    str(request_id),
                )
            )
        expected = int(shard["expected_prompts"])
        total_cached += cached
        total_expected += expected
        rows.append(
            {
                "gpu": shard["gpu"],
                "cached": cached,
                "expected": expected,
                "complete": cached == expected,
            }
        )
    print(
        json.dumps(
            {
                "shards": rows,
                "cached": total_cached,
                "expected": total_expected,
            },
            ensure_ascii=False,
        )
    )


def merge(args: argparse.Namespace) -> None:
    work_dir = args.work_dir.resolve()
    plan = _load_plan(work_dir)
    destination_cache = (
        Path(str(plan["source_output"])) / "ordered_pair_responses"
    )
    destination_cache.mkdir(parents=True, exist_ok=True)
    for shard in plan["shards"]:
        if not shard["request_ids"]:
            continue
        summary = json.loads(
            (Path(str(shard["output"])) / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        if summary.get("stage") != "inference_complete":
            raise ValueError(f"GPU shard {shard['gpu']} is not inference-complete")
        if summary.get("qrels_accessed") is not False:
            raise ValueError(f"GPU shard {shard['gpu']} accessed qrels")
        source_cache = Path(str(shard["output"])) / "ordered_pair_responses"
        for request_id_value in shard["request_ids"]:
            request_id = str(request_id_value)
            source = _cache_path(source_cache, request_id)
            destination = _cache_path(destination_cache, request_id)
            source_records = _load_records(source, request_id)
            destination_records = _load_records(destination, request_id)
            if not set(destination_records).issubset(source_records):
                raise ValueError(f"shard cache conflicts with main cache: {request_id}")
            shutil.copy2(source, destination)

    total = 0
    unique_keys: set[str] = set()
    for _, payload in _teacher_rows(Path(str(plan["teacher_input"]))):
        request_id = str(payload["request_id"])
        expected = len(payload["candidates"]) * (len(payload["candidates"]) - 1)
        records = _load_records(_cache_path(destination_cache, request_id), request_id)
        if len(records) != expected:
            raise ValueError(f"incomplete merged cache for {request_id}")
        total += len(records)
        unique_keys.update(records)
    if total != int(plan["expected_prompts"]) or len(unique_keys) != total:
        raise ValueError("merged cache total or global key uniqueness is invalid")
    result = {
        "stage": "merged",
        "cached_ordered_prompts": total,
        "unique_keys": len(unique_keys),
        "qrels_accessed": False,
    }
    _write_json(work_dir / "merge_summary.json", result)
    print(json.dumps(result, ensure_ascii=False))


class _CachedTeacher:
    def __init__(self, metadata: TeacherMetadata) -> None:
        self._metadata = metadata

    @property
    def metadata(self) -> TeacherMetadata:
        return self._metadata

    def compare_many(self, requests: object) -> NoReturn:
        del requests
        raise RuntimeError("finalization attempted uncached teacher inference")


def finalize(args: argparse.Namespace) -> None:
    plan = _load_plan(args.work_dir)
    output_dir = Path(str(plan["source_output"]))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    teacher = _CachedTeacher(TeacherMetadata(**manifest["teacher"]))
    summary = run_prp_r3_1b(
        teacher,
        teacher_input_path=Path(str(plan["teacher_input"])),
        output_dir=output_dir,
        qrels_path=Path(str(plan["qrels"])),
        query_limit=len(_teacher_rows(Path(str(plan["teacher_input"])))),
        batch_size=8,
    )
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "cached_ordered_prompts": summary["cached_ordered_prompts"],
                "qrels_accessed": summary["qrels_accessed"],
                "evaluation": summary["evaluation"],
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--teacher-input", type=Path, required=True)
    prepare_parser.add_argument("--qrels", type=Path, required=True)
    prepare_parser.add_argument("--source-output", type=Path, required=True)
    prepare_parser.add_argument("--work-dir", type=Path, required=True)
    prepare_parser.add_argument("--shards", type=int, default=8)
    prepare_parser.set_defaults(handler=prepare)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--work-dir", type=Path, required=True)
    launch_parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_prp_r3_1b_flan_t5.py"),
    )
    launch_parser.add_argument("--source-dir", type=Path, default=Path("src"))
    launch_parser.add_argument("--cache-dir", type=Path, default=Path(".hf-cache"))
    launch_parser.add_argument("--batch-size", type=int, default=8)
    launch_parser.set_defaults(handler=launch)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--work-dir", type=Path, required=True)
    status_parser.set_defaults(handler=status)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--work-dir", type=Path, required=True)
    merge_parser.set_defaults(handler=merge)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--work-dir", type=Path, required=True)
    finalize_parser.set_defaults(handler=finalize)

    args = parser.parse_args()
    if getattr(args, "shards", 1) <= 0:
        parser.error("--shards must be positive")
    if getattr(args, "batch_size", 1) <= 0:
        parser.error("--batch-size must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
