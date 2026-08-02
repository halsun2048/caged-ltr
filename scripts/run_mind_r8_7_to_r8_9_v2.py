"""Resume-safe R8.7 audit, fresh confirmation, admission, and one-shot R8.9."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable


def stage(number: int, total: int, label: str) -> None:
    width = 30
    filled = round(width * number / total)
    print(f"\n[R8V2 {number}/{total}] [{'#' * filled:<{width}}] {label}", flush=True)


def run(command: list[str]) -> None:
    print("[command] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def first(split: str, run_root: Path) -> None:
    report = run_root / f"{split}_first/report.json"
    if report.exists() and json.loads(report.read_text()).get("gpu_admission_complete"):
        print(f"[cached] FIRST {split}", flush=True)
        return
    run(
        [
            PYTHON,
            "scripts/run_first_r5_1_gpu_admission.py",
            "--prompt-inputs",
            str(run_root / f"{split}_prompts.jsonl"),
            "--output-dir",
            str(run_root / f"{split}_first"),
            "--query-limit",
            "60000",
            "--variant",
            "all",
            "--no-full-generation",
            "--progress",
        ]
    )


def student(split: str, pool_root: Path, run_root: Path, report: Path) -> None:
    metrics = run_root / f"{split}_query_metrics.parquet"
    if report.exists() and metrics.exists():
        print(f"[cached] MiniLM {split}", flush=True)
        return
    run(
        [
            PYTHON,
            "scripts/evaluate_mind_r8_large_dev.py",
            "--dev",
            str(pool_root / f"{split}.parquet"),
            "--expected-split",
            split,
            "--model",
            "/root/caged-ltr/all-MiniLM-L6-v2",
            "--checkpoint",
            "artifacts/mind_r8_2_hard_random.pt",
            "--name",
            split,
            "--output-dir",
            str(run_root),
            "--report",
            str(report),
            "--progress",
        ]
    )


def mark_materialized(guard_path: Path) -> None:
    guard = json.loads(guard_path.read_text())
    if guard["evaluation_count"] == 0:
        guard["status"] = "materialized_pending_evaluation"
        guard["labels_materialized"] = True
        guard_path.write_text(json.dumps(guard, indent=2) + "\n")


def prepare(
    split_ids: Path, split: str, pool_root: Path, run_root: Path, report: Path
) -> None:
    pool = pool_root / f"{split}.parquet"
    prompts = run_root / f"{split}_prompts.jsonl"
    if report.exists() and pool.exists() and prompts.exists():
        print(f"[cached] pool and prompts {split}", flush=True)
        return
    run(
        [
            PYTHON,
            "scripts/prepare_mind_r8_5d_gate.py",
            "--split-ids",
            str(split_ids),
            "--output-dir",
            str(pool_root),
            "--run-dir",
            str(run_root),
            "--report",
            str(report),
            "--splits",
            split,
            "--resume",
            "--progress",
        ]
    )


def main() -> None:
    source_root = str((Path.cwd() / "src").resolve())
    os.environ["PYTHONPATH"] = source_root + os.pathsep + str(Path.cwd() / "scripts")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    total = 9
    large_guard = Path("artifacts/mind_r8_0_large_test_guard.json")
    fresh_guard = Path("artifacts/mind_r8_8a_confirm_guard.json")
    stage(1, total, "R8.7 frozen gate audit")
    run([PYTHON, "scripts/audit_mind_r8_7_gate_freeze.py", "--progress"])
    stage(2, total, "R8.8a fresh-confirm preregistration audit")
    run([PYTHON, "scripts/preregister_mind_r8_8a_confirm.py", "--progress"])
    fresh_pool = Path("data/processed/mind_r8_8b")
    fresh_run = Path("runs/mind_r8_8b")
    stage(3, total, "fresh-confirm top-20 pool and frozen FIRST prompts")
    prepare(
        Path("data/processed/mind_r8_8a/fresh_confirm_ids.parquet"),
        "fresh_confirm",
        fresh_pool,
        fresh_run,
        Path("reports/data/mind_r8_8b_package.json"),
    )
    mark_materialized(fresh_guard)
    stage(4, total, "fresh-confirm FIRST inference with resume cache")
    first("fresh_confirm", fresh_run)
    stage(5, total, "fresh-confirm MiniLM features")
    student(
        "fresh_confirm",
        fresh_pool,
        fresh_run,
        Path("reports/experiments/mind_r8_8b_student.json"),
    )
    stage(6, total, "frozen-threshold fresh-confirm evaluation once")
    run([PYTHON, "scripts/evaluate_mind_r8_8b_confirm_once.py", "--progress"])
    fresh_result = json.loads(
        Path("reports/experiments/mind_r8_8b_fresh_confirm.json").read_text()
    )
    admitted = fresh_result["acceptance"]["all_passed"]
    decision = {
        "schema": "mind_r8_8c_admission_v1",
        "admitted": admitted,
        "fresh_confirm_evaluation_count": 1,
        "large_test_evaluation_count_before_decision": json.loads(large_guard.read_text())[
            "evaluation_count"
        ],
        "action": "execute_r8_9_once" if admitted else "preserve_locked_large_test",
    }
    decision_path = Path("reports/experiments/mind_r8_8c_admission.json")
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    stage(7, total, "R8.8c admission decision")
    print(json.dumps(decision), flush=True)
    if not admitted:
        stage(8, total, "R8.9 remains locked")
        stage(9, total, "pipeline complete without test access")
        return
    large_pool = Path("data/processed/mind_r8_9_v2")
    large_run = Path("runs/mind_r8_9_v2")
    stage(8, total, "materialize and infer untouched large-test once")
    prepare(
        Path("data/processed/mind_r8_0/large_split_ids.parquet"),
        "large_test",
        large_pool,
        large_run,
        Path("reports/data/mind_r8_9_v2_package.json"),
    )
    mark_materialized(large_guard)
    first("large_test", large_run)
    student(
        "large_test",
        large_pool,
        large_run,
        Path("reports/experiments/mind_r8_9_v2_student.json"),
    )
    stage(9, total, "apply frozen gate and permanently close large-test")
    run(
        [
            PYTHON,
            "scripts/evaluate_mind_r8_8b_confirm_once.py",
            "--split",
            "large_test",
            "--metrics-root",
            str(large_run),
            "--pool-root",
            str(large_pool),
            "--first-root",
            str(large_run),
            "--guard",
            str(large_guard),
            "--output",
            "reports/experiments/mind_r8_9_v2_final_once.json",
            "--progress",
        ]
    )


if __name__ == "__main__":
    main()
