"""Resumable GPU pipeline for R8.5b through the guarded R8.9 decision."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
FIRST_RECORDS = 60_000


def stage(number: int, label: str) -> None:
    width = 30
    filled = round(width * number / 8)
    print(f"\n[R8G {number}/8] [{'#' * filled:<{width}}] {label}", flush=True)


def run(command: list[str]) -> None:
    print("[command] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def gate_package_ready() -> bool:
    report = Path("reports/data/mind_r8_5d_gate_package.json")
    required = [
        Path("data/processed/mind_r8_5d/gate_dev.parquet"),
        Path("data/processed/mind_r8_5d/gate_confirm.parquet"),
        Path("runs/mind_r8_5d/gate_dev_prompts.jsonl"),
        Path("runs/mind_r8_5d/gate_confirm_prompts.jsonl"),
    ]
    return report.exists() and all(path.exists() and path.stat().st_size > 0 for path in required)


def require() -> None:
    paths = [
        "artifacts/mind_r8_0_large_test_guard.json",
        "artifacts/mind_r8_2_hard_random.pt",
        "/root/caged-ltr/all-MiniLM-L6-v2",
        "data/external/mind/mteb_english/queries",
        "data/external/mind/mteb_english/corpus",
        "data/external/mind/mteb_english/data",
        "data/external/mind/mteb_english/top_ranked",
        "data/processed/mind_r8_0/large_split_ids.parquet",
    ]
    if not Path("reports/data/mind_r8_5b_gate_preregistration.json").exists():
        paths.extend(
            [
                "data/processed/mind_r7_5/queries_selected.parquet",
            ]
        )
    missing = [value for value in paths if not Path(value).exists()]
    if missing:
        raise FileNotFoundError(f"missing prerequisites: {missing}")
    guard = json.loads(Path("artifacts/mind_r8_0_large_test_guard.json").read_text())
    if guard["status"] != "locked_unaccessed" or guard["evaluation_count"] != 0:
        raise RuntimeError("large-test guard is not locked at evaluation_count=0")


def first(split: str) -> None:
    report = Path(f"runs/mind_r8_5d/{split}_first/report.json")
    if report.exists() and json.loads(report.read_text()).get("gpu_admission_complete"):
        print(f"[cached] FIRST {split}", flush=True)
        return
    run(
        [
            PYTHON,
            "scripts/run_first_r5_1_gpu_admission.py",
            "--prompt-inputs",
            f"runs/mind_r8_5d/{split}_prompts.jsonl",
            "--output-dir",
            f"runs/mind_r8_5d/{split}_first",
            "--query-limit",
            str(FIRST_RECORDS),
            "--variant",
            "all",
            "--no-full-generation",
            "--progress",
        ]
    )


def student(split: str) -> None:
    report = Path(f"reports/experiments/mind_r8_6_student_{split}.json")
    if report.exists():
        print(f"[cached] MiniLM {split}", flush=True)
        return
    run(
        [
            PYTHON,
            "scripts/evaluate_mind_r8_large_dev.py",
            "--dev",
            f"data/processed/mind_r8_5d/{split}.parquet",
            "--expected-split",
            split,
            "--model",
            "/root/caged-ltr/all-MiniLM-L6-v2",
            "--checkpoint",
            "artifacts/mind_r8_2_hard_random.pt",
            "--name",
            split,
            "--output-dir",
            "runs/mind_r8_6",
            "--report",
            str(report),
            "--progress",
        ]
    )


def main() -> None:
    source_root = str((Path.cwd() / "src").resolve())
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        source_root if not existing_pythonpath else source_root + os.pathsep + existing_pythonpath
    )
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    stage(1, "preflight and untouched-test guard")
    require()
    stage(2, "R8.5b independent gate split preregistration")
    run([PYTHON, "scripts/preregister_mind_r8_5b_gate.py", "--progress"])
    stage(3, "R8.5c-d top-20 pools and frozen FIRST prompts")
    if gate_package_ready():
        print("[cached] verified R8.5d pools and FIRST prompts", flush=True)
    else:
        run([PYTHON, "scripts/prepare_mind_r8_5d_gate.py", "--resume", "--progress"])
    stage(4, "FIRST gate-dev inference with resume cache")
    first("gate_dev")
    stage(5, "distilled MiniLM gate-dev and fixed-policy inputs")
    student("gate_dev")
    stage(6, "single gate-confirm inference")
    first("gate_confirm")
    student("gate_confirm")
    stage(7, "R8.6 calibration, route search, and R8.8 admission")
    run([PYTHON, "scripts/select_mind_r8_6_gate.py"])
    gate = json.loads(Path("reports/experiments/mind_r8_6_gate.json").read_text())
    admitted = gate["acceptance"]["gate_confirm_passed"]
    stage(8, "R8.9 one-shot decision")
    if admitted:
        run(
            [
                PYTHON,
                "scripts/prepare_mind_r8_5d_gate.py",
                "--split-ids",
                "data/processed/mind_r8_0/large_split_ids.parquet",
                "--output-dir",
                "data/processed/mind_r8_9",
                "--run-dir",
                "runs/mind_r8_9",
                "--report",
                "reports/data/mind_r8_9_package.json",
                "--splits",
                "large_test",
                "--resume",
                "--progress",
            ]
        )
        run(
            [
                PYTHON,
                "scripts/run_first_r5_1_gpu_admission.py",
                "--prompt-inputs",
                "runs/mind_r8_9/large_test_prompts.jsonl",
                "--output-dir",
                "runs/mind_r8_9/large_test_first",
                "--query-limit",
                str(FIRST_RECORDS),
                "--variant",
                "all",
                "--no-full-generation",
                "--progress",
            ]
        )
        run(
            [
                PYTHON,
                "scripts/evaluate_mind_r8_large_dev.py",
                "--dev",
                "data/processed/mind_r8_9/large_test.parquet",
                "--expected-split",
                "large_test",
                "--model",
                "/root/caged-ltr/all-MiniLM-L6-v2",
                "--checkpoint",
                "artifacts/mind_r8_2_hard_random.pt",
                "--name",
                "large_test",
                "--output-dir",
                "runs/mind_r8_9",
                "--report",
                "reports/experiments/mind_r8_9_student.json",
                "--progress",
            ]
        )
        run([PYTHON, "scripts/evaluate_mind_r8_9_once.py"])
    result = {
        "schema": "mind_r8_8_to_r8_9_decision_v1",
        "r8_8_admitted": admitted,
        "r8_9_executed": admitted,
        "large_test_evaluation_count": int(admitted),
        "decision": "preserve_locked_test" if not admitted else "executed_once_and_closed",
        "reason": "Gate confirmation failed" if not admitted else None,
    }
    output = Path("reports/experiments/mind_r8_8_to_r8_9_decision.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": "complete", **result}), flush=True)


if __name__ == "__main__":
    main()
