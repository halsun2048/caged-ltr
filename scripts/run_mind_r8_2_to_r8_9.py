"""Resumable one-command runner for the complete preregistered R8.2-R8.9 workflow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = {
    "hard_only": ("rank_hard", "pairwise"),
    "random_only": ("random", "pairwise"),
    "hard_random": ("all", "pairwise"),
    "hard_random_inbatch": ("all", "pairwise_inbatch"),
}
MODEL = "/root/caged-ltr/all-MiniLM-L6-v2"


def process_is_running(required: tuple[str, ...]) -> bool:
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if all(fragment in command for fragment in required):
            return True
    return False


def wait_for_external_run(report: Path) -> bool:
    required = ("train_mind_r8_2_large_student.py", str(report))
    if not process_is_running(required):
        return False
    while not report.exists() and process_is_running(required):
        print(f"[wait-running] {report}", flush=True)
        time.sleep(30)
    return report.exists()


def train(name: str, negative_type: str, objective: str, seed: int) -> None:
    suffix = "" if seed == 42 else f"_seed{seed}"
    report = Path(f"reports/experiments/mind_r8_2_{name}{suffix}.json")
    if report.exists():
        value = json.loads(report.read_text()).get("best_dev_ndcg10")
        print(f"[cached] {name} seed={seed} ndcg10={value}", flush=True)
        return
    if wait_for_external_run(report):
        print(f"[external-complete] {name} seed={seed}", flush=True)
        return
    checkpoint = f"runs/mind_r8_2/{name}{suffix}_latest.pt"
    best = f"artifacts/mind_r8_2_{name}{suffix}.pt"
    command = [
        sys.executable,
        "scripts/train_mind_r8_2_large_student.py",
        "--model",
        MODEL,
        "--negative-type",
        negative_type,
        "--objective",
        objective,
        "--seed",
        str(seed),
        "--checkpoint",
        checkpoint,
        "--best-checkpoint",
        best,
        "--report",
        str(report),
        "--resume",
        "--progress",
    ]
    if seed != 42:
        command.append("--no-evaluate-before-training")
    print(f"[train] {name} seed={seed}", flush=True)
    subprocess.run(command, check=True)


def validate_boundaries() -> None:
    guard_path = Path("artifacts/mind_r8_0_large_test_guard.json")
    guard = json.loads(guard_path.read_text())
    if guard["status"] != "locked_unaccessed" or guard["evaluation_count"] != 0:
        raise RuntimeError("large-test guard is not in its preregistered initial state")
    prereg = json.loads(Path("configs/mind_r8_2_preregistered.json").read_text())
    if not prereg["frozen_before_large_dev_scoring"]:
        raise RuntimeError("R8.2 preregistration is invalid")
    print("[guard] large-test locked; evaluation_count=0", flush=True)


def main() -> None:
    validate_boundaries()
    for name, (negative_type, objective) in VARIANTS.items():
        train(name, negative_type, objective, 42)
    metrics = {
        name: json.loads(
            Path(f"reports/experiments/mind_r8_2_{name}.json").read_text()
        )["best_dev_ndcg10"]
        for name in VARIANTS
    }
    winner = max(metrics, key=metrics.get)
    selection = {
        "schema": "mind_r8_2_selection_v1",
        "seed42_metrics": metrics,
        "winner": winner,
        "selection_metric": "large-dev NDCG@10",
        "large_test_accessed": False,
    }
    selection_path = Path("reports/experiments/mind_r8_2_selection.json")
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"[winner] {winner} ndcg10={metrics[winner]:.6f}", flush=True)
    negative_type, objective = VARIANTS[winner]
    if process_is_running(("run_mind_r8_2_followup.py",)):
        expected = [
            Path(f"reports/experiments/mind_r8_2_{winner}_seed{seed}.json")
            for seed in (2024, 3407)
        ]
        while process_is_running(("run_mind_r8_2_followup.py",)) and not all(
            path.exists() for path in expected
        ):
            print("[wait-running] existing multiseed followup", flush=True)
            time.sleep(30)
    for seed in (2024, 3407):
        train(winner, negative_type, objective, seed)
    print("[R8.2 complete] starting R8.3-R8.8 formal evaluation", flush=True)
    subprocess.run([sys.executable, "scripts/run_mind_r8_3_to_r8_8.py"], check=True)
    final = json.loads(Path("reports/experiments/mind_r8_3_to_r8_9_final.json").read_text())
    decision = final["r8_8_decision"]["decision"]
    if decision == "run_once":
        raise RuntimeError(
            "R8.8 admitted test, but no audited large-test materializer exists; refusing an "
            "unsafe ad-hoc access. Implement the preregistered one-shot evaluator first."
        )
    print("[R8.9 complete] test preserved because R8.8 admission failed", flush=True)


if __name__ == "__main__":
    main()
