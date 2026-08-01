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
TOTAL_STAGES = 8


def stage(number: int, label: str) -> None:
    filled = round(30 * number / TOTAL_STAGES)
    bar = "#" * filled + "-" * (30 - filled)
    print(f"\n[R8 {number}/{TOTAL_STAGES}] [{bar}] {label}", flush=True)


def wait_line(label: str, started: float, tick: int) -> None:
    width = 24
    position = tick % (2 * width - 2)
    if position >= width:
        position = 2 * width - 2 - position
    bar = ["-"] * width
    bar[position] = "#"
    elapsed = int(time.monotonic() - started)
    print(
        f"\r[waiting] [{''.join(bar)}] {label} elapsed={elapsed // 60:02d}m{elapsed % 60:02d}s",
        end="",
        flush=True,
    )


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
    started = time.monotonic()
    tick = 0
    while not report.exists() and process_is_running(required):
        wait_line(str(report), started, tick)
        tick += 1
        time.sleep(5)
    print(flush=True)
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
    stage(1, "边界与 test guard 审计")
    validate_boundaries()
    stage(2, "四组 seed=42 搜索")
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
    stage(3, f"胜出配置 {winner} 多种子训练")
    negative_type, objective = VARIANTS[winner]
    if process_is_running(("run_mind_r8_2_followup.py",)):
        expected = [
            Path(f"reports/experiments/mind_r8_2_{winner}_seed{seed}.json")
            for seed in (2024, 3407)
        ]
        started = time.monotonic()
        tick = 0
        while process_is_running(("run_mind_r8_2_followup.py",)) and not all(
            path.exists() for path in expected
        ):
            wait_line("existing multiseed followup", started, tick)
            tick += 1
            time.sleep(5)
        print(flush=True)
    for seed in (2024, 3407):
        train(winner, negative_type, objective, seed)
    stage(4, "R8.3 large-dev 正式对照与分桶")
    print("[R8.2 complete] starting R8.3-R8.8 formal evaluation", flush=True)
    subprocess.run([sys.executable, "scripts/run_mind_r8_3_to_r8_8.py"], check=True)
    final = json.loads(Path("reports/experiments/mind_r8_3_to_r8_9_final.json").read_text())
    decision = final["r8_8_decision"]["decision"]
    if decision == "run_once":
        raise RuntimeError(
            "R8.8 admitted test, but no audited large-test materializer exists; refusing an "
            "unsafe ad-hoc access. Implement the preregistered one-shot evaluator first."
        )
    stage(5, "R8.4 效率、显存与吞吐汇总")
    stage(6, "R8.5 校准与 R8.6 Gate 审核")
    stage(7, "R8.7-R8.8 证据边界和准入决策")
    stage(8, "R8.9 最终决策")
    print("[R8.9 complete] test preserved because R8.8 admission failed", flush=True)


if __name__ == "__main__":
    main()
