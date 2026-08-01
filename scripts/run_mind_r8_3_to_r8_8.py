"""Run R8 large-dev comparisons and make the preregistered R8.9 admission decision."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(delta: np.ndarray, seed: int = 20260802) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(2000)
    for index in range(len(values)):
        values[index] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def calibration(frame: pd.DataFrame) -> dict[str, float]:
    confidence = 1 / (1 + np.exp(-frame.margin.to_numpy() / 0.05))
    outcome = frame.top1_correct.to_numpy()
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.0
    for bucket in range(10):
        selected = bins == bucket
        if selected.any():
            ece += selected.mean() * abs(confidence[selected].mean() - outcome[selected].mean())
    return {
        "ece10_fixed_margin_temperature_0p05": float(ece),
        "brier": float(np.mean((confidence - outcome) ** 2)),
    }


def wait_for(path: Path) -> None:
    while not path.exists():
        print(f"[wait] {path}", flush=True)
        time.sleep(60)


def evaluate(name: str, checkpoint: str | None) -> None:
    report = Path(f"reports/experiments/mind_r8_3_{name}.json")
    if report.exists():
        print(f"[cached] {name}", flush=True)
        return
    command = [
        sys.executable,
        "scripts/evaluate_mind_r8_large_dev.py",
        "--name",
        name,
        "--progress",
    ]
    if checkpoint:
        command.extend(["--checkpoint", checkpoint])
    print(f"[evaluate] {name}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    selection_path = Path("reports/experiments/mind_r8_2_selection.json")
    wait_for(selection_path)
    selection = json.loads(selection_path.read_text())
    winner = selection["winner"]
    for seed in (2024, 3407):
        wait_for(Path(f"reports/experiments/mind_r8_2_{winner}_seed{seed}.json"))
    models = {
        "pretrained": None,
        "r7_seed42": "/root/r7_7_formal/best.pt",
        "r7_seed2024": "/root/r7_9/seed2024/best.pt",
        "r7_seed3407": "/root/r7_9/seed3407/best.pt",
        "r8_seed42": f"artifacts/mind_r8_2_{winner}.pt",
        "r8_seed2024": f"artifacts/mind_r8_2_{winner}_seed2024.pt",
        "r8_seed3407": f"artifacts/mind_r8_2_{winner}_seed3407.pt",
    }
    for name, checkpoint in models.items():
        evaluate(name, checkpoint)
    frames = {
        name: pd.read_parquet(f"runs/mind_r8_3/{name}_query_metrics.parquet")
        for name in models
    }
    base = frames["pretrained"].set_index("query_id")
    seed_results = {}
    positive_all = True
    tail_nonnegative_all = True
    cis_exclude_zero = True
    for seed in (42, 2024, 3407):
        name = f"r8_seed{seed}"
        current = frames[name].set_index("query_id").loc[base.index]
        delta = current.ndcg10.to_numpy() - base.ndcg10.to_numpy()
        tail = current.frequency_bucket == "tail"
        tail_delta = (
            current.loc[tail, "ndcg10"].mean() - base.loc[tail, "ndcg10"].mean()
        )
        ci = bootstrap_ci(delta, seed=20260802 + seed)
        positive_all &= delta.mean() > 0
        tail_nonnegative_all &= tail_delta >= 0
        cis_exclude_zero &= ci[0] > 0
        seed_results[str(seed)] = {
            "ndcg10": float(current.ndcg10.mean()),
            "hit10": float(current.hit10.mean()),
            "mrr": float(current.mrr.mean()),
            "delta_ndcg10_vs_pretrained": float(delta.mean()),
            "delta_ndcg10_bootstrap_95ci": ci,
            "tail_delta_ndcg10_vs_pretrained": float(tail_delta),
            "calibration": calibration(current),
        }
    ndcgs = np.array([value["ndcg10"] for value in seed_results.values()])
    efficiency = {
        name: json.loads(Path(f"reports/experiments/mind_r8_3_{name}.json").read_text())[
            "efficiency"
        ]
        for name in models
    }
    guard = json.loads(Path("artifacts/mind_r8_0_large_test_guard.json").read_text())
    acceptance = {
        "all_r8_seeds_exceed_pretrained": bool(positive_all),
        "all_bootstrap_intervals_exclude_zero": bool(cis_exclude_zero),
        "tail_nonnegative_all_seeds": bool(tail_nonnegative_all),
        "three_seed_mean_ndcg10": float(ndcgs.mean()),
        "three_seed_std_ndcg10": float(ndcgs.std()),
        "student_latency_budget_measured": True,
        "first_large_dev_predictions_available": False,
        "cost_reducing_first_student_gate_demonstrated": False,
        "original_mind_large_route_available": False,
    }
    test_admitted = all(
        [
            acceptance["all_r8_seeds_exceed_pretrained"],
            acceptance["all_bootstrap_intervals_exclude_zero"],
            acceptance["tail_nonnegative_all_seeds"],
            acceptance["first_large_dev_predictions_available"],
            acceptance["cost_reducing_first_student_gate_demonstrated"],
        ]
    )
    payload = {
        "schema": "mind_r8_3_to_r8_8_admission_v1",
        "source": "mteb/MindSmallReranking English derivative",
        "winner": winner,
        "seed_results": seed_results,
        "efficiency": efficiency,
        "acceptance": acceptance,
        "r8_6_gate": {
            "status": "not_admitted",
            "reason": (
                "No FIRST predictions exist on large-dev, and pretrained/student MiniLM have "
                "the same serving cost; a quality-cost FIRST gate cannot be claimed."
            ),
        },
        "r8_7_original_mind_large": {
            "status": "unavailable",
            "mixed_with_derivative_evidence": False,
        },
        "r8_8_decision": {
            "large_test_admitted": test_admitted,
            "decision": "run_once" if test_admitted else "preserve_locked_test",
        },
        "r8_9": {
            "executed": False,
            "reason": None if test_admitted else "R8.8 admission failed",
            "guard_status": guard["status"],
            "evaluation_count": guard["evaluation_count"],
        },
    }
    report = Path("reports/experiments/mind_r8_3_to_r8_9_final.json")
    report.write_text(json.dumps(payload, indent=2) + "\n")
    report.with_suffix(".md").write_text(
        "# R8.3-R8.9 final admission\n\n```json\n"
        + json.dumps(payload, indent=2)
        + "\n```\n"
    )
    print(json.dumps({"stage": "complete", "report": str(report), "r8_9": payload["r8_9"]}))


if __name__ == "__main__":
    main()
