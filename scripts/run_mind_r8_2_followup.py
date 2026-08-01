"""Wait for R8.2 seed-42 search, then run the frozen winner on two more seeds."""

from __future__ import annotations

import json
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


def main() -> None:
    reports = {
        name: Path(f"reports/experiments/mind_r8_2_{name}.json") for name in VARIANTS
    }
    while not all(path.exists() for path in reports.values()):
        missing = [name for name, path in reports.items() if not path.exists()]
        print(f"[wait] pending seed-42 variants: {', '.join(missing)}", flush=True)
        time.sleep(60)
    metrics = {
        name: json.loads(path.read_text())["best_dev_ndcg10"] for name, path in reports.items()
    }
    winner = max(metrics, key=metrics.get)
    negative_type, objective = VARIANTS[winner]
    selection = {
        "schema": "mind_r8_2_selection_v1",
        "seed42_metrics": metrics,
        "winner": winner,
        "selection_metric": "large-dev NDCG@10",
        "large_test_accessed": False,
    }
    selection_path = Path("reports/experiments/mind_r8_2_selection.json")
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"[selection] winner={winner} ndcg10={metrics[winner]:.6f}", flush=True)
    for seed in (2024, 3407):
        command = [
            sys.executable,
            "scripts/train_mind_r8_2_large_student.py",
            "--model",
            "/root/caged-ltr/all-MiniLM-L6-v2",
            "--negative-type",
            negative_type,
            "--objective",
            objective,
            "--seed",
            str(seed),
            "--checkpoint",
            f"runs/mind_r8_2/{winner}_seed{seed}_latest.pt",
            "--best-checkpoint",
            f"artifacts/mind_r8_2_{winner}_seed{seed}.pt",
            "--report",
            f"reports/experiments/mind_r8_2_{winner}_seed{seed}.json",
            "--resume",
            "--progress",
            "--no-evaluate-before-training",
        ]
        print(f"[multiseed] starting {winner} seed={seed}", flush=True)
        subprocess.run(command, check=True)
    print("[complete] R8.2 winner multiseed training finished", flush=True)


if __name__ == "__main__":
    main()
