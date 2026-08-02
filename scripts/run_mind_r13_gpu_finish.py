"""Wait for R13 reweight runs, select by dev, then benchmark the winner."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/mind_r13_reweight"
REPORTS = ROOT / "reports/experiments"


def main() -> None:
    while not (RUN / "DONE").exists():
        reports = sorted(REPORTS.glob("mind_r13_reweight_*.json"))
        print(f"[waiting reweight] completed={len(reports)}/3", flush=True)
        time.sleep(60)
    rows = []
    for path in sorted(REPORTS.glob("mind_r13_reweight_*.json")):
        payload = json.loads(path.read_text())
        rows.append(
            {
                "name": path.stem.removeprefix("mind_r13_reweight_"),
                "best_dev_ndcg10": payload["best_dev_ndcg10"],
                "checkpoint": payload["checkpoint"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "frequency_reweighting": payload["frequency_reweighting"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "peak_gpu_memory_gib": payload["peak_gpu_memory_gib"],
            }
        )
    if len(rows) != 3:
        raise RuntimeError(f"expected 3 completed variants, got {len(rows)}")
    winner = max(rows, key=lambda row: row["best_dev_ndcg10"])
    summary = {
        "schema": "mind_r13_reweight_selection_v1",
        "selection_split": "MIND dev only",
        "variants": rows,
        "selected": winner,
        "confirm_accessed": False,
        "large_test_accessed": False,
    }
    output = REPORTS / "mind_r13_reweight_selection.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[selected] {winner['name']} ndcg10={winner['best_dev_ndcg10']:.6f}", flush=True)
    command = [
        "/usr/local/miniconda3/bin/python",
        "-u",
        "scripts/benchmark_mind_r13_service_latency.py",
        "--checkpoint",
        winner["checkpoint"],
        "--progress",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:scripts"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    (RUN / "ALL_DONE").write_text("complete\n")
    print("[R13 GPU complete]", flush=True)


if __name__ == "__main__":
    main()
