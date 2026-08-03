"""Build CPU-only thesis statistics from frozen JSON reports.

The bootstrap interval here resamples the three frozen seeds. It is explicitly
seed-level uncertainty, not a replacement for query-level paired bootstrap.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "thesis"
REPORT = ROOT / "reports" / "experiments"


def main() -> None:
    multiseed = json.loads((REPORT / "mind_r7_9_multiseed.json").read_text())
    tail_floor = json.loads((REPORT / "mind_r8_11_tail_floor.json").read_text())
    runs = multiseed["runs"]
    metric_names = ("ndcg10", "hit10", "mrr")
    rng = np.random.default_rng(20240727)
    seed_summary = {}
    for metric in metric_names:
        values = np.asarray([float(row["calibration"]["trained"][metric]) for row in runs])
        samples = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
        seed_summary[metric] = {
            "seed_values": values.tolist(),
            "mean": float(values.mean()),
            "std_sample": float(values.std(ddof=1)),
            "bootstrap_ci95_seed_level": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        }
    bucket = multiseed["calibration_bucket_delta_summary"]
    bucket_rows = {
        name: {
            "mean_delta": float(row["mean"]),
            "std_sample": float(row["std"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
        }
        for name, row in bucket.items()
    }
    selected = tail_floor["selected"]
    overall = selected["overall"]
    report = {
        "schema": "caged_ltr_thesis_statistics_v1",
        "source_reports": [str(REPORT / "mind_r7_9_multiseed.json"), str(REPORT / "mind_r8_11_tail_floor.json")],
        "seed_count": len(runs),
        "bootstrap": {"iterations": 10000, "seed": 20240727, "unit": "frozen seed means"},
        "seed_summary": seed_summary,
        "bucket_delta_summary": bucket_rows,
        "tail_floor_gate": {
            "ndcg10": float(overall["ndcg10"]),
            "hit10": float(overall["hit10"]),
            "mrr": float(overall["mrr"]),
            "first_call_rate": float(overall["first_call_rate"]),
            "latency_ms": float(overall["latency_ms"]),
            "overall_gap_vs_first": float(selected["overall_gap_vs_first"]),
            "tail_gap_vs_first": float(selected["tail_gap_vs_first"]),
        },
        "untouched_test_reaccessed": False,
        "interpretation": "CI is seed-level because frozen aggregate JSON does not contain per-query vectors; do not call it a query-level paired bootstrap.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final_statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    labels = list(bucket_rows)
    means = [bucket_rows[label]["mean_delta"] for label in labels]
    errors = [bucket_rows[label]["std_sample"] for label in labels]
    figure = OUT / "head_torso_tail_delta.png"
    plt.figure(figsize=(6, 4))
    plt.bar(labels, means, yerr=errors, capsize=5, color=["#35608d", "#4d9078", "#c77c3b"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Calibration NDCG@10 delta")
    plt.title("Frozen multi-seed Head/Torso/Tail delta")
    plt.tight_layout()
    plt.savefig(figure, dpi=180)
    print(json.dumps({"stage": "complete", "report": str(OUT / "final_statistics.json"), "figure": str(figure)}))


if __name__ == "__main__":
    main()
