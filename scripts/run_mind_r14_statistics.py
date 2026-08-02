"""R14 paired statistical audit over already-materialized per-query reports.

This script never runs a model. If comparator columns are absent it records that
the audit cannot be performed without reconstructing a locked evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_bootstrap(delta: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def audit(path: Path, label: str, draws: int) -> dict[str, object]:
    frame = pd.read_parquet(path)
    result: dict[str, object] = {
        "label": label,
        "path": str(path),
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "source_sha256": sha256(path),
        "test_accessed": False,
    }
    required = {"first_ndcg10", "gate_ndcg10"}
    if not required.issubset(frame.columns):
        result.update(
            {
                "status": "not_available",
                "reason": "saved artifact contains student metrics only; first_ndcg10/gate_ndcg10 were not persisted",
                "paired_bootstrap": None,
            }
        )
        return result
    deltas = frame["gate_ndcg10"].to_numpy(float) - frame["first_ndcg10"].to_numpy(float)
    result.update(
        {
            "status": "complete",
            "paired_bootstrap": {
                "gate_minus_first_ndcg10": paired_bootstrap(deltas, 20260810, draws),
                "observed_mean": float(deltas.mean()),
                "positive_fraction": float((deltas > 0).mean()),
            },
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r9", type=Path, default=Path("runs/mind_r8_9_tail/large_test_query_metrics.parquet"))
    parser.add_argument("--r12", type=Path, default=Path("runs/mind_r12_0/r12_confirm_query_metrics.parquet"))
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("reports/experiments/mind_r14_statistics.json"))
    args = parser.parse_args()
    reports = []
    for path, label in ((args.r9, "R9 large-test frozen artifact"), (args.r12, "R12 confirm frozen artifact")):
        if path.exists():
            reports.append(audit(path, label, args.draws))
        else:
            reports.append({"label": label, "status": "missing", "path": str(path), "test_accessed": False})
    payload = {
        "schema": "mind_r14_statistics_v1",
        "protocol": "offline audit only; no model inference and no labels newly materialized",
        "reports": reports,
        "large_test_reopened": False,
        "confirm_reopened": False,
        "interpretation": "A paired CI is reported only when both comparator columns were persisted in the source artifact.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "report": str(args.output), "statuses": [x["status"] for x in reports]}))


if __name__ == "__main__":
    main()
