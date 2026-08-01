"""Summarize fixed-config English MIND runs across three seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def best_epoch_metrics(report: dict[str, object]) -> dict[str, float | int]:
    trained = [row for row in report["history"] if row.get("epoch", 0) > 0]
    best = max(trained, key=lambda row: row["ndcg10"])
    return {
        "epoch": int(best["epoch"]),
        "ndcg10": float(best["ndcg10"]),
        "hit10": float(best["hit10"]),
        "mrr": float(best["mrr"]),
    }


def peak_memory(path: Path) -> float | None:
    values = [float(value) for value in re.findall(r"gpu=([0-9.]+)GiB", path.read_text())]
    return max(values) if values else None


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--train-report", type=Path, action="append", required=True)
    parser.add_argument("--calibration-report", type=Path, action="append", required=True)
    parser.add_argument("--train-log", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiments/mind_r7_9_multiseed.json"),
    )
    args = parser.parse_args()
    lengths = {
        len(args.seed),
        len(args.train_report),
        len(args.calibration_report),
        len(args.train_log),
    }
    if lengths != {3}:
        raise ValueError("exactly three aligned seeds/reports/logs are required")
    runs = []
    for seed, train_path, calibration_path, log_path in zip(
        args.seed,
        args.train_report,
        args.calibration_report,
        args.train_log,
        strict=True,
    ):
        train = json.loads(train_path.read_text())
        calibration = json.loads(calibration_path.read_text())
        best = best_epoch_metrics(train)
        buckets = calibration["bucket_ndcg10_delta"]
        run = {
            "seed": seed,
            "dev": best,
            "calibration": {
                "baseline": calibration["baseline"]["overall"],
                "trained": calibration["trained"]["overall"],
                "ndcg10_delta": calibration["paired_ndcg10_delta"],
                "bucket_ndcg10_delta": buckets,
            },
            "elapsed_seconds": float(train["elapsed_seconds"]),
            "peak_gpu_memory_gib": peak_memory(log_path),
            "files": {
                "train_report": {"path": str(train_path), "sha256": sha256(train_path)},
                "calibration_report": {
                    "path": str(calibration_path),
                    "sha256": sha256(calibration_path),
                },
                "train_log_sha256": sha256(log_path),
            },
        }
        runs.append(run)
    summary = {
        "dev_ndcg10": stats([run["dev"]["ndcg10"] for run in runs]),
        "dev_hit10": stats([run["dev"]["hit10"] for run in runs]),
        "dev_mrr": stats([run["dev"]["mrr"] for run in runs]),
        "calibration_ndcg10": stats([run["calibration"]["trained"]["ndcg10"] for run in runs]),
        "calibration_hit10": stats([run["calibration"]["trained"]["hit10"] for run in runs]),
        "calibration_mrr": stats([run["calibration"]["trained"]["mrr"] for run in runs]),
        "calibration_ndcg10_delta": stats(
            [run["calibration"]["ndcg10_delta"]["mean"] for run in runs]
        ),
        "elapsed_seconds": stats([run["elapsed_seconds"] for run in runs]),
        "peak_gpu_memory_gib": stats([run["peak_gpu_memory_gib"] for run in runs]),
    }
    bucket_summary = {
        bucket: stats([run["calibration"]["bucket_ndcg10_delta"][bucket]["mean"] for run in runs])
        for bucket in ("head", "torso", "tail")
    }
    payload = {
        "schema": "mind_r7_9_multiseed_v1",
        "configuration": {
            "fixed_without_search": True,
            "seeds": args.seed,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "epochs_max": 5,
            "batch_size": 256,
            "learning_rate": 2e-5,
            "precision": "bf16",
            "patience": 2,
        },
        "runs": runs,
        "summary": summary,
        "calibration_bucket_delta_summary": bucket_summary,
        "boundary": {
            "calibration_is_fixed_confirmation_not_tuning": True,
            "mind_large_test_accessed": False,
            "nfcorpus_locked_test_accessed": False,
        },
        "acceptance": {
            "all_seeds_calibration_beat_pretrained": all(
                run["calibration"]["ndcg10_delta"]["mean"] > 0 for run in runs
            ),
            "all_seed_bootstrap_ci95_exclude_zero": all(
                run["calibration"]["ndcg10_delta"]["ci95_low"] > 0 for run in runs
            ),
            "head_direction_positive_all_seeds": all(
                run["calibration"]["bucket_ndcg10_delta"]["head"]["mean"] > 0 for run in runs
            ),
            "torso_direction_positive_all_seeds": all(
                run["calibration"]["bucket_ndcg10_delta"]["torso"]["mean"] > 0 for run in runs
            ),
            "tail_direction_positive_all_seeds": all(
                run["calibration"]["bucket_ndcg10_delta"]["tail"]["mean"] > 0 for run in runs
            ),
            "large_test_not_accessed": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(
        "# R7.9 fixed-config three-seed stability\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    print(
        json.dumps(
            {"stage": "complete", "report": str(args.output), "acceptance": payload["acceptance"]}
        )
    )


if __name__ == "__main__":
    main()
