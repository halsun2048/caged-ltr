"""Run the locked Beauty confidence gate against the complete item catalog."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from caged_ltr.reproducibility import sha256_file
from caged_ltr.sequential import (
    YelpSASRecRunConfig,
    evaluate_full_catalog,
    semantic_control,
)

DEFAULT_SEEDS = (42, 2024, 3407)
CONTROL_SEED = 20240725
METHOD_MAP = {
    "llm_init": "llm_init",
    "semantic_only": "semantic_only_real",
    "fixed_fusion": "fusion_real",
    "confidence_gate": "confidence_gate_real",
    "shuffled_gate": "confidence_gate_shuffled",
}


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:02d}m{seconds:02d}s"


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.index = 0
        self.seed = 0
        self.started = 0.0

    def start(self, index: int, seed: int, users: int) -> None:
        self.index = index
        self.seed = seed
        self.started = time.monotonic()
        self.update(0, users)

    def update(self, done: int, total: int) -> None:
        width = 24
        filled = round(width * done / max(total, 1))
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(
            f"\r\033[2K[{self.index}/{self.total}] seed={self.seed:<4} "
            f"[{bar}] users={done:>5}/{total} "
            f"elapsed={_duration(time.monotonic() - self.started)}"
        )
        sys.stderr.flush()

    def finish(self, *, cached: bool = False) -> None:
        print(" cached" if cached else " done", file=sys.stderr, flush=True)


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def _aggregate(per_seed: dict[str, dict[str, Any]], method: str) -> dict[str, Any]:
    seeds = list(per_seed)
    output: dict[str, Any] = {}
    for family in ("item_frequency", "item_paper", "user_frequency", "user_paper"):
        output[family] = {}
        for bucket in per_seed[seeds[0]][method][family]:
            output[family][bucket] = {
                metric: _mean_std(
                    [
                        float(per_seed[seed][method][family][bucket][metric])
                        for seed in seeds
                    ]
                )
                for metric in ("Hit@10", "NDCG@10")
            }
    return output


def _metric(metrics: dict[str, Any], bucket: str) -> float:
    return float(metrics["item_frequency"][bucket]["NDCG@10"])


def _cache_name(max_eval_users: int | None) -> str:
    suffix = "all" if max_eval_users is None else str(max_eval_users)
    return f"full_catalog_{suffix}"


def _load_cache(
    path: Path,
    *,
    expected: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("inputs") != expected:
        return None
    ranks_path = Path(payload.get("ranks_path", ""))
    if not ranks_path.is_file():
        return None
    return payload


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Beauty R1.5 full-catalog robustness audit",
        "",
        "Post-hoc protocol audit with the gate and all weights locked before "
        "full-catalog access.",
        "",
        f"Evaluated users per seed: `{report['protocol']['evaluated_users_per_seed']}`; "
        f"catalog items: `{report['protocol']['catalog_items']}`.",
        "",
        "| Method | H@10 mean ± std | NDCG@10 mean ± std |",
        "|---|---:|---:|",
    ]
    for method in METHOD_MAP:
        overall = report["aggregate_test"][method]["item_frequency"]["overall"]
        hit = overall["Hit@10"]
        ndcg = overall["NDCG@10"]
        lines.append(
            f"| {method} | {hit['mean']:.6f} ± {hit['std']:.6f} | "
            f"{ndcg['mean']:.6f} ± {ndcg['std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Item-frequency NDCG@10",
            "",
            "| Method | Head | Torso | Tail | Cold-start |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_MAP:
        groups = report["aggregate_test"][method]["item_frequency"]
        cells = [
            f"{groups[bucket]['NDCG@10']['mean']:.6f} ± "
            f"{groups[bucket]['NDCG@10']['std']:.6f}"
            for bucket in ("head", "torso", "tail", "cold_start")
        ]
        lines.append(f"| {method} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            *[
                f"- {name}: {'pass' if passed else 'fail'}"
                for name, passed in report["acceptance"].items()
            ],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reproduction/beauty_sasrec.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs/r1_4/beauty"))
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("runs/r1_5/beauty"),
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path("reports/experiments/beauty_r1_4.json"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/beauty_r1_5_full_catalog.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/beauty_r1_5_full_catalog.md"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("seeds must be unique and non-negative")
    if args.max_eval_users is not None and args.max_eval_users <= 0:
        raise ValueError("max-eval-users must be positive")

    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source["status"] != "complete":
        raise ValueError("Beauty R1.4 source report is incomplete")
    protocol = source["protocol"]
    base_weight = float(protocol["base_semantic_weight"])
    gate_weight = float(protocol["gated_residual_weight"])
    if protocol["fusion_selection_dataset"] != "Yelp validation only":
        raise ValueError("fusion settings were not externally locked")

    base = YelpSASRecRunConfig.from_yaml(args.config)
    if base.semantic_path is None:
        raise ValueError("semantic_path is required")
    if args.evaluation_batch_size is not None:
        if args.evaluation_batch_size <= 0:
            raise ValueError("evaluation-batch-size must be positive")
        base = replace(base, evaluation_batch_size=args.evaluation_batch_size)
    real_semantics = np.load(base.semantic_path, allow_pickle=False).astype(np.float32)
    shuffled_path = (
        base.processed_dir
        / "semantic_controls"
        / f"shuffled_seed{CONTROL_SEED}.npy"
    )
    if not shuffled_path.is_file():
        shuffled_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            shuffled_path,
            semantic_control(
                real_semantics,
                kind="shuffled",
                seed=CONTROL_SEED,
            ),
            allow_pickle=False,
        )
    shuffled_semantics = np.load(shuffled_path, allow_pickle=False).astype(np.float32)
    data_report = json.loads(base.report_path.read_text(encoding="utf-8"))
    data_fingerprint = data_report["processed_fingerprint"]
    available_users = int(data_report["statistics"]["evaluable_users"])
    expected_users = min(
        available_users,
        args.max_eval_users or available_users,
    )
    cache_stem = _cache_name(args.max_eval_users)
    progress = _Progress(len(args.seeds))
    per_seed: dict[str, dict[str, Any]] = {}
    input_records: list[dict[str, Any]] = []
    selected_users: int | None = None
    catalog_items: int | None = None

    for index, seed in enumerate(args.seeds, start=1):
        checkpoint = args.run_root / f"llm_init_seed{seed}" / "best_model.pt"
        seed_root = args.audit_root / f"llm_init_seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        cache_path = seed_root / f"{cache_stem}.json"
        ranks_path = seed_root / f"{cache_stem}_ranks.npz"
        expected = {
            "seed": seed,
            "checkpoint_sha256": sha256_file(checkpoint),
            "real_semantic_sha256": sha256_file(base.semantic_path),
            "shuffled_semantic_sha256": sha256_file(shuffled_path),
            "data_fingerprint": data_fingerprint,
            "base_semantic_weight": base_weight,
            "gated_residual_weight": gate_weight,
            "max_eval_users": args.max_eval_users,
            "evaluation_batch_size": base.evaluation_batch_size,
        }
        progress.start(index, seed, expected_users)
        cached = _load_cache(cache_path, expected=expected)
        if cached is not None:
            progress.finish(cached=True)
            payload = cached
        else:
            config = replace(
                base,
                model="llm_init",
                seed=seed,
                output_dir=args.run_root / f"llm_init_seed{seed}",
                max_eval_users=args.max_eval_users,
                test_after_selection=False,
            )
            started = time.monotonic()
            result = evaluate_full_catalog(
                config,
                checkpoint_path=checkpoint,
                semantic_variants={
                    "real": real_semantics,
                    "shuffled": shuffled_semantics,
                },
                semantic_weight=base_weight,
                gated_residual_weight=gate_weight,
                progress_callback=progress.update,
            )
            np.savez(
                ranks_path,
                **{
                    method: ranks.astype(np.int64, copy=False)
                    for method, ranks in result.ranks.items()
                },
            )
            payload = {
                "inputs": expected,
                "elapsed_seconds": time.monotonic() - started,
                "ranks_path": str(ranks_path),
                "protocol": result.protocol,
                "metrics": result.metrics,
            }
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            progress.finish()
        protocol_result = payload["protocol"]
        if selected_users is None:
            selected_users = int(protocol_result["selected_users"])
            catalog_items = int(protocol_result["candidate_catalog_size"])
        elif (
            selected_users != int(protocol_result["selected_users"])
            or catalog_items != int(protocol_result["candidate_catalog_size"])
        ):
            raise ValueError("cached full-catalog scopes differ across seeds")
        per_seed[str(seed)] = {
            public: payload["metrics"][internal]
            for public, internal in METHOD_MAP.items()
        }
        input_records.append(
            {
                **expected,
                "elapsed_seconds": payload["elapsed_seconds"],
                "cache": str(cache_path),
                "ranks": payload["ranks_path"],
            }
        )

    aggregate = {
        method: _aggregate(per_seed, method)
        for method in METHOD_MAP
    }
    overall_gate_minus_fixed = [
        _metric(per_seed[str(seed)]["confidence_gate"], "overall")
        - _metric(per_seed[str(seed)]["fixed_fusion"], "overall")
        for seed in args.seeds
    ]
    tail_gate_minus_fixed = [
        _metric(per_seed[str(seed)]["confidence_gate"], "tail")
        - _metric(per_seed[str(seed)]["fixed_fusion"], "tail")
        for seed in args.seeds
    ]
    tail_real_minus_shuffled = [
        _metric(per_seed[str(seed)]["confidence_gate"], "tail")
        - _metric(per_seed[str(seed)]["shuffled_gate"], "tail")
        for seed in args.seeds
    ]
    report = {
        "experiment": "R1.5-full-catalog",
        "dataset": "beauty",
        "status": (
            "complete" if args.max_eval_users is None else "smoke_complete"
        ),
        "analysis_type": "post-hoc evaluation-protocol robustness audit",
        "seeds": args.seeds,
        "protocol": {
            "method_selection": "Yelp validation only; unchanged from R1.4",
            "base_semantic_weight": base_weight,
            "gated_residual_weight": gate_weight,
            "test_usage": "reused after primary sampled-1000 result; no retuning",
            "evaluation_scope": "complete item catalog",
            "evaluated_users_per_seed": selected_users,
            "catalog_items": catalog_items,
            "max_eval_users": args.max_eval_users,
        },
        "inputs": input_records,
        "per_seed_test": per_seed,
        "aggregate_test": aggregate,
        "gains_gate_minus_fixed": {
            "overall_ndcg_per_seed": overall_gate_minus_fixed,
            "tail_ndcg_per_seed": tail_gate_minus_fixed,
            "overall_ndcg_mean": float(np.mean(overall_gate_minus_fixed)),
            "tail_ndcg_mean": float(np.mean(tail_gate_minus_fixed)),
        },
        "gains_real_gate_minus_shuffled_gate": {
            "tail_ndcg_per_seed": tail_real_minus_shuffled,
            "tail_ndcg_mean": float(np.mean(tail_real_minus_shuffled)),
        },
        "acceptance": {
            "overall_gate_beats_fixed_all_seeds": all(
                gain > 0.0 for gain in overall_gate_minus_fixed
            ),
            "tail_gate_beats_fixed_all_seeds": all(
                gain > 0.0 for gain in tail_gate_minus_fixed
            ),
            "tail_real_gate_beats_shuffled_all_seeds": all(
                gain > 0.0 for gain in tail_real_minus_shuffled
            ),
            "tail_gate_mean_absolute_gain_at_least_0p005": (
                float(np.mean(tail_gate_minus_fixed)) >= 0.005
            ),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, args.report_markdown)
    print(
        json.dumps(
            {
                "stage": report["status"],
                "report": str(args.report_json),
                "acceptance": report["acceptance"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
