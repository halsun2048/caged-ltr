"""Retrain Fashion LLMInit with shuffled and matched-random semantic controls."""

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
    run_yelp_sasrec,
)

DEFAULT_SEEDS = (42, 2024, 3407)
CONTROL_SEED = 20240725
SEMANTIC_WEIGHT = 0.25
VARIANTS = ("shuffled", "matched_random")


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:02d}m{secs:02d}s"


class _TrainingProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.index = 0
        self.label = ""
        self.max_epochs = 0
        self.patience = 0
        self.started = 0.0

    @staticmethod
    def _clear() -> None:
        sys.stderr.write("\r\033[2K")
        sys.stderr.flush()

    def start(self, index: int, label: str, config: YelpSASRecRunConfig) -> None:
        self.index = index
        self.label = label
        self.max_epochs = config.max_epochs
        self.patience = config.patience
        self.started = time.monotonic()
        self._render(epoch=0, best=0.0, stale=0, loss=None)

    def epoch(self, record: dict[str, Any]) -> None:
        key = next(name for name in record if name.startswith("best_NDCG@"))
        self._render(
            epoch=int(record["epoch"]),
            best=float(record[key]),
            stale=int(record["stale_epochs"]),
            loss=float(record["train_bpr"]),
        )

    def _render(
        self,
        *,
        epoch: int,
        best: float,
        stale: int,
        loss: float | None,
    ) -> None:
        width = 24
        filled = round(width * min(epoch / self.max_epochs, 1.0))
        bar = "#" * filled + "-" * (width - filled)
        loss_text = "" if loss is None else f" loss={loss:.4f}"
        self._clear()
        sys.stderr.write(
            f"[{self.index}/{self.total}] {self.label:<26} [{bar}] "
            f"epoch={epoch:>3}/{self.max_epochs} best={best:.6f} "
            f"stale={stale:>2}/{self.patience}{loss_text} "
            f"elapsed={_duration(time.monotonic() - self.started)}"
        )
        sys.stderr.flush()

    def finish(self, summary: dict[str, Any], *, cached: bool = False) -> None:
        self._clear()
        status = "cached" if cached else "done"
        print(
            f"[{self.index}/{self.total}] {status:<6} {self.label:<26} "
            f"best_epoch={summary['best_epoch']}",
            file=sys.stderr,
            flush=True,
        )

    def abort(self) -> None:
        self._clear()


class _EvaluationProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.index = 0
        self.label = ""

    def start(self, label: str) -> None:
        self.index += 1
        self.label = label
        self.update(0, 1)

    def update(self, done: int, total: int) -> None:
        width = 24
        filled = round(width * done / max(total, 1))
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(
            f"\r\033[2K[test {self.index}/{self.total}] {self.label:<26} "
            f"[{bar}] users={done:>4}/{total}"
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


def _overall_ndcg(aggregate: dict[str, Any]) -> float:
    return float(aggregate["item_frequency"]["overall"]["NDCG@10"]["mean"])


def _load_training_summary(
    path: Path,
    *,
    seed: int,
    semantic_sha256: str,
) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("model") != "llm_init" or summary.get("seed") != seed:
        raise ValueError(f"cached training result does not match: {path}")
    if summary.get("semantic_sha256") != semantic_sha256:
        raise ValueError(f"cached semantic control does not match: {path}")
    return summary


def _load_or_evaluate(
    config: YelpSASRecRunConfig,
    *,
    checkpoint: Path,
    kind: str,
    semantics: np.ndarray,
    path: Path,
    progress: _EvaluationProgress,
) -> dict[str, Any]:
    checkpoint_hash = sha256_file(checkpoint)
    semantic_hash = sha256_file(config.semantic_path) if config.semantic_path else ""
    progress.start(f"{kind} seed={config.seed}")
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("checkpoint_sha256") == checkpoint_hash
            and payload.get("semantic_sha256") == semantic_hash
        ):
            progress.finish(cached=True)
            return payload
    result = evaluate_full_catalog(
        config,
        checkpoint_path=checkpoint,
        semantic_variants={kind: semantics},
        semantic_weight=SEMANTIC_WEIGHT,
        progress_callback=progress.update,
    )
    payload = {
        "checkpoint_sha256": checkpoint_hash,
        "semantic_sha256": semantic_hash,
        "protocol": result.protocol,
        "metrics": {
            "llm_init": result.metrics["llm_init"],
            "semantic_only": result.metrics[f"semantic_only_{kind}"],
            "fusion": result.metrics[f"fusion_{kind}"],
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress.finish()
    return payload


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Fashion R1.3b retrained semantic controls",
        "",
        "Each control is retrained with validation-only early stopping and tested once "
        "against the full catalog.",
        "",
        "| Initialization | LLMInit NDCG@10 | Fusion NDCG@10 |",
        "|---|---:|---:|",
    ]
    real = report["real_reference"]
    lines.append(
        f"| real | {real['llm_init_overall_ndcg']:.6f} | "
        f"{real['fusion_overall_ndcg']:.6f} |"
    )
    for kind in VARIANTS:
        aggregate = report["aggregate_full_catalog"][kind]
        lines.append(
            f"| {kind} | {_overall_ndcg(aggregate['llm_init']):.6f} | "
            f"{_overall_ndcg(aggregate['fusion']):.6f} |"
        )
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
        default=Path("configs/reproduction/fashion_sasrec.yaml"),
    )
    parser.add_argument(
        "--control-report",
        type=Path,
        default=Path("reports/data/fashion_semantic_controls.json"),
    )
    parser.add_argument(
        "--real-audit-report",
        type=Path,
        default=Path("reports/experiments/fashion_r1_3b_semantic_audit.json"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/r1_3b_retrain/fashion"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/experiments/fashion_r1_3b_retrained_controls.json"),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path("reports/experiments/fashion_r1_3b_retrained_controls.md"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("seeds must be unique and non-negative")

    base = YelpSASRecRunConfig.from_yaml(args.config)
    control_report = json.loads(args.control_report.read_text(encoding="utf-8"))
    paths = {
        kind: Path(control_report["controls"][kind]["path"]) for kind in VARIANTS
    }
    semantics = {
        kind: np.load(path, allow_pickle=False).astype(np.float32)
        for kind, path in paths.items()
    }
    total = len(args.seeds) * len(VARIANTS)
    training_progress = _TrainingProgress(total)
    index = 0
    for kind in VARIANTS:
        for seed in args.seeds:
            index += 1
            run_dir = args.run_root / f"{kind}_seed{seed}"
            config = replace(
                base,
                model="llm_init",
                semantic_path=paths[kind],
                seed=seed,
                output_dir=run_dir,
                test_after_selection=False,
            )
            summary_path = run_dir / "summary.json"
            label = f"{kind} seed={seed}"
            if summary_path.is_file():
                summary = _load_training_summary(
                    summary_path,
                    seed=seed,
                    semantic_sha256=sha256_file(paths[kind]),
                )
                training_progress.start(index, label, config)
                training_progress.finish(summary, cached=True)
                continue
            training_progress.start(index, label, config)
            try:
                summary = run_yelp_sasrec(
                    config,
                    epoch_callback=training_progress.epoch,
                )
            except BaseException:
                training_progress.abort()
                raise
            training_progress.finish(summary)

    evaluation_progress = _EvaluationProgress(total)
    per_variant: dict[str, dict[str, dict[str, Any]]] = {
        kind: {} for kind in VARIANTS
    }
    protocols: dict[str, Any] = {}
    for kind in VARIANTS:
        protocols[kind] = {}
        for seed in args.seeds:
            run_dir = args.run_root / f"{kind}_seed{seed}"
            config = replace(
                base,
                model="llm_init",
                semantic_path=paths[kind],
                seed=seed,
                output_dir=run_dir,
                test_after_selection=False,
            )
            payload = _load_or_evaluate(
                config,
                checkpoint=run_dir / "best_model.pt",
                kind=kind,
                semantics=semantics[kind],
                path=run_dir / "full_catalog_test.json",
                progress=evaluation_progress,
            )
            per_variant[kind][str(seed)] = payload["metrics"]
            protocols[kind][str(seed)] = payload["protocol"]

    aggregate = {
        kind: {
            method: _aggregate(per_variant[kind], method)
            for method in ("llm_init", "semantic_only", "fusion")
        }
        for kind in VARIANTS
    }
    real_audit = json.loads(args.real_audit_report.read_text(encoding="utf-8"))
    real_llm = _overall_ndcg(real_audit["aggregate_full_catalog"]["llm_init"])
    real_fusion = _overall_ndcg(real_audit["aggregate_full_catalog"]["fusion_real"])
    report = {
        "experiment": "R1.3b-retrained-controls",
        "dataset": "fashion",
        "status": "complete",
        "seeds": args.seeds,
        "control_seed": control_report["control_seed"],
        "protocol": (
            "validation-only checkpoint selection; full-catalog test once after all "
            "settings were locked"
        ),
        "per_variant_full_catalog": per_variant,
        "aggregate_full_catalog": aggregate,
        "protocols": protocols,
        "real_reference": {
            "source": str(args.real_audit_report),
            "llm_init_overall_ndcg": real_llm,
            "fusion_overall_ndcg": real_fusion,
        },
        "acceptance": {
            "real_initialization_beats_shuffled_llm_init": (
                real_llm > _overall_ndcg(aggregate["shuffled"]["llm_init"])
            ),
            "real_initialization_beats_matched_random_llm_init": (
                real_llm > _overall_ndcg(aggregate["matched_random"]["llm_init"])
            ),
            "real_fusion_beats_shuffled_retrained_fusion": (
                real_fusion > _overall_ndcg(aggregate["shuffled"]["fusion"])
            ),
            "real_fusion_beats_matched_random_retrained_fusion": (
                real_fusion > _overall_ndcg(aggregate["matched_random"]["fusion"])
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
                "stage": "complete",
                "acceptance": report["acceptance"],
                "report": str(args.report_json),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
