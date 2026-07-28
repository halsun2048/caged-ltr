"""Run staged validation-only semantic identity controls for Yelp R1.2a."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from caged_ltr.reproducibility import sha256_file
from caged_ltr.sequential import YelpSASRecRunConfig, run_yelp_sasrec

CONTROL_SEED = 20240725
TRAINING_VARIANTS = (
    "shuffled_raw",
    "matched_random_raw",
    "raw_semantic_only",
    "collaborative_only",
)


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return (
        f"{hours:d}h{minutes:02d}m"
        if hours
        else f"{minutes:02d}m{secs:02d}s"
    )


class _Progress:
    def __init__(self, total: int, *, enabled: bool) -> None:
        self.total = total
        self.enabled = enabled
        self.index = 0
        self.label = ""
        self.config: YelpSASRecRunConfig | None = None
        self.started = 0.0

    @staticmethod
    def _clear() -> None:
        sys.stderr.write("\r\033[2K")
        sys.stderr.flush()

    def start(self, index: int, label: str, config: YelpSASRecRunConfig) -> None:
        self.index = index
        self.label = label
        self.config = config
        self.started = time.monotonic()
        if self.enabled:
            self._render(0, 0.0, 0, None)

    def epoch(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            print(json.dumps(record), flush=True)
            return
        best_key = next(key for key in record if key.startswith("best_NDCG@"))
        self._render(
            int(record["epoch"]),
            float(record[best_key]),
            int(record["stale_epochs"]),
            float(record["train_bpr"]),
        )

    def _render(
        self,
        epoch: int,
        best: float,
        stale: int,
        loss: float | None,
    ) -> None:
        assert self.config is not None
        width = 24
        filled = round(width * min(epoch / self.config.max_epochs, 1.0))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self.started
        loss_text = "" if loss is None else f" loss={loss:.4f}"
        eta_text = ""
        if epoch:
            remaining = max(self.config.patience - stale, 0)
            eta_text = f" stop-ETA~{_duration(elapsed / epoch * remaining)}"
        self._clear()
        sys.stderr.write(
            f"[{self.index}/{self.total}] {self.label:<22} [{bar}] "
            f"epoch={epoch:>3}/{self.config.max_epochs} best={best:.6f} "
            f"stale={stale:>2}/{self.config.patience}{loss_text} "
            f"elapsed={_duration(elapsed)}{eta_text}"
        )
        sys.stderr.flush()

    def finish(self, summary: dict[str, Any], *, cached: bool = False) -> None:
        assert self.config is not None
        if self.enabled:
            self._clear()
        status = "cached" if cached else "done"
        score = _metric(summary, "overall", self.config.top_k)
        print(
            f"[{self.index}/{self.total}] {status:<6} {self.label:<22} "
            f"best_epoch={summary['best_epoch']:<3} "
            f"valid NDCG@{self.config.top_k}={score:.6f} "
            f"time={_duration(time.monotonic() - self.started)}",
            file=sys.stderr,
            flush=True,
        )

    def abort(self) -> None:
        if self.enabled:
            self._clear()


def _metric(summary: dict[str, Any], bucket: str, top_k: int) -> float:
    return float(
        summary["validation"]["item_frequency"][bucket][f"NDCG@{top_k}"]
    )


def _load_summary(
    path: Path,
    *,
    config: YelpSASRecRunConfig,
) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("model") != config.model or summary.get("seed") != config.seed:
        raise ValueError(f"cached model or seed mismatch: {path}")
    if summary.get("test") is not None:
        raise ValueError(f"control run must remain validation-only: {path}")
    expected_raw_hash = (
        sha256_file(config.raw_semantic_path)
        if config.raw_semantic_path is not None
        and (
            config.model == "raw_semantic_only"
            or config.model.startswith("dual_view")
        )
        else None
    )
    expected_semantic_hash = (
        sha256_file(config.semantic_path)
        if config.semantic_path is not None
        and config.model not in {"sasrec", "raw_semantic_only"}
        else None
    )
    if summary.get("raw_semantic_sha256") != expected_raw_hash:
        raise ValueError(f"cached raw semantic hash mismatch: {path}")
    if summary.get("semantic_sha256") != expected_semantic_hash:
        raise ValueError(f"cached PCA semantic hash mismatch: {path}")
    if (
        config.max_users is not None
        and int(summary.get("selected_users", -1)) != config.max_users
    ):
        raise ValueError(f"cached selected-user count mismatch: {path}")
    resolved = yaml.safe_load(
        path.with_name("resolved_config.yaml").read_text(encoding="utf-8")
    )
    expected = {
        "model": config.model,
        "seed": config.seed,
        "evaluation_seed": config.evaluation_seed,
        "max_users": config.max_users,
        "max_eval_users": config.max_eval_users,
        "max_length": config.max_length,
        "hidden_dim": config.hidden_dim,
        "num_blocks": config.num_blocks,
        "num_heads": config.num_heads,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "evaluation_negatives": config.evaluation_negatives,
        "top_k": config.top_k,
        "test_after_selection": False,
    }
    mismatches = {
        key: (resolved.get(key), value)
        for key, value in expected.items()
        if resolved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"cached resolved config mismatch in {path}: {mismatches}")
    return summary


def _run_or_load(
    config: YelpSASRecRunConfig,
    *,
    label: str,
    index: int,
    progress: _Progress,
) -> dict[str, Any]:
    summary_path = config.output_dir / "summary.json"
    progress.start(index, label, config)
    if summary_path.is_file():
        summary = _load_summary(
            summary_path,
            config=config,
        )
        progress.finish(summary, cached=True)
        return summary
    try:
        summary = run_yelp_sasrec(config, epoch_callback=progress.epoch)
    except BaseException:
        progress.abort()
        raise
    progress.finish(summary)
    return summary


def _compact(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": summary["model"],
        "seed": summary["seed"],
        "best_epoch": summary["best_epoch"],
        "epochs_ran": summary["epochs_ran"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "semantic_sha256": summary["semantic_sha256"],
        "raw_semantic_sha256": summary["raw_semantic_sha256"],
        "parameters": summary["parameters"],
        "validation": summary["validation"],
        "test": summary["test"],
        "protocol": summary["protocol"],
    }


def _write_report(
    *,
    path: Path,
    markdown_path: Path,
    summaries: dict[str, dict[str, Any]],
    top_k: int,
    minimum_overall_gain: float,
    minimum_tail_gain: float,
    identity_gate_passed: bool,
    status: str,
    controls_report: Path,
) -> dict[str, Any]:
    gains = {
        control: {
            bucket: _metric(summaries["real_raw"], bucket, top_k)
            - _metric(summaries[control], bucket, top_k)
            for bucket in ("overall", "head", "torso", "tail", "cold_start")
        }
        for control in ("shuffled_raw", "matched_random_raw")
    }
    branch_gains: dict[str, dict[str, float]] | None = None
    if {"raw_semantic_only", "collaborative_only"} <= summaries.keys():
        branch_gains = {
            branch: {
                bucket: _metric(summaries["real_raw"], bucket, top_k)
                - _metric(summaries[branch], bucket, top_k)
                for bucket in ("overall", "head", "torso", "tail", "cold_start")
            }
            for branch in ("raw_semantic_only", "collaborative_only")
        }
    report = {
        "experiment": "R1.2a-control-semantic-identity",
        "dataset": "yelp",
        "status": status,
        "seed": 42,
        "control_seed": CONTROL_SEED,
        "split": "validation",
        "test_accessed": False,
        "control_report": str(controls_report),
        "thresholds": {
            "minimum_overall_gain_over_each_identity_control": minimum_overall_gain,
            "minimum_tail_gain_over_each_identity_control": minimum_tail_gain,
        },
        "variants": {
            name: _compact(summary) for name, summary in summaries.items()
        },
        "real_minus_identity_controls": gains,
        "real_minus_single_branches": branch_gains,
        "acceptance": {
            "test_not_accessed": all(
                summary["test"] is None for summary in summaries.values()
            ),
            "identity_gate_passed": identity_gate_passed,
            "real_beats_each_identity_control_overall": all(
                value["overall"] >= minimum_overall_gain
                for value in gains.values()
            ),
            "real_beats_each_identity_control_tail": all(
                value["tail"] >= minimum_tail_gain for value in gains.values()
            ),
            "dual_beats_each_single_branch_overall": (
                all(value["overall"] > 0.0 for value in branch_gains.values())
                if branch_gains is not None
                else None
            ),
            "dual_beats_each_single_branch_tail": (
                all(value["tail"] > 0.0 for value in branch_gains.values())
                if branch_gains is not None
                else None
            ),
        },
        "stop_rule": (
            "Do not run single branches or R1.2b when the identity gate fails."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Yelp R1.2a semantic identity controls",
        "",
        f"Status: `{status}`. Seed 42 validation only; test was not accessed.",
        "",
        "| Variant | Overall | Head | Torso | Tail | Cold-start |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        values = [
            _metric(summary, bucket, top_k)
            for bucket in ("overall", "head", "torso", "tail", "cold_start")
        ]
        lines.append(
            f"| {name} | " + " | ".join(f"{value:.6f}" for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "Identity gate: "
            + ("pass." if identity_gate_passed else "fail; single branches were skipped."),
            "",
            "## Acceptance",
            "",
            *[
                f"- {name}: {value if value is None else ('pass' if value else 'fail')}"
                for name, value in report["acceptance"].items()
            ],
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reproduction/yelp_sasrec.yaml"),
    )
    parser.add_argument(
        "--controls-report",
        type=Path,
        default=Path("reports/data/yelp_raw_semantic_controls.json"),
    )
    parser.add_argument(
        "--real-summary",
        type=Path,
        default=Path(
            "runs/r1_2a/yelp/dual_view_no_ca_seed42/summary.json"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/r1_2a_controls/yelp"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path(
            "reports/experiments/yelp_r1_2a_semantic_controls_validation.json"
        ),
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=Path(
            "reports/experiments/yelp_r1_2a_semantic_controls_validation.md"
        ),
    )
    parser.add_argument("--minimum-overall-gain", type=float, default=0.003)
    parser.add_argument("--minimum-tail-gain", type=float, default=0.005)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--max-eval-users", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    base = YelpSASRecRunConfig.from_yaml(args.config)
    if base.raw_semantic_path is None:
        raise ValueError("raw_semantic_path is required")
    controls_payload = json.loads(
        args.controls_report.read_text(encoding="utf-8")
    )
    if (
        controls_payload.get("dataset") != "yelp"
        or controls_payload.get("control_seed") != CONTROL_SEED
        or controls_payload["source"]["sha256"]
        != sha256_file(base.raw_semantic_path)
    ):
        raise ValueError("raw semantic control report does not match the experiment")
    control_paths = {
        kind: Path(controls_payload["controls"][kind]["path"])
        for kind in ("shuffled", "matched_random")
    }
    for kind, control_path in control_paths.items():
        if sha256_file(control_path) != controls_payload["controls"][kind]["sha256"]:
            raise ValueError(f"{kind} control hash mismatch")

    selected_users = args.max_users if args.max_users is not None else base.max_users
    common = {
        "seed": 42,
        "max_users": selected_users,
        "max_eval_users": (
            args.max_eval_users
            if args.max_eval_users is not None
            else base.max_eval_users
        ),
        "max_epochs": (
            args.max_epochs if args.max_epochs is not None else base.max_epochs
        ),
        "test_after_selection": False,
    }
    real_config = replace(
        base,
        model="dual_view_no_ca",
        output_dir=args.real_summary.parent,
        **common,
    )
    real = _load_summary(args.real_summary, config=real_config)
    progress = _Progress(len(TRAINING_VARIANTS), enabled=args.progress)
    summaries = {"real_raw": real}
    for index, (label, kind) in enumerate(
        (
            ("shuffled raw", "shuffled"),
            ("matched-random raw", "matched_random"),
        ),
        start=1,
    ):
        config = replace(
            base,
            model="dual_view_no_ca",
            raw_semantic_path=control_paths[kind],
            output_dir=args.run_root / f"{kind}_seed42",
            **common,
        )
        summaries[f"{kind}_raw"] = _run_or_load(
            config,
            label=label,
            index=index,
            progress=progress,
        )

    top_k = base.top_k
    identity_gate_passed = all(
        _metric(real, "overall", top_k) - _metric(summaries[name], "overall", top_k)
        >= args.minimum_overall_gain
        and _metric(real, "tail", top_k) - _metric(summaries[name], "tail", top_k)
        >= args.minimum_tail_gain
        for name in ("shuffled_raw", "matched_random_raw")
    )
    if not identity_gate_passed:
        report = _write_report(
            path=args.report_json,
            markdown_path=args.report_markdown,
            summaries=summaries,
            top_k=top_k,
            minimum_overall_gain=args.minimum_overall_gain,
            minimum_tail_gain=args.minimum_tail_gain,
            identity_gate_passed=False,
            status="stopped_after_identity_controls",
            controls_report=args.controls_report,
        )
        print(
            json.dumps(
                {
                    "stage": "stopped_after_identity_controls",
                    "report": str(args.report_json),
                    "acceptance": report["acceptance"],
                }
            ),
            flush=True,
        )
        return

    semantic_config = replace(
        base,
        model="raw_semantic_only",
        output_dir=args.run_root / "raw_semantic_only_seed42",
        **common,
    )
    summaries["raw_semantic_only"] = _run_or_load(
        semantic_config,
        label="w/o collaborative",
        index=3,
        progress=progress,
    )
    collaborative_config = replace(
        base,
        model="llm_init",
        output_dir=args.run_root / "collaborative_only_seed42",
        **common,
    )
    summaries["collaborative_only"] = _run_or_load(
        collaborative_config,
        label="w/o raw semantic",
        index=4,
        progress=progress,
    )
    report = _write_report(
        path=args.report_json,
        markdown_path=args.report_markdown,
        summaries=summaries,
        top_k=top_k,
        minimum_overall_gain=args.minimum_overall_gain,
        minimum_tail_gain=args.minimum_tail_gain,
        identity_gate_passed=True,
        status="branch_controls_complete",
        controls_report=args.controls_report,
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "report": str(args.report_json),
                "acceptance": report["acceptance"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
