"""Select late-fusion semantic weight on validation, then test only the winner."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from caged_ltr.sequential import (
    YelpSASRecRunConfig,
    evaluate_yelp_test_checkpoint,
    run_yelp_sasrec,
)


def _weight_name(weight: float) -> str:
    return str(weight).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reproduction/yelp_sasrec.yaml"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/yelp_weight_search"))
    parser.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    base = YelpSASRecRunConfig.from_yaml(args.config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "weight_search.json"
    previous_report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else None
    )
    candidates: list[dict[str, float | str | int]] = []
    configs: dict[float, YelpSASRecRunConfig] = {}
    tested_summaries: dict[float, dict[str, object]] = {}
    for weight in args.weights:
        output = args.output_root / f"weight_{_weight_name(weight)}"
        config = replace(
            base,
            model="late_fusion",
            semantic_weight=weight,
            output_dir=output,
            max_epochs=args.max_epochs or base.max_epochs,
            test_after_selection=False,
        )
        configs[weight] = config
        print(
            json.dumps(
                {"stage": "weight_search", "weight": weight, "output_dir": str(output)}
            ),
            flush=True,
        )
        summary_path = output / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["test"] is not None:
                was_selected = (
                    previous_report is not None
                    and previous_report.get("test_accessed_after_selection") is True
                    and float(previous_report["selected_weight"]) == weight
                )
                if not was_selected:
                    raise ValueError(f"non-selected grid run already accessed test: {output}")
                tested_summaries[weight] = summary
        else:
            summary = run_yelp_sasrec(config)
        score = float(summary["validation"]["item_frequency"]["overall"]["NDCG@10"])
        candidates.append(
            {
                "weight": weight,
                "validation_NDCG@10": score,
                "best_epoch": int(summary["best_epoch"]),
                "output_dir": str(output),
            }
        )
    selected = max(candidates, key=lambda row: float(row["validation_NDCG@10"]))
    search_report = {
        "selection_metric": "validation NDCG@10",
        "test_accessed_during_search": False,
        "evaluation_seed": base.evaluation_seed,
        "candidates": candidates,
        "selected_weight": selected["weight"],
        "selected_output_dir": selected["output_dir"],
    }
    report_path.write_text(
        json.dumps(search_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_weight = float(selected["weight"])
    selected_config = configs[selected_weight]
    if selected_weight in tested_summaries:
        test_metrics = tested_summaries[selected_weight]["test"]
    else:
        test_metrics = evaluate_yelp_test_checkpoint(selected_config)
    search_report["selected_test"] = test_metrics["item_frequency"]["overall"]
    search_report["test_accessed_after_selection"] = True
    report_path.write_text(
        json.dumps(search_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(search_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
