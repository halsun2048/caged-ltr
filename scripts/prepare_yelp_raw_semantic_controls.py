"""Create deterministic controls for Yelp raw LLM item embeddings."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from caged_ltr.reproducibility import sha256_file
from caged_ltr.sequential import semantic_control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "data/processed/yelp_llmesr_author/raw_item_embeddings.npy"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/yelp_llmesr_author/raw_semantic_controls"
        ),
    )
    parser.add_argument("--seed", type=int, default=20240725)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data/yelp_raw_semantic_controls.json"),
    )
    args = parser.parse_args()

    source = np.load(args.source, allow_pickle=False).astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    controls: dict[str, dict[str, object]] = {}
    for kind in ("shuffled", "matched_random"):
        values = semantic_control(source, kind=kind, seed=args.seed)
        path = args.output_dir / f"{kind}_seed{args.seed}.npy"
        np.save(path, values, allow_pickle=False)
        controls[kind] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "finite": bool(np.isfinite(values).all()),
            "per_dimension_mean_max_abs_error": float(
                np.max(np.abs(values.mean(axis=0) - source.mean(axis=0)))
            ),
            "per_dimension_std_max_abs_error": float(
                np.max(np.abs(values.std(axis=0) - source.std(axis=0)))
            ),
        }
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": "yelp",
        "control_seed": args.seed,
        "source": {
            "path": str(args.source),
            "sha256": sha256_file(args.source),
            "shape": list(source.shape),
            "dtype": str(source.dtype),
        },
        "controls": controls,
        "definitions": {
            "shuffled": "row permutation preserving every source vector exactly",
            "matched_random": (
                "Gaussian matrix standardized to every source dimension's "
                "empirical mean and standard deviation"
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "report": str(args.report),
                "controls": controls,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
