"""Safely convert an LLM-ESR PCA64 item pickle to a non-pickle NPY file."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from caged_ltr.data.safe_numpy_pickle import decode_numpy_ndarray_pickle
from caged_ltr.reproducibility import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("yelp", "fashion", "beauty"), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    processed_dir = Path(f"data/processed/{args.dataset}_llmesr_author")
    source = args.source or processed_dir / "author_assets/pca64_itm_emb_np.pkl"
    output = args.output or processed_dir / "pca64_item_embeddings.npy"
    manifest_path = args.manifest or Path(
        f"reports/data/{args.dataset}_llmesr_author_summary.json"
    )
    report_path = args.report or Path(
        f"reports/data/{args.dataset}_semantic_conversion.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha256 = manifest["author_assets"]["pca64_itm_emb_np.pkl"]["sha256"]
    array = decode_numpy_ndarray_pickle(
        source,
        expected_sha256=expected_sha256,
    ).astype(np.float32)
    expected_items = int(manifest["statistics"]["items"])
    if array.shape != (expected_items, 64):
        raise ValueError(
            f"semantic matrix shape {array.shape} does not match ({expected_items}, 64)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, array, allow_pickle=False)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "decoder": "non-executing pickletools opcode validation plus numpy.frombuffer",
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": bool(np.isfinite(array).all()),
        },
        "pickle_load_called": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["output"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
