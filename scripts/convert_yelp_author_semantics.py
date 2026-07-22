"""Safely convert the authors' legacy NumPy item pickle to non-pickle NPY."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from caged_ltr.data.safe_numpy_pickle import decode_numpy_ndarray_pickle
from caged_ltr.reproducibility import sha256_file

EXPECTED_SOURCE_SHA256 = "a293092dca2d8fb0b6a0effb9fdcc3efdc23e9ea1cd06a36ad427fc782001cf5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "data/processed/yelp_llmesr_author/author_assets/pca64_itm_emb_np.pkl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/yelp_llmesr_author/pca64_item_embeddings.npy"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data/yelp_semantic_conversion.json"),
    )
    args = parser.parse_args()
    array = decode_numpy_ndarray_pickle(
        args.source,
        expected_sha256=EXPECTED_SOURCE_SHA256,
    ).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array, allow_pickle=False)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "decoder": "non-executing pickletools opcode validation plus numpy.frombuffer",
        "source": {
            "path": str(args.source),
            "sha256": sha256_file(args.source),
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": bool(np.isfinite(array).all()),
        },
        "pickle_load_called": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["output"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
