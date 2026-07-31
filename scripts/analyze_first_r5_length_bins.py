"""Bootstrap FIRST minus BM25 by frozen NFCorpus query-length bins."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.rows)
    frame["length_bin"] = pd.cut(
        frame.query_words,
        bins=[0, 2, 4, float("inf")],
        labels=["1-2", "3-4", "5+"],
    )
    rng = random.Random(42)
    result = {}
    for label, group in frame.groupby("length_bin", observed=True):
        values = group.first_minus_bm25.astype(float).tolist()
        boot = []
        for _ in range(10000):
            sample = [values[rng.randrange(len(values))] for _ in values]
            boot.append(sum(sample) / len(sample))
        result[str(label)] = {
            "queries": len(values),
            "mean_first_minus_bm25": sum(values) / len(values),
            "bootstrap_95ci": [sorted(boot)[250], sorted(boot)[9749]],
            "positive_fraction": sum(value > 0 for value in values) / len(values),
        }
    payload = {"queries": len(frame), "bins": result, "seed": 42}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
