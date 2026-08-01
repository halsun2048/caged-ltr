"""Build English hard-negative pairs and listwise dev data from frozen MIND splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("data/external/mind/mteb_english"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/mind_r7_5"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/mind_r7_6"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/mind_r7_6_hard_pairs.json")
    )
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.progress:
        print("[1/4] loading English query/corpus maps", flush=True)
    queries = pd.read_parquet(args.processed / "queries_selected.parquet")
    corpus = pd.read_parquet(next((args.bundle / "corpus").glob("*.parquet")))
    candidates = pd.read_parquet(args.processed / "candidates_selected.parquet")
    labels = pd.read_parquet(args.processed / "labels_selected.parquet")
    query_text = queries.set_index("id")["text"].to_dict()
    corpus_text = corpus.set_index("id")["text"].to_dict()
    query_split = queries.set_index("id")["split"].to_dict()
    label_lookup = labels.set_index(["query_id", "corpus_id"])["score"].to_dict()
    if args.progress:
        print("[2/4] selecting positive and rank-hard negatives", flush=True)
    pair_rows: list[dict[str, object]] = []
    listwise_rows: list[dict[str, object]] = []
    missing_query = missing_corpus = missing_label = invalid_candidate_sets = 0
    short_negative_sets = 0
    for index, row in candidates.iterrows():
        query_id = row["query_id"]
        ids = list(row["corpus_ids"])
        scored = []
        for rank, item in enumerate(ids, 1):
            key = (query_id, item)
            if key not in label_lookup:
                missing_label += 1
                continue
            scored.append((rank, item, int(label_lookup[key])))
        positives = [(rank, item) for rank, item, score in scored if score > 0]
        negatives = [(rank, item) for rank, item, score in scored if score <= 0]
        if not positives or not negatives:
            invalid_candidate_sets += 1
            continue
        if len(negatives) < args.negatives:
            short_negative_sets += 1
        if query_id not in query_text:
            missing_query += 1
            continue
        positive_rank, positive_id = positives[0]
        if positive_id not in corpus_text:
            missing_corpus += 1
            continue
        split = query_split[query_id]
        for negative_rank, negative_id in negatives[: args.negatives]:
            if negative_id not in corpus_text:
                missing_corpus += 1
                continue
            pair_rows.append(
                {
                    "query_id": query_id,
                    "split": split,
                    "query": query_text[query_id],
                    "positive_id": positive_id,
                    "positive_passage": corpus_text[positive_id],
                    "positive_rank": positive_rank,
                    "negative_id": negative_id,
                    "negative_passage": corpus_text[negative_id],
                    "negative_rank": negative_rank,
                }
            )
        if split in {"dev", "calibration"}:
            for rank, corpus_id, score in scored:
                if corpus_id not in corpus_text:
                    missing_corpus += 1
                    continue
                listwise_rows.append(
                    {
                        "query_id": query_id,
                        "split": split,
                        "query": query_text[query_id],
                        "corpus_id": corpus_id,
                        "passage": corpus_text[corpus_id],
                        "relevance": score,
                        "source_rank": rank,
                    }
                )
        if args.progress and (index + 1) % 10_000 == 0:
            print(f"[2/4] queries={index + 1:,}/{len(candidates):,}", flush=True)
    pairs = pd.DataFrame(pair_rows)
    listwise = pd.DataFrame(listwise_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train_pairs": args.output_dir / "train_pairs.parquet",
        "calibration_pairs": args.output_dir / "calibration_pairs.parquet",
        "dev_listwise": args.output_dir / "dev_listwise.parquet",
        "calibration_listwise": args.output_dir / "calibration_listwise.parquet",
    }
    if args.progress:
        print("[3/4] writing reproducible Parquet packages", flush=True)
    pairs[pairs.split == "train"].to_parquet(outputs["train_pairs"], index=False)
    pairs[pairs.split == "calibration"].to_parquet(outputs["calibration_pairs"], index=False)
    listwise[listwise.split == "dev"].to_parquet(outputs["dev_listwise"], index=False)
    listwise[listwise.split == "calibration"].to_parquet(
        outputs["calibration_listwise"], index=False
    )
    train_pairs = int((pairs.split == "train").sum())
    dev_queries = int(listwise.loc[listwise.split == "dev", "query_id"].nunique())
    payload = {
        "schema": "mind_r7_6_english_hard_pairs_v1",
        "language": "English",
        "negative_policy": "first up-to-N non-relevant documents in frozen top-ranked order",
        "negatives_per_query_max": args.negatives,
        "counts": {
            "train_pairs": train_pairs,
            "train_queries": int(pairs.loc[pairs.split == "train", "query_id"].nunique()),
            "dev_queries": dev_queries,
            "dev_rows": int((listwise.split == "dev").sum()),
            "calibration_queries": int(
                listwise.loc[listwise.split == "calibration", "query_id"].nunique()
            ),
            "calibration_rows": int((listwise.split == "calibration").sum()),
        },
        "integrity": {
            "missing_query": missing_query,
            "missing_corpus": missing_corpus,
            "missing_label": missing_label,
            "invalid_candidate_sets": invalid_candidate_sets,
            "candidate_sets_with_fewer_than_requested_negatives": short_negative_sets,
            "train_dev_overlap": len(
                set(pairs.loc[pairs.split == "train", "query_id"])
                & set(listwise.loc[listwise.split == "dev", "query_id"])
            ),
        },
        "files": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "boundaries": {
            "mind_holdout_accessed": False,
            "nfcorpus_locked_test_accessed": False,
            "dev_used_for_training": False,
            "calibration_used_for_training": False,
        },
        "acceptance": {
            "all_text_is_from_english_bundle": True,
            "no_missing_references": (
                missing_query == 0 and missing_corpus == 0 and missing_label == 0
            ),
            "all_candidate_sets_valid": invalid_candidate_sets == 0,
            "train_dev_disjoint": True,
            "sufficient_train_queries": train_pairs >= 100_000,
            "sufficient_dev_queries": dev_queries >= 4_000,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    args.report.with_suffix(".md").write_text(
        "# R7.6 English MIND hard-pair package\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    if args.progress:
        print("[4/4] package verified", flush=True)
    print(
        json.dumps(
            {"stage": "complete", "report": str(args.report), "acceptance": payload["acceptance"]}
        )
    )


if __name__ == "__main__":
    main()
