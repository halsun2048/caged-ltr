"""Prepare immutable R6 train/dev manifests and English FIRST distillation data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_split(query_ids: list[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(query_ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    cut = int(0.8 * len(ordered))
    return ordered[:cut], ordered[cut:]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--teacher-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    queries = pd.read_parquet(args.queries)
    candidates = pd.read_parquet(args.candidates)
    qrels = pd.read_parquet(args.qrels)
    labels = pd.read_parquet(args.teacher_labels)
    labels = labels[labels["variant"] == "baseline"].copy()
    train_ids, dev_ids = stable_split(queries["query_id"].astype(str).tolist())
    train_set, dev_set = set(train_ids), set(dev_ids)
    if train_set & dev_set:
        raise RuntimeError("train/dev query leakage")

    key = ["query_id", "passage_id"]
    label_keys = labels.rename(columns={"candidate_id": "passage_id"})
    duplicated_labels = int(label_keys.duplicated(key).sum())
    joined = candidates.merge(
        label_keys[["query_id", "passage_id", "logit", "retrieval_rank", "identifier"]],
        on=key,
        how="left",
        validate="one_to_one",
    ).merge(queries[["query_id", "query"]], on="query_id", how="left", validate="many_to_one")
    missing_logits = int(joined["logit"].isna().sum())
    missing_queries = int(joined["query"].isna().sum())
    extra_labels = int(len(label_keys.merge(candidates[key], on=key, how="left", indicator=True).query("_merge == 'left_only'")))
    if duplicated_labels or missing_logits or missing_queries or extra_labels:
        raise RuntimeError(
            f"alignment failed: duplicates={duplicated_labels}, missing_logits={missing_logits}, "
            f"missing_queries={missing_queries}, extra_labels={extra_labels}"
        )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_queries.txt").write_text("\n".join(train_ids) + "\n")
    (output / "dev_queries.txt").write_text("\n".join(dev_ids) + "\n")

    train = joined[joined["query_id"].isin(train_set)].copy()
    dev = joined[joined["query_id"].isin(dev_set)].copy()
    # Popularity is fitted strictly on train candidate exposure; unseen items are tail.
    exposure = train.groupby("passage_id")["query_id"].nunique().rename("train_item_frequency")
    positive_frequency = exposure[exposure > 0]
    tail_max, head_min = np.quantile(positive_frequency.to_numpy(), [0.2, 0.8])
    buckets = exposure.reset_index()
    buckets["frequency_bucket"] = np.where(
        buckets["train_item_frequency"] <= tail_max,
        "tail",
        np.where(buckets["train_item_frequency"] >= head_min, "head", "torso"),
    )
    buckets.to_parquet(output / "item_buckets.parquet", index=False)

    train = train.merge(buckets, on="passage_id", how="left")
    dev = dev.merge(buckets, on="passage_id", how="left")
    dev["train_item_frequency"] = dev["train_item_frequency"].fillna(0).astype(int)
    dev["frequency_bucket"] = dev["frequency_bucket"].fillna("tail")
    rel = qrels[["query_id", "passage_id", "graded_relevance"]]
    train = train.merge(rel, on=key, how="left")
    dev = dev.merge(rel, on=key, how="left")
    train["graded_relevance"] = train["graded_relevance"].fillna(0.0)
    dev["graded_relevance"] = dev["graded_relevance"].fillna(0.0)
    train["query_chars"] = train["query"].str.len()
    train["passage_chars"] = train["passage"].str.len()
    dev["query_chars"] = dev["query"].str.len()
    dev["passage_chars"] = dev["passage"].str.len()
    train["teacher_rank"] = train.groupby("query_id")["logit"].rank(method="first", ascending=False).astype(int)
    dev["teacher_rank"] = dev.groupby("query_id")["logit"].rank(method="first", ascending=False).astype(int)
    columns = [
        "query_id", "query", "passage_id", "passage", "bm25_rank", "retrieval_rank",
        "identifier", "logit", "teacher_rank", "graded_relevance", "train_item_frequency",
        "frequency_bucket", "query_chars", "passage_chars",
    ]
    train[columns].to_parquet(output / "train_listwise.parquet", index=False)
    dev[columns].to_parquet(output / "dev_listwise.parquet", index=False)

    pairs: list[dict[str, object]] = []
    hard: list[dict[str, object]] = []
    for query_id, group in train.sort_values(["query_id", "teacher_rank"]).groupby("query_id"):
        ranked = group.sort_values("teacher_rank").reset_index(drop=True)
        positive = ranked.iloc[0]
        negative_indices = sorted(set(min(index, len(ranked) - 1) for index in (1, 3, 7, 15, 19)))
        for index in negative_indices:
            negative = ranked.iloc[index]
            pairs.append({
                "query_id": query_id, "query": positive["query"],
                "positive_passage_id": positive["passage_id"], "positive_passage": positive["passage"],
                "negative_passage_id": negative["passage_id"], "negative_passage": negative["passage"],
                "positive_logit": float(positive["logit"]), "negative_logit": float(negative["logit"]),
                "teacher_margin": float(positive["logit"] - negative["logit"]),
                "negative_teacher_rank": int(negative["teacher_rank"]),
                "positive_frequency_bucket": positive["frequency_bucket"],
                "negative_frequency_bucket": negative["frequency_bucket"],
            })
        hard_pool = ranked[(ranked["teacher_rank"] > 3)].sort_values("bm25_rank").head(3)
        for _, negative in hard_pool.iterrows():
            hard.append({
                "query_id": query_id, "query": positive["query"],
                "positive_passage_id": positive["passage_id"], "positive_passage": positive["passage"],
                "negative_passage_id": negative["passage_id"], "negative_passage": negative["passage"],
                "positive_logit": float(positive["logit"]), "negative_logit": float(negative["logit"]),
                "teacher_margin": float(positive["logit"] - negative["logit"]),
                "negative_bm25_rank": int(negative["bm25_rank"]),
                "negative_teacher_rank": int(negative["teacher_rank"]),
                "positive_frequency_bucket": positive["frequency_bucket"],
                "negative_frequency_bucket": negative["frequency_bucket"],
            })
    pd.DataFrame(pairs).to_parquet(output / "train_pairwise.parquet", index=False)
    pd.DataFrame(hard).to_parquet(output / "train_hard_negatives.parquet", index=False)

    generated = [
        output / "train_queries.txt", output / "dev_queries.txt", output / "item_buckets.parquet",
        output / "train_listwise.parquet", output / "dev_listwise.parquet",
        output / "train_pairwise.parquet", output / "train_hard_negatives.parquet",
    ]
    manifest = {
        "schema": "nfcorpus_r6_distillation_v1",
        "seed": SEED,
        "split_method": "sha256(query_id), first 80% train, final 20% dev",
        "train_queries": len(train_ids), "dev_queries": len(dev_ids),
        "train_candidate_rows": len(train), "dev_candidate_rows": len(dev),
        "pairwise_rows": len(pairs), "hard_negative_rows": len(hard),
        "frequency_protocol": "unique train-query candidate exposure; 20/80 percentiles; unseen=tail",
        "frequency_boundaries": {"tail_max": float(tail_max), "head_min": float(head_min)},
        "bucket_metric_protocol": "conditional NDCG@10 over queries with relevant candidates in bucket; original rank positions retained",
        "latency_protocol": {"cached_first": "I/O only, never model latency", "first_model": "measured prefill/inference", "students": "warmup then synchronized end-to-end scoring"},
        "mind_role": "pretraining only; never NFCorpus final evidence",
        "untouched_test_accessed": False,
        "alignment": {"duplicate_teacher_keys": duplicated_labels, "missing_logits": missing_logits, "missing_queries": missing_queries, "extra_teacher_keys": extra_labels},
        "source_sha256": {str(path): sha256(path) for path in (args.queries, args.candidates, args.qrels, args.teacher_labels)},
        "generated_sha256": {path.name: sha256(path) for path in generated},
    }
    write_json(output / "manifest.json", manifest)
    write_json(args.report, manifest)
    args.report.with_suffix(".md").write_text(
        "# R6.0/R6.1 admission and distillation data\n\n```json\n" +
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n"
    )
    print(json.dumps({"stage": "complete", "report": str(args.report), "train": len(train_ids), "dev": len(dev_ids), "pairs": len(pairs), "hard": len(hard)}))


if __name__ == "__main__":
    main()
