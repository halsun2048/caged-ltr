"""One-time frozen calibration evaluation of the English MIND student."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from train_mind_r7_7_english_student import BiEncoder, embed_texts, ndcg_at_10
from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def item_frequency(train: pd.DataFrame) -> tuple[dict[str, int], float, float]:
    unique = train[["query_id", "positive_id"]].drop_duplicates()
    frequency = unique.positive_id.value_counts()
    return frequency.to_dict(), float(frequency.quantile(0.5)), float(frequency.quantile(0.8))


def metrics_by_query(dev, query_vectors, corpus_vectors, frequency, torso_cut, head_cut):
    query_table = dev[["query_id", "query"]].drop_duplicates("query_id")
    corpus_table = dev[["corpus_id", "passage"]].drop_duplicates("corpus_id")
    query_map = dict(zip(query_table.query_id, query_vectors, strict=True))
    corpus_map = dict(zip(corpus_table.corpus_id, corpus_vectors, strict=True))
    rows = []
    for query_id, group in dev.groupby("query_id", sort=False):
        passages = np.stack([corpus_map[value] for value in group.corpus_id])
        scores = passages @ query_map[query_id]
        relevance = group.relevance.to_numpy()[np.argsort(-scores, kind="stable")]
        relevant = np.flatnonzero(relevance > 0)
        positive_ids = group.loc[group.relevance > 0, "corpus_id"]
        maximum_frequency = max((frequency.get(value, 0) for value in positive_ids), default=0)
        if maximum_frequency >= head_cut:
            bucket = "head"
        elif maximum_frequency >= torso_cut:
            bucket = "torso"
        else:
            bucket = "tail"
        rows.append(
            {
                "query_id": query_id,
                "ndcg10": ndcg_at_10(relevance),
                "hit10": float(np.any(relevance[:10] > 0)),
                "mrr": 1 / (int(relevant[0]) + 1) if len(relevant) else 0.0,
                "frequency_bucket": bucket,
            }
        )
    return pd.DataFrame(rows)


def score_model(model, tokenizer, dev, device, max_length, batch_size, dtype):
    query_table = dev[["query_id", "query"]].drop_duplicates("query_id")
    corpus_table = dev[["corpus_id", "passage"]].drop_duplicates("corpus_id")
    started = time.perf_counter()
    query_vectors = embed_texts(
        model,
        tokenizer,
        query_table["query"].tolist(),
        device,
        max_length,
        batch_size,
        dtype,
    )
    corpus_vectors = embed_texts(
        model,
        tokenizer,
        corpus_table["passage"].tolist(),
        device,
        max_length,
        batch_size,
        dtype,
    )
    elapsed = time.perf_counter() - started
    return query_vectors, corpus_vectors, 1_000 * elapsed / len(query_table)


def summarize(frame: pd.DataFrame, latency: float) -> dict[str, object]:
    return {
        "overall": {
            "ndcg10": float(frame.ndcg10.mean()),
            "hit10": float(frame.hit10.mean()),
            "mrr": float(frame.mrr.mean()),
            "latency_ms_per_query": latency,
        },
        "buckets": {
            bucket: {
                "queries": len(group),
                "ndcg10": float(group.ndcg10.mean()),
                "hit10": float(group.hit10.mean()),
                "mrr": float(group.mrr.mean()),
            }
            for bucket, group in frame.groupby("frequency_bucket")
        },
    }


def bootstrap(delta: np.ndarray, samples: int, seed: int) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    means = np.empty(samples)
    for index in range(samples):
        means[index] = generator.choice(delta, size=len(delta), replace=True).mean()
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "probability_positive": float(np.mean(means > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("formal calibration evaluation requires CUDA")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    train = pd.read_parquet(args.train)
    calibration = pd.read_parquet(args.calibration)
    frequency, torso_cut, head_cut = item_frequency(train)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.progress:
        print("[1/3] scoring frozen pretrained baseline", flush=True)
    baseline_model = BiEncoder(args.model).to(device).eval()
    baseline_q, baseline_c, baseline_latency = score_model(
        baseline_model,
        tokenizer,
        calibration,
        device,
        args.max_length,
        args.batch_size,
        dtype,
    )
    baseline = metrics_by_query(calibration, baseline_q, baseline_c, frequency, torso_cut, head_cut)
    del baseline_model, baseline_q, baseline_c
    torch.cuda.empty_cache()
    if args.progress:
        print("[2/3] scoring frozen trained checkpoint", flush=True)
    trained_model = BiEncoder(args.model).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trained_model.load_state_dict(checkpoint["model"], strict=True)
    trained_q, trained_c, trained_latency = score_model(
        trained_model,
        tokenizer,
        calibration,
        device,
        args.max_length,
        args.batch_size,
        dtype,
    )
    trained = metrics_by_query(calibration, trained_q, trained_c, frequency, torso_cut, head_cut)
    merged = baseline.merge(
        trained, on=["query_id", "frequency_bucket"], suffixes=("_base", "_trained")
    )
    confidence = bootstrap(
        (merged.ndcg10_trained - merged.ndcg10_base).to_numpy(),
        args.bootstrap_samples,
        args.seed,
    )
    bucket_delta = {}
    for bucket, group in merged.groupby("frequency_bucket"):
        bucket_delta[bucket] = bootstrap(
            (group.ndcg10_trained - group.ndcg10_base).to_numpy(),
            args.bootstrap_samples,
            args.seed,
        )
    payload = {
        "schema": "mind_r7_8_calibration_v1",
        "language": "English",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(),
        "queries": len(merged),
        "frequency_policy": {
            "train_only_positive_item_frequency": True,
            "torso_cut": torso_cut,
            "head_cut": head_cut,
            "unseen_items": "tail",
        },
        "baseline": summarize(baseline, baseline_latency),
        "trained": summarize(trained, trained_latency),
        "paired_ndcg10_delta": confidence,
        "bucket_ndcg10_delta": bucket_delta,
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
        "boundaries": {
            "calibration_accessed_once": True,
            "calibration_used_for_tuning": False,
            "mind_holdout_accessed": False,
            "nfcorpus_locked_test_accessed": False,
        },
        "acceptance": {
            "overall_direction_positive": confidence["mean"] > 0,
            "overall_ci95_excludes_zero": confidence["ci95_low"] > 0,
            "all_frequency_buckets_positive": all(
                value["mean"] > 0 for value in bucket_delta.values()
            ),
            "checkpoint_hash_recorded": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(
        "# R7.8 frozen English MIND calibration evaluation\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    if args.progress:
        print("[3/3] paired bootstrap complete", flush=True)
    print(
        json.dumps(
            {"stage": "complete", "report": str(args.output), "acceptance": payload["acceptance"]}
        )
    )


if __name__ == "__main__":
    main()
