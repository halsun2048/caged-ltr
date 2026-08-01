"""Train a lightweight TF-IDF text student and report efficiency/calibration bins."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss


def ndcg(values: list[float]) -> float:
    values = values[:10]
    dcg = sum((2**v - 1) / np.log2(i + 2) for i, v in enumerate(values))
    ideal = sorted(values, reverse=True)
    idcg = sum((2**v - 1) / np.log2(i + 2) for i, v in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--qrels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    labels = pd.read_parquet(args.labels)
    labels = labels[labels.variant == "baseline"]
    candidates = pd.read_parquet(args.candidates)[["query_id", "passage_id", "passage"]].rename(
        columns={"passage_id": "candidate_id"}
    )
    data = labels.merge(candidates, on=["query_id", "candidate_id"], how="inner")
    qrels = pd.read_parquet(args.qrels).set_index(["query_id", "passage_id"])["graded_relevance"]
    queries = sorted(data.query_id.unique(), key=lambda q: hashlib.sha256(q.encode()).hexdigest())
    train_q = set(queries[: int(0.8 * len(queries))])
    train = data[data.query_id.isin(train_q)].copy()
    valid = data[~data.query_id.isin(train_q)].copy()
    start = time.perf_counter()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train.passage.fillna(""))
    X_valid = vectorizer.transform(valid.passage.fillna(""))
    model = Ridge(alpha=10.0)
    model.fit(X_train, train.logit)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    valid["student_score"] = model.predict(X_valid)
    score_seconds = time.perf_counter() - start
    valid["relevance"] = [
        float(qrels.get((q, p), 0.0))
        for q, p in zip(valid.query_id, valid.candidate_id, strict=False)
    ]
    per_query = valid.groupby("query_id").apply(
        lambda g: ndcg(g.sort_values("student_score", ascending=False).relevance.tolist()),
        include_groups=False,
    )
    teacher_query = valid.groupby("query_id").apply(
        lambda g: ndcg(g.sort_values("logit", ascending=False).relevance.tolist()),
        include_groups=False,
    )
    valid["item_frequency"] = valid.groupby("candidate_id")["query_id"].transform("count")
    valid["bucket"] = pd.qcut(
        valid.item_frequency, 3, labels=["tail", "torso", "head"], duplicates="drop"
    )
    bucket = {}
    for name, g in valid.groupby("bucket", observed=True):
        bucket[str(name)] = {"rows": len(g), "student_mean_score": float(g.student_score.mean())}
    probs = 1 / (1 + np.exp(-valid.student_score.clip(-20, 20)))
    y = (valid.relevance > 0).astype(int)
    result = {
        "dataset": "NFCorpus independent train split",
        "train_queries": len(train_q),
        "valid_queries": len(queries) - len(train_q),
        "student": "TF-IDF(1-2gram,50k)+Ridge",
        "features": int(X_train.shape[1]),
        "fit_seconds": fit_seconds,
        "valid_score_seconds": score_seconds,
        "student_valid_ndcg10": float(per_query.mean()),
        "teacher_valid_ndcg10": float(teacher_query.mean()),
        "brier_score": float(brier_score_loss(y, probs)),
        "head_torso_tail": bucket,
        "no_test_access": True,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
