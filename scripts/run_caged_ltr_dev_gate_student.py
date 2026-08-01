"""Select a dev-only confidence gate and train a tiny CPU student."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


def ndcg(labels: list[float], k: int = 10) -> float:
    labels = labels[:k]
    dcg = sum((2**v - 1) / np.log2(i + 2) for i, v in enumerate(labels))
    ideal = sorted(labels, reverse=True)
    idcg = sum((2**v - 1) / np.log2(i + 2) for i, v in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-inputs", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--qrels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    prompts = {}
    with args.prompt_inputs.open() as f:
        for line in f:
            x = json.loads(line)
            if x["variant"] == "baseline":
                prompts[x["fingerprint"]] = x
    pd.read_parquet(args.candidates)  # validate the frozen candidate artifact
    qrels = pd.read_parquet(args.qrels).set_index(["query_id", "passage_id"])["graded_relevance"]
    rows = []
    with args.results.open() as f:
        for line in f:
            x = json.loads(line)
            p = x["payload"]
            if p["variant"] != "baseline":
                continue
            prompt = prompts[x["key"]]
            logits = p["identifier_logits"]
            rank = {pid: i for i, pid in enumerate(p["first_token_ranking"])}
            for m in prompt["candidate_mapping"]:
                pid = m["candidate_id"]
                rows.append(
                    {
                        "query_id": p["query_id"],
                        "passage_id": pid,
                        "teacher_logit": float(logits[m["identifier"]]),
                        "teacher_rank": rank[m["identifier"]],
                        "bm25_rank": m["retrieval_rank"],
                        "entropy": float(p["normalized_entropy"]),
                        "margin": float(p["top1_top2_margin"]),
                        "relevance": float(qrels.get((p["query_id"], pid), 0.0)),
                    }
                )
    frame = pd.DataFrame(rows)
    grouped = {q: g for q, g in frame.groupby("query_id")}

    def score_query(g: pd.DataFrame, mode: str, threshold: float = 0.0) -> float:
        use_teacher = mode == "teacher" or (
            mode == "gate" and float(g["entropy"].iloc[0]) <= threshold
        )
        col = "teacher_logit" if use_teacher else "bm25_rank"
        order = g.sort_values(col, ascending=(col == "bm25_rank"))["relevance"].tolist()
        return ndcg(order)

    query_ids = sorted(grouped, key=lambda q: hashlib.sha256(q.encode()).hexdigest())
    split = set(query_ids[: int(0.8 * len(query_ids))])
    train = frame[frame.query_id.isin(split)].copy()
    valid = frame[~frame.query_id.isin(split)].copy()
    train_grouped = {q: g for q, g in train.groupby("query_id")}
    valid_grouped = {q: g for q, g in valid.groupby("query_id")}
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    threshold_scores = [
        {
            "threshold": t,
            "train_ndcg10": float(
                np.mean([score_query(g, "gate", t) for g in train_grouped.values()])
            ),
        }
        for t in thresholds
    ]
    best = max(threshold_scores, key=lambda x: x["train_ndcg10"])
    valid_gate_ndcg = float(
        np.mean([score_query(g, "gate", best["threshold"]) for g in valid_grouped.values()])
    )
    features = ["teacher_logit", "teacher_rank", "bm25_rank", "entropy", "margin"]
    mu, sigma = train[features].mean(), train[features].std().replace(0, 1)
    X = torch.tensor(((train[features] - mu) / sigma).to_numpy(), dtype=torch.float32)
    y = torch.tensor(train.relevance.to_numpy(), dtype=torch.float32)
    model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(100):
        pred = model(X).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        valid["student_score"] = (
            model(torch.tensor(((valid[features] - mu) / sigma).to_numpy(), dtype=torch.float32))
            .squeeze(-1)
            .numpy()
        )
    student_ndcg = float(
        np.mean(
            [
                ndcg(g.sort_values("student_score", ascending=False).relevance.tolist())
                for _, g in valid.groupby("query_id")
            ]
        )
    )
    result = {
        "split": "dev_only",
        "queries": len(grouped),
        "train_queries": len(split),
        "valid_queries": len(grouped) - len(split),
        "threshold_candidates": threshold_scores,
        "selected_entropy_threshold": best["threshold"],
        "selected_gate_train_ndcg10": best["train_ndcg10"],
        "selected_gate_valid_ndcg10": valid_gate_ndcg,
        "student_valid_ndcg10": student_ndcg,
        "student_features": features,
        "test_accessed": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
