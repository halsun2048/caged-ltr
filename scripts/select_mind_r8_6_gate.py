"""Select on gate-dev and confirm once a deployable MiniLM-to-FIRST route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ndcg(rel: np.ndarray) -> float:
    top = rel[:10]
    ideal = np.sort(rel)[::-1][:10]
    discount = np.log2(np.arange(2, len(top) + 2))
    denominator = np.sum((2**ideal - 1) / discount)
    return float(np.sum((2**top - 1) / discount) / denominator) if denominator else 0.0


def first_frame(pool_path: Path, prompts_path: Path, results_path: Path) -> pd.DataFrame:
    prompts = {json.loads(line)["fingerprint"]: json.loads(line) for line in prompts_path.open()}
    pool = pd.read_parquet(pool_path)
    relevance = pool.set_index(["query_id", "corpus_id"]).relevance
    variants = {}
    features = {}
    for line in results_path.open():
        item = json.loads(line)
        payload = item["payload"]
        prompt = prompts[item["key"]]
        mapping = {row["identifier"]: row["candidate_id"] for row in prompt["candidate_mapping"]}
        ranking = [mapping[value] for value in payload["first_token_ranking"]]
        query_id = str(payload["query_id"])
        variants.setdefault(query_id, {})[payload["variant"]] = ranking
        if payload["variant"] == "baseline":
            features[query_id] = {
                "first_entropy": payload["normalized_entropy"],
                "first_margin": payload["top1_top2_margin"],
                "first_latency_ms": 1000
                * (payload["prefill_seconds"] + payload["decoding_seconds"]),
            }
    rows = []
    for query_id, ranks in variants.items():
        if set(ranks) != {"baseline", "reverse", "random_permutation"}:
            raise RuntimeError(f"incomplete FIRST variants for {query_id}")
        base = ranks["baseline"]
        stability = []
        for variant in ("reverse", "random_permutation"):
            positions = {value: index for index, value in enumerate(ranks[variant])}
            agreements = [
                abs(index - positions[value]) <= 2 for index, value in enumerate(base)
            ]
            stability.append(np.mean(agreements))
        rel = np.array([relevance.get((query_id, value), 0) for value in base])
        relevant = np.flatnonzero(rel > 0)
        rows.append(
            {
                "query_id": query_id,
                "first_ndcg10": ndcg(rel),
                "first_hit10": float(np.any(rel[:10] > 0)),
                "first_mrr": 1 / (int(relevant[0]) + 1) if len(relevant) else 0.0,
                "stability": float(np.mean(stability)),
                **features[query_id],
            }
        )
    return pd.DataFrame(rows)


def merge(
    split: str,
    root: Path,
    pool_root: Path = Path("data/processed/mind_r8_5d"),
    first_root: Path = Path("runs/mind_r8_5d"),
) -> pd.DataFrame:
    first = first_frame(
        pool_root / f"{split}.parquet",
        first_root / f"{split}_prompts.jsonl",
        first_root / f"{split}_first/results.jsonl",
    )
    student = pd.read_parquet(root / f"{split}_query_metrics.parquet")
    return student.merge(first, on="query_id", validate="one_to_one")


def metrics(frame: pd.DataFrame, route: np.ndarray) -> dict[str, float]:
    return {
        "ndcg10": float(np.where(route, frame.first_ndcg10, frame.ndcg10).mean()),
        "hit10": float(np.where(route, frame.first_hit10, frame.hit10).mean()),
        "mrr": float(np.where(route, frame.first_mrr, frame.mrr).mean()),
        "first_call_rate": float(route.mean()),
        "latency_ms": float(
            np.mean(0.67 + np.where(route, frame.first_latency_ms, 0.0))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", type=Path, default=Path("runs/mind_r8_6"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/experiments/mind_r8_6_gate.json")
    )
    parser.add_argument(
        "--model-output", type=Path, default=Path("artifacts/mind_r8_6_gate.joblib")
    )
    args = parser.parse_args()
    dev = merge("gate_dev", args.metrics_root)
    confirm = merge("gate_confirm", args.metrics_root)
    columns = ["margin", "query_characters", "candidate_count"]
    target = (dev.first_ndcg10 > dev.ndcg10).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=20260802, max_iter=500))
    model.fit(dev[columns], target)
    probability = model.predict_proba(dev[columns])[:, 1]
    confirm_probability = model.predict_proba(confirm[columns])[:, 1]
    policies = []
    for budget in (0.1, 0.25, 0.5, 0.75):
        threshold = float(np.quantile(probability, 1 - budget))
        route = probability >= threshold
        policies.append({"budget": budget, "threshold": threshold, **metrics(dev, route)})
    first_quality = float(dev.first_ndcg10.mean())
    eligible = [
        value
        for value in policies
        if value["first_call_rate"] <= 0.5 and value["ndcg10"] >= first_quality - 0.003
    ]
    selected = max(eligible, key=lambda value: value["ndcg10"]) if eligible else None
    confirm_result = None
    admitted = False
    if selected:
        route = confirm_probability >= selected["threshold"]
        confirm_result = metrics(confirm, route)
        admitted = bool(
            confirm_result["first_call_rate"] <= 0.55
            and confirm_result["ndcg10"] >= float(confirm.first_ndcg10.mean()) - 0.003
            and confirm_result["ndcg10"] >= float(confirm.ndcg10.mean())
        )
    payload = {
        "schema": "mind_r8_6_gate_v1",
        "features": columns,
        "dev": {
            "student_ndcg10": float(dev.ndcg10.mean()),
            "first_ndcg10": first_quality,
            "policies": policies,
        },
        "selected": selected,
        "confirm": confirm_result,
        "acceptance": {
            "gate_confirm_passed": admitted,
            "first_call_rate_reduced_at_least_45pct": bool(
                confirm_result and confirm_result["first_call_rate"] <= 0.55
            ),
            "large_test_accessed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": columns, "selected": selected}, args.model_output)
    print(json.dumps({"stage": "complete", "admitted": admitted, "report": str(args.output)}))


if __name__ == "__main__":
    main()
