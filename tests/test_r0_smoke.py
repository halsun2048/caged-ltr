from __future__ import annotations

import json

import pandas as pd

from caged_ltr.r0_smoke import run_smoke


def test_r0_smoke_generates_raw_predictions_and_tables(tmp_path) -> None:
    summary = run_smoke(tmp_path, seed=42)

    assert summary["models"] == ["mlp", "dcn_v2", "lambda_mart"]
    assert len(list(tmp_path.glob("*_predictions.parquet"))) == 3
    assert len(list(tmp_path.glob("*_reliability.png"))) == 3
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert set(metrics["model"]) == {"mlp", "dcn_v2", "lambda_mart"}
    assert set(metrics["bucket"]) == {"Overall", "head", "torso", "tail"}
    environment = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    assert environment["git"]["commit"]
