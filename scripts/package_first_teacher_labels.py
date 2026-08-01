"""Package frozen FIRST JSONL outputs into the standard teacher-label handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-inputs", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompts = {}
    with args.prompt_inputs.open() as f:
        for line in f:
            row = json.loads(line)
            prompts[row["fingerprint"]] = row

    prediction_rows = []
    logit_rows = []
    with args.results.open() as f:
        for line in f:
            record = json.loads(line)
            payload = record["payload"]
            prompt = prompts[record["key"]]
            for item in prompt["candidate_mapping"]:
                ident = item["identifier"]
                logit_rows.append(
                    {
                        "query_id": payload["query_id"],
                        "slate_id": payload["slate_id"],
                        "candidate_id": item["candidate_id"],
                        "logit": float(payload["identifier_logits"][ident]),
                        "input_position": int(item["input_position"]),
                        "identifier": ident,
                        "retrieval_rank": int(item["retrieval_rank"]),
                        "variant": payload["variant"],
                        "prompt_sha256": payload["prompt_sha256"],
                    }
                )
            prediction_rows.append(
                {
                    "query_id": payload["query_id"],
                    "slate_id": payload["slate_id"],
                    "variant": payload["variant"],
                    "ranking": json.dumps(payload["first_token_ranking"]),
                    "normalized_entropy": float(payload["normalized_entropy"]),
                    "top1_top2_margin": float(payload["top1_top2_margin"]),
                    "prefill_seconds": float(payload["prefill_seconds"]),
                    "decoding_seconds": float(payload["decoding_seconds"]),
                    "prompt_sha256": payload["prompt_sha256"],
                    "protocol_fingerprint": record["protocol_fingerprint"],
                }
            )

    logits = pd.DataFrame(logit_rows)
    predictions = pd.DataFrame(prediction_rows)
    logits_path = args.output_dir / "listwise_logits.parquet"
    pred_path = args.output_dir / "teacher_predictions.parquet"
    logits.to_parquet(logits_path, index=False)
    predictions.to_parquet(pred_path, index=False)
    metadata = {
        "teacher_model": args.model,
        "revision": args.revision,
        "prompt_inputs": str(args.prompt_inputs),
        "results": str(args.results),
        "query_records": int(predictions["query_id"].nunique()),
        "prompt_records": len(predictions),
        "logit_records": len(logits),
        "input_sha256": sha256(args.prompt_inputs),
        "results_sha256": sha256(args.results),
        "files": {
            logits_path.name: sha256(logits_path),
            pred_path.name: sha256(pred_path),
        },
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    checksums = "\n".join(f"{v}  {k}" for k, v in metadata["files"].items()) + "\n"
    (args.output_dir / "checksums.sha256").write_text(checksums)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
