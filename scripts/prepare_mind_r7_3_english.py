"""Audit MIND candidates and admit only reproducible English behavior data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ENGLISH_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "is",
    "are",
    "was",
    "were",
    "this",
    "that",
    "from",
    "by",
    "as",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_columns(frame: pd.DataFrame) -> list[str]:
    preferred = ("query", "positive", "negative", "text", "title", "abstract", "history")
    columns = [name for name in preferred if name in frame.columns]
    if columns:
        return columns
    return [name for name in frame.columns if frame[name].dtype == object][:3]


def english_score(values: list[str]) -> dict[str, float | int]:
    sample = " ".join(values)[:1_000_000]
    tokens = re.findall(r"[A-Za-z]+", sample.lower())
    latin = sum(character.isascii() and character.isalpha() for character in sample)
    alphabetic = sum(character.isalpha() for character in sample)
    stopwords = sum(token in ENGLISH_WORDS for token in tokens)
    return {
        "sample_characters": len(sample),
        "latin_alphabetic_ratio": latin / max(alphabetic, 1),
        "english_stopword_ratio": stopwords / max(len(tokens), 1),
        "tokens": len(tokens),
    }


def inspect(path: Path) -> dict[str, object]:
    record: dict[str, object] = {"path": str(path), "bytes": path.stat().st_size}
    try:
        frame = pd.read_parquet(path)
    except Exception as error:  # corrupted/incomplete downloads are expected here
        return {**record, "valid_parquet": False, "admitted": False, "error": str(error)}
    columns = text_columns(frame)
    values: list[str] = []
    for column in columns:
        values.extend(frame[column].dropna().astype(str).head(2_000).tolist())
    language = english_score(values)
    is_english = (
        language["sample_characters"] >= 1_000
        and language["latin_alphabetic_ratio"] >= 0.95
        and language["english_stopword_ratio"] >= 0.015
    )
    behavior_fields = [
        name
        for name in frame.columns
        if any(
            key in name.lower()
            for key in ("score", "label", "relevance", "positive", "negative", "rank")
        )
    ]
    return {
        **record,
        "sha256": sha256(path),
        "valid_parquet": True,
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "text_columns_checked": columns,
        "behavior_fields": behavior_fields,
        "language": language,
        "english": is_english,
        "admitted": bool(is_english and behavior_fields and len(frame) >= 1_000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/external/mind"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/data/mind_r7_3_english_audit.json")
    )
    parser.add_argument(
        "--bundle-report",
        type=Path,
        default=Path("reports/data/mind_r7_5_english_bundle.json"),
    )
    args = parser.parse_args()
    official = args.root / "official_small"
    candidates = sorted(official.glob("*.parquet"))
    rejected_language_variants = sorted(
        path
        for path in args.root.glob("*/*")
        if path.is_file() and "official_small" not in path.parts
    )
    inspected = [inspect(path) for path in candidates]
    bundle = json.loads(args.bundle_report.read_text()) if args.bundle_report.exists() else None
    bundle_admitted = bool(
        bundle and all(bundle["acceptance"].values()) and bundle["language"]["english_gate_passed"]
    )
    admitted = bundle_admitted or any(item["admitted"] for item in inspected)
    payload = {
        "schema": "mind_r7_3_english_admission_v1",
        "language_policy": {
            "required_language": "English",
            "reason": (
                "NFCorpus, FIRST prompts, and the MiniLM distillation/evaluation "
                "pipeline are English."
            ),
            "translated_or_non_english_data_allowed": False,
        },
        "preferred_source": {
            "dataset": "mteb/MindSmallReranking",
            "provenance": "English MIND-small reranking derivative",
            "use": "external behavior pretraining/evaluation only; never NFCorpus final evidence",
        },
        "official_candidates": inspected,
        "validated_relational_bundle": bundle,
        "explicitly_rejected_non_official_files": [
            str(path) for path in rejected_language_variants
        ],
        "download_status": "complete" if admitted else "blocked",
        "blocker": None
        if admitted
        else (
            "No complete English behavior shard is locally available; direct Microsoft "
            "returned HTTP 409 and the public mirror transfer did not complete."
        ),
        "acceptance": {
            "english_only_policy_frozen": True,
            "at_least_one_complete_english_behavior_shard": admitted,
            "complete_relational_bundle": bundle_admitted,
            "non_english_data_used": False,
            "nfcorpus_test_accessed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(
        "# R7.3 English behavior-data admission\n\n"
        "All model inputs and external behavior data are frozen to **English**. "
        "Non-English and translated MIND variants are rejected.\n\n"
        "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```\n"
    )
    print(json.dumps({"stage": payload["download_status"], "acceptance": payload["acceptance"]}))


if __name__ == "__main__":
    main()
