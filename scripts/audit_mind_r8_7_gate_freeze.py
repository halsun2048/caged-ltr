"""Audit the deployability and reproducibility of the frozen R8 gain gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate-report", type=Path, default=Path("reports/experiments/mind_r8_6_gate_v2.json")
    )
    parser.add_argument(
        "--gate-model", type=Path, default=Path("artifacts/mind_r8_6_gate_v2.joblib")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/mind_r8_6_gate_v2.manifest.json")
    )
    parser.add_argument(
        "--large-test-guard", type=Path, default=Path("artifacts/mind_r8_0_large_test_guard.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/experiments/mind_r8_7_gate_freeze_audit.json")
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    gate = json.loads(args.gate_report.read_text())
    manifest = json.loads(args.manifest.read_text())
    guard = json.loads(args.large_test_guard.read_text())
    frozen = joblib.load(args.gate_model)
    if args.progress:
        print("[1/4] loaded frozen gate and metadata", flush=True)
    selected = gate["selected"]
    features = gate["features"]
    forbidden = [value for value in features if value.startswith("first_") or "relevance" in value]
    checks = {
        "artifact_hash_matches_manifest": sha256(args.gate_model) == manifest["sha256"],
        "artifact_size_matches_manifest": args.gate_model.stat().st_size == manifest["bytes"],
        "model_type_frozen": selected["model"] == manifest["model"] == "extra_trees_leaf10",
        "features_match_artifact": frozen["features"] == features,
        "threshold_matches_artifact": (
            frozen["selected"]["threshold"] == selected["threshold"] == manifest["frozen_threshold"]
        ),
        "features_are_pre_first": not forbidden,
        "confirm_gate_passed": gate["acceptance"]["gate_confirm_passed"],
        "first_call_reduction_passed": gate["acceptance"][
            "first_call_rate_reduced_at_least_45pct"
        ],
        "large_test_locked_unaccessed": (
            guard["status"] == "locked_unaccessed" and guard["evaluation_count"] == 0
        ),
    }
    if args.progress:
        print("[2/4] checked model identity, feature order, and absolute threshold", flush=True)
        print("[3/4] checked pre-FIRST deployability and large-test guard", flush=True)
    payload = {
        "schema": "mind_r8_7_gate_freeze_audit_v1",
        "model": selected["model"],
        "features": features,
        "threshold": selected["threshold"],
        "artifact": {
            "path": str(args.gate_model),
            "bytes": args.gate_model.stat().st_size,
            "sha256": sha256(args.gate_model),
        },
        "efficiency": {
            "dev_first_call_rate": selected["first_call_rate"],
            "dev_latency_ms": selected["latency_ms"],
            "confirm_first_call_rate": gate["confirm"]["first_call_rate"],
            "confirm_latency_ms": gate["confirm"]["latency_ms"],
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    if not payload["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"R8.7 freeze audit failed: {failed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.progress:
        print("[4/4] audit passed; frozen metadata report written", flush=True)
    print(json.dumps({"stage": "complete", "report": str(args.output)}))


if __name__ == "__main__":
    main()
