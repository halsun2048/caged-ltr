"""Build the R14 bounded-experiment closeout without reopening locked tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    stats = load(Path("reports/experiments/mind_r14_statistics.json"))
    latency = load(Path("reports/experiments/mind_r14_service_latency.json"))
    cross = load(Path("reports/experiments/mind_r14_cross_domain_audit.json"))
    dual_path = Path("reports/experiments/mind_r14_4_dual_granularity_pilot.json")
    dual = load(dual_path)
    payload = {
        "schema": "mind_r14_closeout_v1",
        "status": "bounded_closeout_complete",
        "decisions": {
            "R14.0_asset_freeze": "complete",
            "R14.1_paired_statistics": "not_available_without_reconstructing_comparators",
            "R14.2_service_latency": "complete",
            "R14.3_cross_domain": cross["decision"]["status"],
            "R14.4_dual_granularity": "no_go" if not dual["go"] else "go",
        },
        "evidence": {
            "statistics": stats,
            "service_latency": latency,
            "cross_domain": cross,
            "dual_granularity_audit": dual,
        },
        "final_interpretation": {
            "supported_claim": "Tail-safe budgeted routing between a low-latency MiniLM student and FIRST preserves ranking quality with substantially fewer teacher calls.",
            "not_supported": [
                "A statistically significant large-test advantage over FIRST, because comparator per-query arrays were not persisted.",
                "Strict zero-shot transfer of the same gate thresholds across datasets.",
                "A dual-granularity pairwise-plus-listwise distillation gain: the 2,000-query, two-seed pilot produced a negative mean gain and Tail reversal.",
            ],
            "recommendation": "Stop broad model exploration; freeze the routing result and report these boundaries explicitly.",
        },
        "boundaries": {
            "large_test_reopened": False,
            "r12_confirm_reopened": False,
            "nfcorpus_test_reopened": False,
        },
        "source_sha256": {
            str(path): sha(path)
            for path in (
                Path("reports/experiments/mind_r14_statistics.json"),
                Path("reports/experiments/mind_r14_service_latency.json"),
                Path("reports/experiments/mind_r14_cross_domain_audit.json"),
                dual_path,
            )
        },
    }
    out = Path("reports/experiments/mind_r14_closeout.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "report": str(out), "status": payload["status"]}))


if __name__ == "__main__":
    main()
