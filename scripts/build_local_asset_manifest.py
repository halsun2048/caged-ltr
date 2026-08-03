"""Write hashes for local model/checkpoint assets without copying them to Git."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "data" / "local_asset_manifest.json"
TARGETS = [
    ROOT / "artifacts" / "models" / "all-MiniLM-L6-v2",
    ROOT / "artifacts" / "r16_runtime" / "mind_r13_reweight_mild.pt",
    ROOT / "runs" / "mind_r10_0" / "dev_first" / "results.jsonl",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    assets = []
    for target in TARGETS:
        if not target.exists():
            assets.append({"path": str(target.relative_to(ROOT)), "status": "missing"})
            continue
        paths = sorted(p for p in target.rglob("*") if p.is_file()) if target.is_dir() else [target]
        for path in paths:
            assets.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    report = {
        "schema": "caged_ltr_local_asset_manifest_v1",
        "status": "complete",
        "assets": assets,
        "note": "Large assets remain local and are intentionally excluded from Git.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "complete", "report": str(OUT), "assets": len(assets)}))


if __name__ == "__main__":
    main()
