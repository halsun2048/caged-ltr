"""Audit R16 serving assets and write a reproducible manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(path: Path, required: bool) -> dict[str, object]:
    entry: dict[str, object] = {"path": str(path), "required": required, "exists": path.is_file()}
    if path.is_file():
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256(path)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--first-results", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/experiments/r16_asset_manifest.json"))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    model_files = [args.model / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json")]
    paths = model_files + args.checkpoint + [args.first_results]
    if args.progress:
        print(f"[R16 1/3] auditing {len(paths)} required assets", flush=True)
    entries = [file_entry(path, True) for path in paths]
    missing = [entry["path"] for entry in entries if not entry["exists"]]
    if args.progress:
        print(f"[R16 2/3] hashed {sum(bool(entry['exists']) for entry in entries)}/{len(entries)} assets", flush=True)
    cuda = {"available": False}
    try:
        import torch

        cuda = {"available": bool(torch.cuda.is_available()), "torch": torch.__version__}
        if cuda["available"]:
            cuda.update({"device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())})
    except Exception as error:  # pragma: no cover - audit must report environment failures
        cuda["error"] = repr(error)
    payload = {
        "schema": "caged_ltr_r16_asset_manifest_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "assets": entries,
        "missing": missing,
        "cuda": cuda,
        "admission": {"all_required_assets_present": not missing, "cuda_available": cuda["available"], "passed": not missing and cuda["available"]},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if args.progress:
        print("[R16 3/3] asset audit complete", flush=True)
    print(json.dumps({"stage": "complete", "passed": payload["admission"]["passed"], "report": str(args.report)}))


if __name__ == "__main__":
    main()
