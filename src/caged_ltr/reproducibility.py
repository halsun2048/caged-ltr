"""Utilities for recording reproducible local execution metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

TRACKED_PACKAGES = (
    "caged-ltr",
    "datasets",
    "duckdb",
    "lightgbm",
    "matplotlib",
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "torch",
    "transformers",
)


def seed_everything(seed: int, *, deterministic_algorithms: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for a repeatable local run."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_first_matching_line(path: Path, prefix: str) -> str | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(prefix):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        return None
    return None


def _git_value(project_root: Path, *args: str) -> str | None:
    return _command_value("git", *args, cwd=project_root)


def _command_value(*args: str, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = Path("/etc/os-release").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key.lower()] = value.strip().strip('"')
    return values


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def collect_environment(project_root: Path) -> dict[str, Any]:
    """Collect stable machine, runtime, dependency, and repository metadata."""
    root = project_root.resolve()
    memory_kib = _read_first_matching_line(Path("/proc/meminfo"), "MemTotal:")
    disk = shutil.disk_usage(root)
    commit = _git_value(root, "rev-parse", "HEAD")
    status = _git_value(root, "status", "--porcelain") if commit else None
    gpu_controller = _command_value("lspci", "-nnk", "-d", "::0300")
    mesa_packages = _command_value(
        "dpkg-query",
        "-W",
        "-f=${binary:Package}=${Version}\\n",
        "libgl1-mesa-dri",
        "mesa-vulkan-drivers",
        "xserver-xorg-video-amdgpu",
    )

    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "hip_version": torch.version.hip,
        }
    except ImportError:
        torch_info = {"version": None}

    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "project_root": str(root),
        "platform": {
            "os_release": _os_release(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "cpu_model": _read_first_matching_line(Path("/proc/cpuinfo"), "model name"),
            "logical_cpu_count": os.cpu_count(),
            "memory_total_kib": memory_kib,
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "accelerator": {
            "pci_display_controller": gpu_controller,
            "linux_graphics_packages": mesa_packages,
            "uv_version": _command_value("uv", "--version"),
        },
        "git": {
            "commit": commit,
            "dirty": bool(status) if commit else None,
        },
        "packages": _package_versions(),
        "torch_runtime": torch_info,
    }


def write_environment(output: Path, project_root: Path) -> None:
    """Write an environment snapshot as deterministic, human-readable JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_environment(project_root)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/environment/local_baseline.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    write_environment(args.output, args.project_root)
    print(f"Wrote environment snapshot to {args.output}")


if __name__ == "__main__":
    main()
