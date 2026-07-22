from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from caged_ltr.reproducibility import collect_environment, sha256_file, write_environment


def test_sha256_file(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"caged-ltr\n")

    assert sha256_file(sample) == hashlib.sha256(b"caged-ltr\n").hexdigest()


def test_collect_environment_contains_reproducibility_fields() -> None:
    environment = collect_environment(Path.cwd())

    assert environment["platform"]["python"]
    assert environment["platform"]["cpu_model"]
    assert environment["platform"]["disk_free_bytes"] > 50 * 1024**3
    assert "os_release" in environment["platform"]
    assert "pci_display_controller" in environment["accelerator"]
    assert "torch" in environment["packages"]
    assert "commit" in environment["git"]


def test_write_environment_creates_json(tmp_path: Path) -> None:
    output = tmp_path / "environment.json"

    write_environment(output, Path.cwd())

    assert output.read_text(encoding="utf-8").endswith("\n")
    assert '"captured_at_utc"' in output.read_text(encoding="utf-8")


def test_base_config_declares_required_seed_counts() -> None:
    config = yaml.safe_load(Path("configs/reproduction/base.yaml").read_text(encoding="utf-8"))

    assert len(config["reproducibility"]["seeds"]) >= 3
    assert len(config["reproducibility"]["core_seeds"]) >= 5
