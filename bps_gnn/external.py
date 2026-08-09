from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_acanet_loss(config: dict):
    """Runtime-load GPL ACALoss without copying external source into this repo."""
    external = config["external"]
    root = Path(external["source_root"])
    observed_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed_commit != external["commit"]:
        raise RuntimeError(
            f"ACANet commit mismatch: {observed_commit} != {external['commit']}"
        )
    license_path = root / "LICENSE.txt"
    if sha256_file(license_path) != external["license_sha256"]:
        raise RuntimeError("ACANet license hash mismatch")
    source = root / "clsar/model/loss.py"
    spec = importlib.util.spec_from_file_location("external_acanet_v3_loss", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load external ACANet loss module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ACALoss(**config["aca_loss"])

