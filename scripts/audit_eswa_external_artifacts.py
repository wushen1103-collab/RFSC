#!/usr/bin/env python3
"""Fail closed on every new ESWA external prediction/risk manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOTS = [
    Path("experiments/eswa_aqsol_external/predictions"),
    Path("experiments/eswa_response_free_risk/aqsol_scores"),
    Path("experiments/eswa_qm9_direct_pair/full"),
    Path("experiments/eswa_qm9_structured_missingness/full"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, total = [], 0
    for root in ROOTS:
        manifests = sorted(root.rglob("*.sha256.json"))
        if not manifests:
            raise RuntimeError(f"No manifests under {root}")
        for manifest in manifests:
            artifact = Path(str(manifest)[: -len(".sha256.json")])
            record = json.loads(manifest.read_text(encoding="utf-8"))
            expected = record.get("prediction_sha256") or record.get("risk_score_sha256")
            if not isinstance(expected, str) or not artifact.exists() or sha256_file(artifact) != expected:
                raise RuntimeError(f"Artifact hash mismatch: {artifact}")
        rows.append({"root": str(root), "manifests": len(manifests)})
        total += len(manifests)
    report = {"roots": rows, "total_manifests_verified": total, "all_passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
