#!/usr/bin/env python3
"""One-molecule CheMeleon fingerprint smoke test.

The model construction follows the authors' MIT-licensed
``chemeleon_fingerprint.py`` reference implementation.  This script keeps the
checkpoint path explicit so a stale ``~/.chemprop`` symlink cannot be used by
accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit.Chem import MolFromSmiles


def build_fingerprint_model(checkpoint: Path, device: torch.device) -> MPNN:
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    message_passing = nn.BondMessagePassing(**payload["hyper_parameters"])
    message_passing.load_state_dict(payload["state_dict"])
    model = MPNN(
        message_passing=message_passing,
        agg=nn.MeanAggregation(),
        predictor=RegressionFFN(input_dim=message_passing.output_dim),
    )
    model.eval().to(device=device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    device = torch.device(args.device)
    model = build_fingerprint_model(checkpoint, device)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    smiles = ["CCO", "c1ccccc1O"]
    graph = BatchMolGraph([featurizer(MolFromSmiles(smi)) for smi in smiles])
    graph.to(device=device)
    with torch.no_grad():
        fingerprint = model.fingerprint(graph).detach().cpu().numpy()

    result = {
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": digest,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "shape": list(fingerprint.shape),
        "dtype": str(fingerprint.dtype),
        "finite": bool(np.isfinite(fingerprint).all()),
        "row_norms": np.linalg.norm(fingerprint, axis=1).round(6).tolist(),
        "pair_l2": float(np.linalg.norm(fingerprint[0] - fingerprint[1])),
    }
    if result["shape"] != [len(smiles), model.message_passing.output_dim]:
        raise RuntimeError(result)
    if not result["finite"] or not np.isfinite(result["pair_l2"]):
        raise RuntimeError(result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
