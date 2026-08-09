#!/usr/bin/env python3
"""Tiny TabPFN model-download/inference smoke; no project data is touched."""

from __future__ import annotations

import json

import numpy as np
import torch
from tabpfn import TabPFNRegressor


def main() -> None:
    x = np.arange(40, dtype=np.float32).reshape(10, 4) / 10.0
    y = (x[:, 0] - 0.25 * x[:, 1] + np.sin(x[:, 2])).astype(np.float32)
    model = TabPFNRegressor(
        n_estimators=2,
        device="cuda",
        ignore_pretraining_limits=True,
        random_state=42,
        n_preprocessing_jobs=1,
    )
    model.fit(x, y)
    pred = np.asarray(model.predict(x))
    result = {
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0),
        "shape": list(pred.shape),
        "finite": bool(np.isfinite(pred).all()),
        "mean": float(pred.mean()),
        "std": float(pred.std()),
    }
    if result["shape"] != [len(y)] or not result["finite"]:
        raise RuntimeError(result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
