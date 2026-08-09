#!/usr/bin/env python3
"""Low-cost smoke for the frozen public IBM MoLFormer checkpoint."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import transformers.masking_utils as masking_utils
from transformers import AutoModel, AutoTokenizer


def _compat_bidirectional_mask(*, config, inputs_embeds, attention_mask, **_: object) -> torch.Tensor:
    """Compatibility shim for the public checkpoint's newer Transformers API.

    MoLFormer uses linear attention and expects a repeated Bx1xLxL binary mask;
    Transformers 4.57.6 does not yet expose the helper imported by the checkpoint.
    """
    if attention_mask is None:
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
    if attention_mask.ndim == 2:
        length = attention_mask.shape[-1]
        return attention_mask[:, None, None, :].expand(-1, 1, length, -1)
    return attention_mask


if not hasattr(masking_utils, "create_bidirectional_mask"):
    masking_utils.create_bidirectional_mask = _compat_bidirectional_mask


def main() -> None:
    model_name = os.environ.get("MOLFORMER_MODEL", "ibm/MoLFormer-XL-both-10pct")
    revision = os.environ.get("MOLFORMER_REVISION", "7b12d946c181a37f6012b9dc3b002275de070314")
    device = torch.device(os.environ.get("MOLFORMER_DEVICE", "cuda:3"))
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=True).eval().to(device)
    smiles = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"]
    batch = tokenizer(smiles, padding=True, truncation=True, max_length=512, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        output = model(**batch)
        hidden = output.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    result = {
        "model": model_name,
        "revision": revision,
        "device": str(device),
        "shape": list(embedding.shape),
        "finite": bool(torch.isfinite(embedding).all()),
        "norm_mean": float(embedding.norm(dim=1).mean().cpu()),
        "wall_time_seconds": time.perf_counter() - started,
    }
    if not result["finite"] or embedding.ndim != 2 or embedding.shape[0] != len(smiles):
        raise RuntimeError("Invalid MoLFormer smoke embedding")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
