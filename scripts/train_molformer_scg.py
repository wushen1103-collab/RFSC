#!/usr/bin/env python3
"""Validation-only frozen MoLFormer regression under the SCG protocol.

The checkpoint is used as a frozen molecular representation.  A single fixed
linear probe (Ridge, alpha=1) is fit using only the outer-training partition;
validation and outer-test identifiers are used only to materialize predictions
and are never read for labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoTokenizer

import transformers.masking_utils as masking_utils

from bps_gnn.io import freeze_prediction_file, read_labels_for_ids, read_structure_table


def _compat_bidirectional_mask(*, config, inputs_embeds, attention_mask, **_: object) -> torch.Tensor:
    """Compatibility shim for the checkpoint's newer Transformers helper."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-validation", action="store_true")
    return parser.parse_args()


def split_ids(target: str, fold: int) -> tuple[list[str], list[str], list[str]]:
    assignments = pd.read_parquet(Path("folds/scg") / target / "outer_folds.parquet")
    n_folds = int(assignments["outer_fold"].nunique())
    validation_fold = (fold + 1) % n_folds
    train_ids = assignments.loc[~assignments["outer_fold"].isin([fold, validation_fold]), "mol_id"].tolist()
    valid_ids = assignments.loc[assignments["outer_fold"] == validation_fold, "mol_id"].tolist()
    test_ids = assignments.loc[assignments["outer_fold"] == fold, "mol_id"].tolist()
    if set(train_ids) & set(valid_ids) or set(train_ids) & set(test_ids) or set(valid_ids) & set(test_ids):
        raise RuntimeError("SCG partition overlap")
    return train_ids, valid_ids, test_ids


def embed_smiles(
    smiles: list[str], tokenizer, model, device: torch.device, batch_size: int, max_length: int
) -> np.ndarray:
    output_features: list[np.ndarray] = []
    for start in range(0, len(smiles), batch_size):
        batch_smiles = smiles[start : start + batch_size]
        tokens = tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            output = model(**tokens)
            if getattr(output, "pooler_output", None) is not None:
                features = output.pooler_output
            else:
                hidden = output.last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                features = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        output_features.append(features.float().cpu().numpy())
    features = np.vstack(output_features).astype(np.float32)
    if features.shape[0] != len(smiles) or not np.isfinite(features).all():
        raise RuntimeError("Invalid MoLFormer embedding batch")
    return features


def standardize_fit_query(x_fit: np.ndarray, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = x_fit.mean(axis=0, keepdims=True)
    scales = x_fit.std(axis=0, keepdims=True)
    scales[scales < 1e-8] = 1.0
    return np.clip((x_fit - means) / scales, -6.0, 6.0), np.clip((x_query - means) / scales, -6.0, 6.0)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    if not (args.smoke ^ args.full_validation):
        raise ValueError("Select exactly one of --smoke or --full-validation")
    authorized_tasks = evaluation["smoke_tasks"] if args.smoke else evaluation["full_validation_tasks"]
    if not any(
        args.target == task["target"] and args.fold == int(task["fold"]) and args.seed == int(task["seed"])
        for task in authorized_tasks
    ):
        raise ValueError("Run is outside the preregistered MoLFormer task list")

    structures = read_structure_table(Path(config["data_root"]), args.target).set_index("mol_id")
    train_ids, valid_ids, test_ids = split_ids(args.target, args.fold)
    if args.smoke:
        fit_ids = train_ids[: int(evaluation["smoke_train_size"])]
        query_ids = train_ids[
            int(evaluation["smoke_train_size"]) : int(evaluation["smoke_train_size"]) + int(evaluation["smoke_query_size"])
        ]
    else:
        fit_ids, query_ids = train_ids, valid_ids
    labels = read_labels_for_ids(Path(config["data_root"]), args.target, fit_ids).set_index("mol_id")
    if set(labels.index) & set(valid_ids + test_ids):
        raise RuntimeError("Labels escaped the SCG training partition")

    model_name = str(config["molformer"]["model"])
    revision = str(config["molformer"]["revision"])
    device = torch.device(config["molformer"]["device"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=True).eval().to(device)
    ordered_ids = [*fit_ids, *query_ids]
    smiles = structures.loc[ordered_ids, "canonical_isomeric_smiles"].tolist()
    features = embed_smiles(
        smiles,
        tokenizer,
        model,
        device,
        int(config["molformer"]["batch_size"]),
        int(config["molformer"]["max_length"]),
    )
    x_fit, x_query = standardize_fit_query(features[: len(fit_ids)], features[len(fit_ids) :])
    y_fit = labels.loc[fit_ids, "y_pactivity"].to_numpy(dtype=np.float64)
    y_mean = float(y_fit.mean())
    y_scale = float(y_fit.std())
    if y_scale < 1e-8:
        y_scale = 1.0
    model_probe = Ridge(alpha=float(config["probe"]["alpha"]), fit_intercept=True)
    model_probe.fit(x_fit, (y_fit - y_mean) / y_scale)
    prediction = np.asarray(model_probe.predict(x_query) * y_scale + y_mean, dtype=np.float32)
    if prediction.shape != (len(query_ids),) or not np.isfinite(prediction).all():
        raise RuntimeError("MoLFormer prediction is invalid")

    output = args.output_dir / args.target / f"fold_{args.fold}" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "target": args.target,
        "fold": args.fold,
        "seed": args.seed,
        "method": config["method"],
        "protocol": config["protocol"],
        "smoke": bool(args.smoke),
        "n_train": len(train_ids),
        "n_validation": len(valid_ids),
        "n_test": len(test_ids),
        "n_fit_used": len(fit_ids),
        "n_query_used": len(query_ids),
        "feature_dimension": int(features.shape[1]),
        "feature_preprocessing": "train_only_standard_scale_then_clip[-6,6]",
        "target_preprocessing": "train_only_standard_scale_then_inverse",
        "probe": "Ridge(alpha=1.0)",
        "molformer_model": model_name,
        "molformer_revision": revision,
        "trust_remote_code": True,
        "compatibility_shim": "binary_repeated_bidirectional_mask_for_transformers_4_57_6",
        "validation_labels_materialized": False,
        "outer_test_labels_materialized": False,
        "outer_test_predictions_generated": False,
        "validation_only": True,
    }
    prediction_name = "smoke_predictions.parquet" if args.smoke else "validation_predictions.parquet"
    metadata["prediction_sha256"] = freeze_prediction_file(
        pd.DataFrame({"mol_id": query_ids, "prediction": prediction}), output / prediction_name, metadata
    )
    metadata["wall_time_seconds"] = time.perf_counter() - started
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
