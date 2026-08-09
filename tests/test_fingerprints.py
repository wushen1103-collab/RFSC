from __future__ import annotations

import numpy as np
import pytest

from bps_gnn.fingerprints import (
    fit_clean_rdkit_descriptors,
    multifingerprint_4263_matrix,
    rdkit_descriptor_matrix,
)


def test_multifingerprint_4263_dimensions_and_binary_values() -> None:
    features = multifingerprint_4263_matrix(["CCO", "c1ccccc1Cl"])
    assert features.shape == (2, 4263)
    assert features.dtype == np.uint8
    assert set(np.unique(features)).issubset({0, 1})
    assert not np.array_equal(features[0], features[1])


def test_multifingerprint_rejects_invalid_smiles() -> None:
    with pytest.raises(ValueError, match="Invalid standardized SMILES"):
        multifingerprint_4263_matrix(["not-a-smiles"])


def test_rdkit_descriptors_are_stable_and_preprocessed_train_only() -> None:
    train, names = rdkit_descriptor_matrix(["CCO", "c1ccccc1Cl"])
    validation, validation_names = rdkit_descriptor_matrix(["CC(=O)O"])
    assert names == validation_names
    assert train.shape[0] == 2
    assert train.shape[1] >= 200
    clean_train, clean_validation, medians, usable = fit_clean_rdkit_descriptors(
        train, validation
    )
    assert clean_train.shape == (2, int(usable.sum()))
    assert clean_validation.shape == (1, int(usable.sum()))
    assert medians.shape == (train.shape[1],)
    assert np.isfinite(clean_train).all()
    assert np.isfinite(clean_validation).all()
    assert np.all(np.ptp(clean_train, axis=0) > 0.0)


def test_rdkit_descriptors_reject_invalid_smiles() -> None:
    with pytest.raises(ValueError, match="Invalid standardized SMILES"):
        rdkit_descriptor_matrix(["not-a-smiles"])


def test_descriptor_cleaning_caps_values_for_sklearn_float32() -> None:
    train = np.asarray([[1.0e300, 0.0], [2.0e300, 1.0]])
    validation = np.asarray([[3.0e300, 0.5]])
    clean_train, clean_validation, _, _ = fit_clean_rdkit_descriptors(
        train, validation
    )
    assert np.isfinite(clean_train.astype(np.float32)).all()
    assert np.isfinite(clean_validation.astype(np.float32)).all()
