from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import (
    AllChem,
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,
    rdMolDescriptors,
)


def ecfp_matrix(smiles: list[str], radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    matrix = np.empty((len(smiles), n_bits), dtype=np.uint8)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid standardized SMILES at row {index}: {value}")
        matrix[index] = generator.GetFingerprintAsNumPy(molecule)
    return matrix


def multifingerprint_4263_matrix(smiles: list[str]) -> np.ndarray:
    """FCFP2 + FCFP3 + ECFP2 + MACCS + hashed atom-pair features.

    This exactly follows the 4,263-dimensional feature ordering released by
    Multi-RF-Fusion (arXiv:2603.20724), while rejecting invalid standardized input instead of
    silently replacing it with methane.
    """
    matrix = np.empty((len(smiles), 4263), dtype=np.uint8)
    rdBase.DisableLog("rdApp.warning")
    try:
        for index, value in enumerate(smiles):
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise ValueError(f"Invalid standardized SMILES at row {index}: {value}")
            fingerprints = [
                AllChem.GetMorganFingerprintAsBitVect(
                    molecule, 2, nBits=1024, useFeatures=True
                ),
                AllChem.GetMorganFingerprintAsBitVect(
                    molecule, 3, nBits=1024, useFeatures=True
                ),
                AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024),
                MACCSkeys.GenMACCSKeys(molecule),
                rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                    molecule, nBits=1024
                ),
            ]
            offset = 0
            for fingerprint in fingerprints:
                block = np.empty(len(fingerprint), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(fingerprint, block)
                matrix[index, offset : offset + len(block)] = block
                offset += len(block)
            if offset != matrix.shape[1]:
                raise RuntimeError(f"Unexpected Multi-RF feature dimension: {offset}")
    finally:
        rdBase.EnableLog("rdApp.warning")
    return matrix


def rdkit_descriptor_matrix(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return RDKit's registered scalar 2D descriptors in a stable order.

    Descriptor failures are represented by NaN and must be handled using a
    preprocessor fitted on the training partition only. Invalid standardized
    SMILES remain a hard error.
    """
    descriptor_list = list(Descriptors._descList)
    names = [name for name, _ in descriptor_list]
    matrix = np.full((len(smiles), len(descriptor_list)), np.nan, dtype=np.float64)
    for row, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid standardized SMILES at row {row}: {value}")
        for column, (_, function) in enumerate(descriptor_list):
            try:
                result = float(function(molecule))
            except (ArithmeticError, RuntimeError, TypeError, ValueError):
                continue
            if np.isfinite(result):
                matrix[row, column] = result
    return matrix, names


def fit_clean_rdkit_descriptors(
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train-only median imputation followed by train-only constant removal."""
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("Descriptor matrices must be aligned two-dimensional arrays")
    finite_train = np.where(np.isfinite(train), train, np.nan)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(finite_train, axis=0)
    usable = np.isfinite(medians)
    train_imputed = np.where(np.isfinite(train), train, medians)
    validation_imputed = np.where(np.isfinite(validation), validation, medians)
    # sklearn tree ensembles cast inputs to float32 internally. A small number
    # of valid RDKit descriptors (notably information-content variants on some
    # molecules) can exceed that representable range despite being finite in
    # float64. This deterministic numeric cap does not use validation statistics.
    float32_limit = float(np.finfo(np.float32).max)
    train_imputed = np.clip(train_imputed, -float32_limit, float32_limit)
    validation_imputed = np.clip(validation_imputed, -float32_limit, float32_limit)
    if usable.any():
        spread = np.ptp(train_imputed[:, usable], axis=0)
        usable_indices = np.flatnonzero(usable)
        usable[usable_indices[spread <= 0.0]] = False
    if not usable.any():
        raise RuntimeError("No usable non-constant RDKit descriptors in training data")
    return (
        train_imputed[:, usable],
        validation_imputed[:, usable],
        medians,
        usable,
    )
