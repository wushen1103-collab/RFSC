from __future__ import annotations

import hashlib

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def make_fingerprints(smiles: list[str], radius: int, bits: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    result = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid standardized SMILES at row {index}")
        result.append(generator.GetFingerprint(molecule))
    return result


def query_neighbors(
    reference_fingerprints: list,
    query_fingerprints: list,
    *,
    k: int,
    floor: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    indices, similarities = [], []
    for fingerprint in query_fingerprints:
        values = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fingerprints),
            dtype=float,
        )
        eligible = np.flatnonzero(values >= floor)
        order = np.argsort(-values[eligible], kind="stable")[:k]
        eligible = eligible[order]
        indices.append(eligible.astype(int))
        similarities.append(values[eligible])
    return indices, similarities


def undirected_knn_pairs(
    fingerprints: list, *, k: int, floor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs: dict[tuple[int, int], float] = {}
    for left, fingerprint in enumerate(fingerprints):
        values = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints), dtype=float
        )
        values[left] = -1.0
        eligible = np.flatnonzero(values >= floor)
        order = np.argsort(-values[eligible], kind="stable")[:k]
        for right in eligible[order]:
            pair = (min(left, int(right)), max(left, int(right)))
            pairs[pair] = max(pairs.get(pair, -1.0), float(values[right]))
    ordered = sorted(pairs)
    return (
        np.asarray([value[0] for value in ordered], dtype=int),
        np.asarray([value[1] for value in ordered], dtype=int),
        np.asarray([pairs[value] for value in ordered], dtype=float),
    )


def symmetric_edge_features(
    left_embedding: np.ndarray,
    right_embedding: np.ndarray,
    left_prediction: np.ndarray,
    right_prediction: np.ndarray,
    similarity: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            similarity,
            np.abs(left_prediction - right_prediction),
            np.abs(left_embedding - right_embedding),
            left_embedding * right_embedding,
        ]
    )


def apply_soft_propagation(
    base_query: np.ndarray,
    base_reference: np.ndarray,
    neighbor_indices: list[np.ndarray],
    similarities: list[np.ndarray],
    gate: list[np.ndarray],
    strength: float,
) -> np.ndarray:
    result = base_query.copy()
    for query_index, (reference_index, similarity, boundary_probability) in enumerate(
        zip(neighbor_indices, similarities, gate)
    ):
        weight = similarity * (1.0 - boundary_probability)
        if len(weight) == 0 or float(weight.sum()) <= 1e-12:
            continue
        message = np.average(base_reference[reference_index], weights=weight)
        result[query_index] = (1.0 - strength) * base_query[query_index] + strength * message
    return result


def apply_mass_aware_soft_propagation(
    base_query: np.ndarray,
    base_reference: np.ndarray,
    neighbor_indices: list[np.ndarray],
    similarities: list[np.ndarray],
    gate: list[np.ndarray],
    strength: float,
) -> np.ndarray:
    """Attenuate both neighbor weights and total propagation by smooth mass.

    ``apply_soft_propagation`` normalizes the surviving edge weights and then
    applies the full propagation strength.  Consequently, a query whose edges
    all have high (but non-unit) boundary probability is still smoothed at full
    strength.  This residual form normalizes by the original similarity mass,
    so boundary probability also controls how much propagation is applied.
    """
    result = base_query.copy()
    for query_index, (reference_index, similarity, boundary_probability) in enumerate(
        zip(neighbor_indices, similarities, gate)
    ):
        if len(similarity) == 0:
            continue
        original_mass = float(similarity.sum())
        if original_mass <= 1e-12:
            continue
        smooth_weight = similarity * (1.0 - boundary_probability)
        residual = np.sum(
            smooth_weight
            * (base_reference[reference_index] - base_query[query_index])
        ) / original_mass
        result[query_index] = base_query[query_index] + strength * residual
    return result


def apply_dual_propagation(
    base_query: np.ndarray,
    base_reference: np.ndarray,
    neighbor_indices: list[np.ndarray],
    similarities: list[np.ndarray],
    gate: list[np.ndarray],
    strength: float,
    boundary_strength: float,
) -> np.ndarray:
    """Combine boundary-cut low pass with a signed boundary high-pass residual."""
    result = apply_soft_propagation(
        base_query,
        base_reference,
        neighbor_indices,
        similarities,
        gate,
        strength,
    )
    for query_index, (reference_index, similarity, boundary_probability) in enumerate(
        zip(neighbor_indices, similarities, gate)
    ):
        weight = similarity * boundary_probability
        if len(weight) == 0 or float(weight.sum()) <= 1e-12:
            continue
        signed_jump = np.average(
            base_query[query_index] - base_reference[reference_index], weights=weight
        )
        result[query_index] += boundary_strength * signed_jump
    return result


def similarity_matched_gate(
    learned_gate: list[np.ndarray], similarities: list[np.ndarray]
) -> list[np.ndarray]:
    result = []
    for probability, similarity in zip(learned_gate, similarities):
        assigned = np.empty_like(probability)
        assigned[np.argsort(similarity, kind="stable")] = np.sort(probability)[::-1]
        result.append(assigned)
    return result


def stable_rng(seed: int, *parts) -> np.random.Generator:
    payload = "\0".join([str(seed), *map(str, parts)]).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(value)


def shuffled_gate(gate: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    return [rng.permutation(value) for value in gate]
