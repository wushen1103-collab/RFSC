from __future__ import annotations

import os
from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.data import Data


ATOM_NUMBERS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
DEGREES = list(range(7))
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
    Chem.HybridizationType.UNSPECIFIED,
]
CHIRAL_TAGS = [
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
TOTAL_HS = list(range(5))
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
BOND_STEREO = [
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOANY,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
]
CACHE_VERSION = 2


def one_hot_other(value, choices: list) -> list[float]:
    result = [0.0] * (len(choices) + 1)
    result[choices.index(value) if value in choices else -1] = 1.0
    return result


def atom_features(atom: Chem.Atom) -> list[float]:
    return [
        *one_hot_other(atom.GetAtomicNum(), ATOM_NUMBERS),
        *one_hot_other(atom.GetDegree(), DEGREES),
        *one_hot_other(atom.GetFormalCharge(), FORMAL_CHARGES),
        *one_hot_other(atom.GetHybridization(), HYBRIDIZATIONS),
        *one_hot_other(atom.GetChiralTag(), CHIRAL_TAGS),
        *one_hot_other(atom.GetTotalNumHs(), TOTAL_HS),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetMass() / 100.0,
    ]


def bond_features(bond: Chem.Bond) -> list[float]:
    return [
        *one_hot_other(bond.GetBondType(), BOND_TYPES),
        *one_hot_other(bond.GetStereo(), BOND_STEREO),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def smiles_to_graph(smiles: str, mol_index: int) -> Data:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid standardized SMILES: {smiles}")
    x = torch.tensor([atom_features(atom) for atom in molecule.GetAtoms()], dtype=torch.float32)
    sources: list[int] = []
    destinations: list[int] = []
    attributes: list[list[float]] = []
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_features(bond)
        sources.extend([begin, end])
        destinations.extend([end, begin])
        attributes.extend([feature, feature])
    edge_index = torch.tensor([sources, destinations], dtype=torch.long)
    if attributes:
        edge_attr = torch.tensor(attributes, dtype=torch.float32)
    else:
        edge_attr = torch.empty((0, BOND_DIM), dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        row_id=torch.tensor([mol_index], dtype=torch.long),
    )


def load_or_build_graphs(smiles: list[str], cache_path: Path) -> list[Data]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("version") == CACHE_VERSION and payload["smiles"] == smiles:
            return payload["graphs"]
        if payload["smiles"] != smiles:
            raise RuntimeError("Molecular graph cache does not match standardized structures")
    graphs = [smiles_to_graph(value, index) for index, value in enumerate(smiles)]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    torch.save({"version": CACHE_VERSION, "smiles": smiles, "graphs": graphs}, temporary)
    os.replace(temporary, cache_path)
    return graphs
