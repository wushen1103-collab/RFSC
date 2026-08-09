from __future__ import annotations

import torch
from torch_geometric.data import Batch

from bps_gnn.models import (
    AttentiveFPRegressor,
    GATv2Regressor,
    GCNRegressor,
    GINRegressor,
    GINERegressor,
    GraphCliffRegressor,
    GraphSAGERegressor,
    MPNNRegressor,
    _normalized_adjacency,
)
from bps_gnn.molecular_graphs import ATOM_DIM, BOND_DIM, smiles_to_graph


def test_graph_dimensions_and_stereo() -> None:
    graph = smiles_to_graph("F[C@H](Cl)C=C", 7)
    assert graph.x.shape[1] == ATOM_DIM
    assert graph.edge_attr.shape[1] == BOND_DIM
    assert graph.row_id.item() == 7


def test_gine_forward_and_backward() -> None:
    batch = Batch.from_data_list(
        [smiles_to_graph("CCO", 0), smiles_to_graph("c1ccccc1", 1)]
    )
    model = GINERegressor(ATOM_DIM, BOND_DIM, hidden_dim=32, layers=2, dropout=0.0)
    output, embedding = model.forward_with_embedding(batch)
    assert output.shape == (2,)
    assert embedding.shape == (2, 32)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_modern_architecture_forward_and_backward() -> None:
    batch = Batch.from_data_list(
        [smiles_to_graph("C", 0), smiles_to_graph("CCO", 1), smiles_to_graph("c1ccccc1", 2)]
    )
    models = [
        GCNRegressor(ATOM_DIM, BOND_DIM, hidden_dim=16, layers=2, dropout=0.0),
        GraphSAGERegressor(ATOM_DIM, BOND_DIM, hidden_dim=16, layers=2, dropout=0.0),
        GINRegressor(ATOM_DIM, BOND_DIM, hidden_dim=16, layers=2, dropout=0.0),
        GATv2Regressor(
            ATOM_DIM, BOND_DIM, hidden_dim=16, layers=2, heads=4, dropout=0.0
        ),
        MPNNRegressor(
            ATOM_DIM, BOND_DIM, hidden_dim=16, layers=2, set2set_steps=2, dropout=0.0
        ),
        AttentiveFPRegressor(
            ATOM_DIM, BOND_DIM, hidden_dim=16, layers=2, timesteps=2, dropout=0.0
        ),
    ]
    for model in models:
        model.zero_grad(set_to_none=True)
        output, embedding = model.forward_with_embedding(batch)
        assert output.shape == (3,)
        assert embedding.shape[0] == 3
        output.square().mean().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())


def test_graphcliff_forward_and_backward() -> None:
    batch = Batch.from_data_list(
        [smiles_to_graph("CCO", 0), smiles_to_graph("c1ccccc1Cl", 1)]
    )
    model = GraphCliffRegressor(
        ATOM_DIM,
        BOND_DIM,
        hidden_dim=16,
        layers=2,
        chebyshev_order=3,
        sag_ratio=0.5,
        dropout=0.0,
    )
    output, embedding = model.forward_with_embedding(batch)
    assert output.shape == (2,)
    assert embedding.shape == (2, 16)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_graphcliff_normalized_adjacency_preserves_autocast_dtype() -> None:
    features = torch.randn(3, 4, dtype=torch.float16)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    propagated = _normalized_adjacency(features, edge_index)
    assert propagated.dtype == features.dtype
