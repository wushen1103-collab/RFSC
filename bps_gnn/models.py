from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import (
    AttentiveFP,
    GATv2Conv,
    GCNConv,
    GINConv,
    GINEConv,
    NNConv,
    SAGPooling,
    SAGEConv,
    Set2Set,
    global_mean_pool,
)
from torch_geometric.utils import add_self_loops


def _regression_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


class _ResidualGraphRegressor(nn.Module):
    """Common readout for same-protocol molecular architecture reruns."""

    def __init__(self, atom_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.atom_encoder = nn.Linear(atom_dim, hidden_dim)
        self.layers = layers
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = _regression_head(hidden_dim, hidden_dim, dropout)

    def _convolve(self, layer: int, hidden: torch.Tensor, batch) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, batch) -> torch.Tensor:
        hidden = self.atom_encoder(batch.x)
        for layer, norm in enumerate(self.norms):
            update = self._convolve(layer, hidden, batch)
            hidden = hidden + self.dropout(torch.nn.functional.silu(norm(update)))
        return global_mean_pool(hidden, batch.batch)

    def forward(self, batch) -> torch.Tensor:
        return self.forward_with_embedding(batch)[0]

    def forward_with_embedding(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(batch)
        return self.head(embedding).squeeze(-1), embedding


class GCNRegressor(_ResidualGraphRegressor):
    def __init__(
        self, atom_dim: int, bond_dim: int, hidden_dim: int = 128, layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        del bond_dim
        super().__init__(atom_dim, hidden_dim, layers, dropout)
        self.convolutions = nn.ModuleList(
            GCNConv(hidden_dim, hidden_dim) for _ in range(layers)
        )

    def _convolve(self, layer: int, hidden: torch.Tensor, batch) -> torch.Tensor:
        return self.convolutions[layer](hidden, batch.edge_index)


class GraphSAGERegressor(_ResidualGraphRegressor):
    def __init__(
        self, atom_dim: int, bond_dim: int, hidden_dim: int = 128, layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        del bond_dim
        super().__init__(atom_dim, hidden_dim, layers, dropout)
        self.convolutions = nn.ModuleList(
            SAGEConv(hidden_dim, hidden_dim) for _ in range(layers)
        )

    def _convolve(self, layer: int, hidden: torch.Tensor, batch) -> torch.Tensor:
        return self.convolutions[layer](hidden, batch.edge_index)


class GINRegressor(_ResidualGraphRegressor):
    def __init__(
        self, atom_dim: int, bond_dim: int, hidden_dim: int = 128, layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        del bond_dim
        super().__init__(atom_dim, hidden_dim, layers, dropout)
        self.convolutions = nn.ModuleList(
            GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                train_eps=True,
            )
            for _ in range(layers)
        )

    def _convolve(self, layer: int, hidden: torch.Tensor, batch) -> torch.Tensor:
        return self.convolutions[layer](hidden, batch.edge_index)


class GATv2Regressor(_ResidualGraphRegressor):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        super().__init__(atom_dim, hidden_dim, layers, dropout)
        self.convolutions = nn.ModuleList(
            GATv2Conv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                edge_dim=bond_dim,
                dropout=dropout,
                add_self_loops=True,
            )
            for _ in range(layers)
        )

    def _convolve(self, layer: int, hidden: torch.Tensor, batch) -> torch.Tensor:
        return self.convolutions[layer](hidden, batch.edge_index, batch.edge_attr)


class MPNNRegressor(nn.Module):
    """Gilmer-style neural message passing with a Set2Set graph readout."""

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 128,
        layers: int = 3,
        set2set_steps: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.atom_encoder = nn.Linear(atom_dim, hidden_dim)
        self.convolutions = nn.ModuleList()
        self.updates = nn.ModuleList()
        for _ in range(layers):
            edge_network = nn.Sequential(
                nn.Linear(bond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            )
            self.convolutions.append(
                NNConv(hidden_dim, hidden_dim, edge_network, aggr="mean", root_weight=True)
            )
            self.updates.append(nn.GRUCell(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.readout = Set2Set(hidden_dim, processing_steps=set2set_steps)
        self.head = _regression_head(hidden_dim * 2, hidden_dim, dropout)

    def encode(self, batch) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.atom_encoder(batch.x))
        for convolution, update in zip(self.convolutions, self.updates):
            message = torch.nn.functional.silu(
                convolution(hidden, batch.edge_index, batch.edge_attr)
            )
            hidden = update(self.dropout(message), hidden)
        return self.readout(hidden, batch.batch)

    def forward(self, batch) -> torch.Tensor:
        return self.forward_with_embedding(batch)[0]

    def forward_with_embedding(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(batch)
        return self.head(embedding).squeeze(-1), embedding


class AttentiveFPRegressor(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 128,
        layers: int = 2,
        timesteps: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.model = AttentiveFP(
            in_channels=atom_dim,
            hidden_channels=hidden_dim,
            out_channels=1,
            edge_dim=bond_dim,
            num_layers=layers,
            num_timesteps=timesteps,
            dropout=dropout,
        )

    def forward(self, batch) -> torch.Tensor:
        return self.forward_with_embedding(batch)[0]

    def forward_with_embedding(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # AttentiveFP does not expose its graph state through the public API.
        # The common trainer only consumes this auxiliary embedding for the
        # ACANet loss, which is not used for this architecture rerun.
        embedding = global_mean_pool(batch.x, batch.batch)
        return prediction.view(-1), embedding


class GINERegressor(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 128,
        layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.atom_encoder = nn.Linear(atom_dim, hidden_dim)
        self.convolutions = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convolutions.append(GINEConv(mlp, train_eps=True, edge_dim=bond_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, batch) -> torch.Tensor:
        hidden = self.atom_encoder(batch.x)
        for convolution, norm in zip(self.convolutions, self.norms):
            update = convolution(hidden, batch.edge_index, batch.edge_attr)
            hidden = hidden + self.dropout(torch.nn.functional.silu(norm(update)))
        return global_mean_pool(hidden, batch.batch)

    def forward(self, batch) -> torch.Tensor:
        return self.forward_with_embedding(batch)[0]

    def forward_with_embedding(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(batch)
        return self.head(embedding).squeeze(-1), embedding


def _normalized_adjacency(features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Apply a symmetrically normalized adjacency matrix with self loops."""
    edge_index, _ = add_self_loops(edge_index, num_nodes=features.shape[0])
    source, destination = edge_index
    degree = torch.bincount(destination, minlength=features.shape[0]).to(features.dtype)
    inverse_sqrt_degree = degree.clamp_min(1.0).pow(-0.5)
    weight = (inverse_sqrt_degree[source] * inverse_sqrt_degree[destination]).to(
        dtype=features.dtype
    )
    propagated = torch.zeros_like(features)
    propagated.index_add_(0, destination, features[source] * weight.unsqueeze(-1))
    return propagated


class GraphCliffFilter(nn.Module):
    """Paper-formula adaptation of the GraphCliff short/long-range filter."""

    def __init__(
        self,
        hidden_dim: int,
        bond_dim: int,
        chebyshev_order: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if chebyshev_order < 1:
            raise ValueError("chebyshev_order must be at least one")
        self.chebyshev_order = chebyshev_order
        self.norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim * 3)
        short_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 6),
            nn.SiLU(),
            nn.Linear(hidden_dim * 6, hidden_dim * 3),
        )
        self.short_filter = GINEConv(short_mlp, train_eps=True, edge_dim=bond_dim)
        self.chebyshev_coefficients = nn.Parameter(torch.zeros(chebyshev_order + 1))
        with torch.no_grad():
            self.chebyshev_coefficients[0] = 1.0
        self.dropout = nn.Dropout(dropout)

    def _long_filter(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        terms = [features, _normalized_adjacency(features, edge_index)]
        for _ in range(2, self.chebyshev_order + 1):
            terms.append(2.0 * _normalized_adjacency(terms[-1], edge_index) - terms[-2])
        return sum(coefficient * term for coefficient, term in zip(self.chebyshev_coefficients, terms))

    def forward(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.projection(self.norm(hidden))
        short = self.short_filter(projected, edge_index, edge_attr)
        long_input, gate_input, residual_value = short.chunk(3, dim=-1)
        long_output = self._long_filter(long_input, edge_index)
        update = torch.sigmoid(gate_input) * long_output + residual_value
        return hidden + self.dropout(update)


class GraphCliffRegressor(nn.Module):
    """Auditable GraphCliff adaptation from arXiv:2511.03170v2 equations 1--8.

    The paper does not release code or complete hyperparameters. This class must therefore be
    reported as a formula-level adaptation, not as an official reproduction.
    """

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int = 128,
        layers: int = 3,
        chebyshev_order: int = 3,
        sag_ratio: float = 0.5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.filters = nn.ModuleList(
            [
                GraphCliffFilter(hidden_dim, bond_dim, chebyshev_order, dropout)
                for _ in range(layers)
            ]
        )
        self.pool = SAGPooling(hidden_dim, ratio=sag_ratio)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, batch) -> torch.Tensor:
        hidden = self.atom_encoder(batch.x)
        for graph_filter in self.filters:
            hidden = graph_filter(hidden, batch.edge_index, batch.edge_attr)
        hidden, _, _, pooled_batch, _, _ = self.pool(
            hidden,
            batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )
        return global_mean_pool(hidden, pooled_batch)

    def forward(self, batch) -> torch.Tensor:
        return self.forward_with_embedding(batch)[0]

    def forward_with_embedding(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(batch)
        return self.head(embedding).squeeze(-1), embedding
