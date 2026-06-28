from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class OrderedQuantileMLP(nn.Module):
    """Tabular MLP that predicts ordered log-price quantiles.

    The network does not classify deal labels. It predicts only three log-scale
    quantiles. Ordering is enforced in the forward pass with positive deltas.
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int],
        hidden_dims: list[int] | None = None,
        dropout: float = 0.15,
        embedding_dropout: float = 0.05,
        max_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if n_num_features == 0 and not cat_cardinalities:
            raise ValueError("At least one numeric or categorical feature is required")

        self.n_num_features = int(n_num_features)
        self.cat_cardinalities = [int(value) for value in cat_cardinalities]
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = float(dropout)
        self.embedding_dropout = float(embedding_dropout)
        self.max_embedding_dim = int(max_embedding_dim)

        self.embedding_dims = [
            min(self.max_embedding_dim, max(4, int(round(math.sqrt(cardinality)))))
            for cardinality in self.cat_cardinalities
        ]
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embedding_dim)
            for cardinality, embedding_dim in zip(
                self.cat_cardinalities,
                self.embedding_dims,
            )
        )
        self.embedding_dropout_layer = nn.Dropout(self.embedding_dropout)

        input_dim = self.n_num_features + sum(self.embedding_dims)
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for idx, hidden_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if idx < 2:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 3))
        self.net = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.n_num_features:
            parts.append(x_num)
        if self.embeddings:
            embedded = [
                embedding(x_cat[:, idx])
                for idx, embedding in enumerate(self.embeddings)
            ]
            parts.append(self.embedding_dropout_layer(torch.cat(embedded, dim=1)))

        x = torch.cat(parts, dim=1)
        raw = self.net(x)

        base = raw[:, 0:1]
        delta_1 = F.softplus(raw[:, 1:2])
        delta_2 = F.softplus(raw[:, 2:3])
        p10 = base
        p50 = base + delta_1
        p90 = p50 + delta_2
        return torch.cat([p10, p50, p90], dim=1)

    def config(self) -> dict[str, Any]:
        return {
            "n_num_features": self.n_num_features,
            "cat_cardinalities": self.cat_cardinalities,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "embedding_dropout": self.embedding_dropout,
            "max_embedding_dim": self.max_embedding_dim,
            "model_class": self.__class__.__name__,
        }
