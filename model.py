"""HGT-only classifier for all-in alert denoising."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from aegis.model import EdgeType, HGTEncoder, HGTModelConfig, TensorHeteroGraph


@dataclass
class AllInModelConfig:
    node_types: tuple[str, ...]
    edge_types: tuple[EdgeType, ...]
    input_dims: dict[str, int]
    hidden_dim: int = 128
    num_heads: int = 4
    hgt_layers: int = 2
    dropout: float = 0.15
    alert_residual: bool = True


class HGTOnlyClassifier(nn.Module):
    def __init__(self, config: AllInModelConfig) -> None:
        super().__init__()
        self.config = config
        hgt_config = HGTModelConfig(
            node_types=config.node_types,
            edge_types=config.edge_types,
            input_dims=config.input_dims,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            pre_hgt_layers=0,
            hgt_layers=config.hgt_layers,
        )
        self.input_proj = nn.ModuleDict(
            {
                node_type: nn.Linear(config.input_dims[node_type], config.hidden_dim)
                for node_type in config.node_types
            }
        )
        self.hgt = HGTEncoder(hgt_config, config.hgt_layers)
        classifier_dim = config.hidden_dim * (2 if config.alert_residual else 1)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, graph: TensorHeteroGraph, alert_indices: torch.Tensor | list[int]) -> torch.Tensor:
        x = {node_type: self.input_proj[node_type](features) for node_type, features in graph.x.items()}
        h = self.hgt(x, graph.edge_index, graph.edge_weight)
        indices = torch.as_tensor(alert_indices, dtype=torch.long, device=h["alert"].device)
        alert_h = h["alert"][indices]
        if self.config.alert_residual:
            alert_h = torch.cat((alert_h, x["alert"][indices]), dim=-1)
        return self.classifier(alert_h).squeeze(-1)
