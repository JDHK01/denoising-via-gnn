"""HGT-only classifier for all-in alert denoising."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


EdgeType = tuple[str, str, str]


@dataclass
class TensorHeteroGraph:
    """Tensor graph consumed by the model.

    `edge_index[(src_type, relation, dst_type)]` has shape [2, num_edges].
    Row 0 stores source node indices and row 1 stores destination node indices.
    """

    x: dict[str, torch.Tensor]
    edge_index: dict[EdgeType, torch.Tensor]
    edge_weight: dict[EdgeType, torch.Tensor] = field(default_factory=dict)
    edge_attr: dict[EdgeType, torch.Tensor] = field(default_factory=dict)


@dataclass
class HGTModelConfig:
    node_types: tuple[str, ...] = ("alert", "ip", "event")
    edge_types: tuple[EdgeType, ...] = ()
    input_dims: dict[str, int] = field(default_factory=dict)
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    hgt_layers: int = 2


def _edge_type_key(edge_type: EdgeType) -> str:
    return "__".join(edge_type).replace(".", "_").replace("-", "_")


def _edge_softmax(score: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
    if score.numel() == 0:
        return score
    num_dst = int(dst_idx.max().item()) + 1
    heads = score.shape[1]
    expanded_dst = dst_idx.view(-1, 1).expand(-1, heads)
    max_per_dst = torch.full(
        (num_dst, heads),
        -torch.inf,
        dtype=score.dtype,
        device=score.device,
    )
    max_per_dst.scatter_reduce_(0, expanded_dst, score, reduce="amax", include_self=True)
    exp_score = torch.exp(score - max_per_dst[dst_idx])
    denom = torch.zeros((num_dst, heads), dtype=score.dtype, device=score.device)
    denom.scatter_add_(0, expanded_dst, exp_score)
    return exp_score / denom[dst_idx].clamp_min(1e-12)


class HGTLayer(nn.Module):
    def __init__(self, config: HGTModelConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.config = config
        self.head_dim = config.hidden_dim // config.num_heads
        self.q = nn.ModuleDict(
            {node_type: nn.Linear(config.hidden_dim, config.hidden_dim) for node_type in config.node_types}
        )
        self.k = nn.ModuleDict(
            {node_type: nn.Linear(config.hidden_dim, config.hidden_dim) for node_type in config.node_types}
        )
        self.v = nn.ModuleDict(
            {node_type: nn.Linear(config.hidden_dim, config.hidden_dim) for node_type in config.node_types}
        )
        self.out = nn.ModuleDict(
            {node_type: nn.Linear(config.hidden_dim, config.hidden_dim) for node_type in config.node_types}
        )
        self.norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(config.hidden_dim) for node_type in config.node_types}
        )
        self.dropout = nn.Dropout(config.dropout)
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        self.relation_pri = nn.ParameterDict()
        for edge_type in config.edge_types:
            key = _edge_type_key(edge_type)
            self.relation_att[key] = nn.Parameter(
                torch.empty(config.num_heads, self.head_dim, self.head_dim)
            )
            self.relation_msg[key] = nn.Parameter(
                torch.empty(config.num_heads, self.head_dim, self.head_dim)
            )
            self.relation_pri[key] = nn.Parameter(torch.ones(config.num_heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module_dict in (self.q, self.k, self.v, self.out):
            for linear in module_dict.values():
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
        for param in self.relation_att.values():
            nn.init.xavier_uniform_(param)
        for param in self.relation_msg.values():
            nn.init.xavier_uniform_(param)

    def forward(
        self,
        x: dict[str, torch.Tensor],
        edge_index: dict[EdgeType, torch.Tensor],
        edge_weight: dict[EdgeType, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = {node_type: torch.zeros_like(features) for node_type, features in x.items()}
        for edge_type, index in edge_index.items():
            if index.numel() == 0:
                continue
            key = _edge_type_key(edge_type)
            if key not in self.relation_att:
                continue
            src_type, _, dst_type = edge_type
            src_idx = index[0].long()
            dst_idx = index[1].long()
            q = self.q[dst_type](x[dst_type])[dst_idx].view(-1, self.config.num_heads, self.head_dim)
            k = self.k[src_type](x[src_type])[src_idx].view(-1, self.config.num_heads, self.head_dim)
            v = self.v[src_type](x[src_type])[src_idx].view(-1, self.config.num_heads, self.head_dim)
            k = torch.einsum("ehd,hdf->ehf", k, self.relation_att[key])
            v = torch.einsum("ehd,hdf->ehf", v, self.relation_msg[key])
            score = (q * k).sum(dim=-1) * self.relation_pri[key] / (self.head_dim**0.5)
            alpha = _edge_softmax(score, dst_idx)
            if edge_weight is not None and edge_type in edge_weight:
                alpha = alpha * edge_weight[edge_type].to(alpha.device, dtype=alpha.dtype).view(-1, 1)
            message = (alpha.unsqueeze(-1) * v).reshape(-1, self.config.hidden_dim)
            outputs[dst_type].index_add_(0, dst_idx, message.to(outputs[dst_type].dtype))

        next_x: dict[str, torch.Tensor] = {}
        for node_type, features in x.items():
            updated = self.dropout(self.out[node_type](outputs[node_type]))
            next_x[node_type] = self.norm[node_type](features + updated)
        return next_x


class HGTEncoder(nn.Module):
    def __init__(self, config: HGTModelConfig, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(HGTLayer(config) for _ in range(layers))

    def forward(
        self,
        x: dict[str, torch.Tensor],
        edge_index: dict[EdgeType, torch.Tensor],
        edge_weight: dict[EdgeType, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return x


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
