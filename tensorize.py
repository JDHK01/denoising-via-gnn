"""Convert all-in graphs to tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from aegis.model import TensorHeteroGraph

from .graph import EdgeType, HeteroGraph, NodeRef


NODE_TYPES = ("alert", "ip", "event")


@dataclass
class TensorizedGraph:
    graph: TensorHeteroGraph
    node_index: dict[NodeRef, int]
    index_node: dict[str, list[NodeRef]]
    input_dims: dict[str, int]
    edge_types: tuple[EdgeType, ...]

    def local_index(self, ref: NodeRef) -> int:
        return self.node_index[ref]


def to_tensor_graph(
    graph: HeteroGraph,
    *,
    alert_tfidf: np.ndarray | None = None,
    input_dims: dict[str, int] | None = None,
) -> TensorizedGraph:
    index_node = {node_type: sorted(graph.node_refs(node_type)) for node_type in NODE_TYPES}
    node_index: dict[NodeRef, int] = {}
    for node_type, refs in index_node.items():
        for local, ref in enumerate(refs):
            node_index[ref] = local

    dims = input_dims or infer_input_dims(graph, alert_tfidf=alert_tfidf)
    x = {
        node_type: torch.from_numpy(_feature_matrix(graph, refs, dims[node_type], alert_tfidf=alert_tfidf))
        for node_type, refs in index_node.items()
    }

    edge_pairs: dict[EdgeType, list[tuple[int, int]]] = {}
    edge_weights: dict[EdgeType, list[float]] = {}
    for edge in graph.edges:
        if edge.src not in node_index or edge.dst not in node_index:
            continue
        edge_type = edge.edge_type
        edge_pairs.setdefault(edge_type, []).append((node_index[edge.src], node_index[edge.dst]))
        edge_weights.setdefault(edge_type, []).append(edge.weight)

    edge_index: dict[EdgeType, torch.Tensor] = {}
    edge_weight: dict[EdgeType, torch.Tensor] = {}
    for edge_type, pairs in edge_pairs.items():
        edge_index[edge_type] = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        edge_weight[edge_type] = torch.tensor(edge_weights[edge_type], dtype=torch.float32)

    return TensorizedGraph(
        graph=TensorHeteroGraph(x=x, edge_index=edge_index, edge_weight=edge_weight),
        node_index=node_index,
        index_node=index_node,
        input_dims=dims,
        edge_types=tuple(sorted(edge_index)),
    )


def infer_input_dims(graph: HeteroGraph, *, alert_tfidf: np.ndarray | None = None) -> dict[str, int]:
    dims: dict[str, int] = {}
    for node_type in NODE_TYPES:
        dim = max((len(graph.nodes[ref].features) for ref in graph.node_refs(node_type)), default=1)
        dims[node_type] = max(dim, 1)
    if alert_tfidf is not None:
        dims["alert"] += int(alert_tfidf.shape[1])
    return dims


def move_graph(graph: TensorHeteroGraph, device: torch.device) -> TensorHeteroGraph:
    return TensorHeteroGraph(
        x={key: value.to(device) for key, value in graph.x.items()},
        edge_index={key: value.to(device) for key, value in graph.edge_index.items()},
        edge_weight={key: value.to(device) for key, value in graph.edge_weight.items()},
        edge_attr={key: value.to(device) for key, value in graph.edge_attr.items()},
    )


def _feature_matrix(
    graph: HeteroGraph,
    refs: Sequence[NodeRef],
    dim: int,
    *,
    alert_tfidf: np.ndarray | None,
) -> np.ndarray:
    matrix = np.zeros((len(refs), dim), dtype=np.float32)
    for row, ref in enumerate(refs):
        node = graph.nodes[ref]
        features = node.features
        if ref.node_type == "alert" and alert_tfidf is not None:
            record_index = int(node.attrs["record_index"])
            features = np.concatenate((alert_tfidf[record_index], features)).astype(np.float32, copy=False)
        length = min(len(features), dim)
        if length:
            matrix[row, :length] = features[:length]
    return matrix
