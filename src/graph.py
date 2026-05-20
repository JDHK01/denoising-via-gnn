"""Small heterogeneous graph builder for all-in alerts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from data import AlertRecord
from features import (
    EventTfidfVectorizer,
    FeatureStats,
    IpInfoTfidfVectorizer,
    alert_struct_features,
    event_node_features,
    ip_node_features,
    make_event_key,
    timestamp_seconds,
)


NodeType = Literal["alert", "ip", "event"]
EdgeType = tuple[str, str, str]


@dataclass(frozen=True, order=True, slots=True)
class NodeRef:
    node_type: NodeType
    key: tuple[str, ...]

    @property
    def platform(self) -> str:
        return self.key[0]


@dataclass(slots=True)
class Node:
    ref: NodeRef
    features: np.ndarray
    attrs: dict


@dataclass(frozen=True, slots=True)
class Edge:
    src: NodeRef
    relation: str
    dst: NodeRef
    weight: float = 1.0

    @property
    def edge_type(self) -> EdgeType:
        return (self.src.node_type, self.relation, self.dst.node_type)


@dataclass
class HeteroGraph:
    nodes: dict[NodeRef, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    _edge_keys: set[tuple[NodeRef, str, NodeRef]] = field(default_factory=set, repr=False)

    def add_node(self, ref: NodeRef, features: np.ndarray, attrs: dict | None = None) -> None:
        if ref in self.nodes:
            self.nodes[ref].attrs.update(attrs or {})
            return
        self.nodes[ref] = Node(ref=ref, features=features.astype(np.float32, copy=False), attrs=dict(attrs or {}))

    def add_edge(self, src: NodeRef, relation: str, dst: NodeRef, weight: float = 1.0) -> None:
        key = (src, relation, dst)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append(Edge(src=src, relation=relation, dst=dst, weight=float(weight)))

    def add_bidirectional(self, src: NodeRef, relation: str, dst: NodeRef, weight: float = 1.0) -> None:
        self.add_edge(src, relation, dst, weight)
        self.add_edge(dst, f"rev_{relation}", src, weight)

    def node_refs(self, node_type: str) -> list[NodeRef]:
        return [ref for ref in self.nodes if ref.node_type == node_type]

    def edge_types(self) -> tuple[EdgeType, ...]:
        return tuple(sorted({edge.edge_type for edge in self.edges}))


@dataclass
class GraphBuilderConfig:
    graph_learning: Literal["none", "temporal", "similarity", "both"] = "temporal"
    temporal_window_seconds: int = 3600
    temporal_prev: int = 2
    similarity_candidate_window: int = 8
    similarity_topk: int = 2
    similarity_min_score: float = 0.05
    add_src_dst_edges: bool = True


class AllInGraphBuilder:
    def __init__(self, config: GraphBuilderConfig | None = None) -> None:
        self.config = config or GraphBuilderConfig()

    # alert and ip use two different tfidf
    def build(self, records: Sequence[AlertRecord], alert_tfidf: np.ndarray | None, ip_info_tfidf_vec: IpInfoTfidfVectorizer, ip_info_map: dict, event_tfidf_vec: EventTfidfVectorizer) -> HeteroGraph:
        stats = FeatureStats.from_records(records, isolate_platform=True)
        graph = HeteroGraph()
        alert_refs: list[NodeRef] = []

        for record in records:
            platform = record.platform_name or "__missing_platform__"
            alert_ref = NodeRef("alert", (platform, f"alert:{record.index}"))
            alert_refs.append(alert_ref)
            graph.add_node(
                alert_ref,
                alert_struct_features(record, stats, isolate_platform=True),
                {
                    "record_index": record.index,
                    "label": record.label,
                    "start_time": record.start_time,
                    "event_key": make_event_key(record),
                },
            )

            event_ref = NodeRef("event", make_event_key(record))
            if event_ref not in graph.nodes:
                graph.add_node(event_ref, event_node_features(event_ref.key, stats, event_tfidf=event_tfidf_vec.transform_event(event_ref.key)), {"event_key": event_ref.key})
            graph.add_bidirectional(alert_ref, "alert_has_event", event_ref)

            source_refs = self._add_ip_role_edges(graph, record, stats, alert_ref, "source", record.source_ip_list, ip_info_tfidf_vec, ip_info_map)
            destination_refs = self._add_ip_role_edges(
                graph,
                record,
                stats,
                alert_ref,
                "destination",
                record.destination_ip_list,
                ip_info_tfidf_vec,
                ip_info_map,
            )
            if self.config.add_src_dst_edges:
                for src_ref in source_refs:
                    for dst_ref in destination_refs:
                        if src_ref != dst_ref:
                            graph.add_bidirectional(src_ref, "ip_connects_ip", dst_ref)

        if self.config.graph_learning in {"temporal", "both"}:
            self._add_temporal_edges(graph, records, alert_refs)
        if self.config.graph_learning in {"similarity", "both"}:
            self._add_similarity_edges(graph, records, alert_refs, alert_tfidf)
        return graph

    def _add_ip_role_edges(
        self,
        graph: HeteroGraph,
        record: AlertRecord,
        stats: FeatureStats,
        alert_ref: NodeRef,
        role: Literal["source", "destination"],
        ips: Sequence[str],
        ip_info_tfidf_vec: IpInfoTfidfVectorizer,
        ip_info_map: dict,
    ) -> list[NodeRef]:
        platform = record.platform_name or "__missing_platform__"
        refs: list[NodeRef] = []
        for ip in ips:
            ip_ref = NodeRef("ip", (platform, ip))
            refs.append(ip_ref)
            if ip_ref not in graph.nodes:
                ip_tfidf = ip_info_tfidf_vec.transform_ip(ip, ip_info_map)
                graph.add_node(ip_ref, ip_node_features(platform, ip, stats, ip_info_tfidf=ip_tfidf), {"ip": ip, "platform_name": platform})
            graph.add_bidirectional(alert_ref, f"alert_has_{role}_ip", ip_ref)
        return refs

    def _add_temporal_edges(
        self,
        graph: HeteroGraph,
        records: Sequence[AlertRecord],
        alert_refs: Sequence[NodeRef],
    ) -> None:
        buckets: dict[tuple[str, str, str], list[tuple[int, int, NodeRef]]] = defaultdict(list)
        for record, ref in zip(records, alert_refs, strict=True):
            ts = timestamp_seconds(record.start_time)
            if ts is None:
                continue
            buckets[make_event_key(record)].append((ts, record.index, ref))

        for rows in buckets.values():
            rows.sort()
            for pos, (ts, _idx, ref) in enumerate(rows):
                added = 0
                cursor = pos - 1
                while cursor >= 0 and added < self.config.temporal_prev:
                    prev_ts, _prev_idx, prev_ref = rows[cursor]
                    if ts - prev_ts > self.config.temporal_window_seconds:
                        break
                    weight = 1.0 / (1.0 + (ts - prev_ts) / max(self.config.temporal_window_seconds, 1))
                    graph.add_bidirectional(prev_ref, "alert_temporal_near", ref, weight)
                    added += 1
                    cursor -= 1

    def _add_similarity_edges(
        self,
        graph: HeteroGraph,
        records: Sequence[AlertRecord],
        alert_refs: Sequence[NodeRef],
        alert_tfidf: np.ndarray | None,
    ) -> None:
        buckets: dict[tuple[str, str, str], list[tuple[int, NodeRef]]] = defaultdict(list)
        for record, ref in zip(records, alert_refs, strict=True):
            buckets[make_event_key(record)].append((record.index, ref))

        for rows in buckets.values():
            rows.sort(key=lambda item: item[0])
            for pos, (record_index, ref) in enumerate(rows):
                candidates = rows[max(0, pos - self.config.similarity_candidate_window) : pos]
                scored: list[tuple[float, NodeRef]] = []
                for candidate_index, candidate_ref in candidates:
                    score = self._similarity(records[record_index], records[candidate_index], alert_tfidf)
                    if score >= self.config.similarity_min_score:
                        scored.append((score, candidate_ref))
                scored.sort(reverse=True, key=lambda item: item[0])
                for score, candidate_ref in scored[: self.config.similarity_topk]:
                    graph.add_bidirectional(candidate_ref, "alert_feature_similar", ref, score)

    def _similarity(
        self,
        left: AlertRecord,
        right: AlertRecord,
        alert_tfidf: np.ndarray | None,
    ) -> float:
        if alert_tfidf is not None and left.index < len(alert_tfidf) and right.index < len(alert_tfidf):
            return float(np.dot(alert_tfidf[left.index], alert_tfidf[right.index]))
        left_items = set(left.source_ip_list) | set(left.destination_ip_list) | {left.event_name or ""}
        right_items = set(right.source_ip_list) | set(right.destination_ip_list) | {right.event_name or ""}
        union = left_items | right_items
        return len(left_items & right_items) / len(union) if union else 0.0
