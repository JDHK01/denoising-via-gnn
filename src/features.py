"""TF-IDF text features and compact IP-aware structural features."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import ipaddress
import math
import pickle
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from data import AlertRecord
from ip_enrichment import IpEnrichment, IpInfo


TEXT_FIELDS = (
    "event_name",
    "event_type",
    # "start_time",
    # "end_time",
    "q_body",
    "payload",
    "r_body",
)
COMMON_PORTS = {
    "20",
    "21",
    "22",
    "23",
    "25",
    "53",
    "80",
    "110",
    "123",
    "143",
    "389",
    "443",
    "445",
    "465",
    "587",
    "993",
    "995",
    "1433",
    "1521",
    "3306",
    "3389",
    "5432",
    "5900",
    "6379",
    "8080",
    "8443",
    "9200",
    "9300",
}


def alert_to_text(record: AlertRecord) -> str:
    """
    use a text to describe alert, will be used for tfidf encode

    eg.
        record = AlertRecord(
            event_name="Suspicious Login",
            event_type="auth_failure",
            start_time="2026-05-20 10:00:00",
            end_time="2026-05-20 10:05:00",
            q_body="user=admin&pass=123456",
            payload=None,
            r_body="Login failed for user admin",
            source_ip_list=("192.168.1.100", "10.0.0.5"),
            destination_ip_list=("192.168.1.1",),
            source_port_list=("54321", "12345"),
            destination_port_list=("22", "443"),
        )

        event_name:Suspicious Login event_type:auth_failure start_time:2026-05-20 10:00:00 end_time:2026-05-20 10:05:00 q_body:user=admin&pass=123456 r_body:Login failed for user admin src_ip:192.168.1.100 src_ip:10.0.0.5 dst_ip:192.168.1.1 src_port:54321 src_port:12345 dst_port:22 dst_port:443


    """
    # append "field_name"
    parts: list[str] = []
    for field_name in TEXT_FIELDS:
        value = getattr(record, field_name)
        if value is not None and str(value).strip():
            parts.append(f"{field_name}:{str(value).strip()}")

    # append "ip, port"
    for ip in record.source_ip_list:
        parts.extend((f"src_ip:{ip}")) # , f"src_subnet:{ip_prefix(ip)}"))
    for ip in record.destination_ip_list:
        parts.extend((f"dst_ip:{ip}")) # , f"dst_subnet:{ip_prefix(ip)}"))
    for port in record.source_port_list:
        parts.append(f"src_port:{port}")
    for port in record.destination_port_list:
        parts.append(f"dst_port:{port}")
    # 没这么高级, 直接 append 就 ok
    return " ".join(part for part in parts if part)#  and not part.endswith(":")) #  or "__empty_alert__"


class AlertTfidfVectorizer:
    def __init__(
        self,
        *,
        max_features: int = 512, # scale
        min_df: int = 2, # min time exist in 2 document
        max_df: float = 0.98, # max time exist in "all"*0.98 document
        ngram_range: tuple[int, int] = (1, 2), # 1 ngram and 2 ngram, token is separat by " "
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self._vectorizer = None # from sklearn

    @property
    def output_dim(self) -> int:
        if self._vectorizer is None:
            return self.max_features
        return len(self._vectorizer.get_feature_names_out())

    def fit(self, records: Sequence[AlertRecord]) -> "AlertTfidfVectorizer":
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [alert_to_text(record) for record in records]
        # don't need to be too complex
        min_df = self.min_df # min(self.min_df, max(len(corpus), 1))
        # use sklearn class
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            # sublinear, l2
            sublinear_tf=True,
            norm="l2",
            # how to seperate token
            token_pattern=r"(?u)[^\s]+", 
        )
        # event alert_text is a "document"
        self._vectorizer.fit(corpus)
        return self

    def transform(self, records: Sequence[AlertRecord]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer is not fitted")
        
        matrix = self._vectorizer.transform(alert_to_text(record) for record in records)
        return matrix.toarray().astype(np.float32, copy=False)

    def fit_transform(self, records: Sequence[AlertRecord]) -> np.ndarray:
        self.fit(records)
        return self.transform(records)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            # save as a dict
            pickle.dump(
                {
                    "config": {
                        "max_features": self.max_features,
                        "min_df": self.min_df,
                        "max_df": self.max_df,
                        "ngram_range": self.ngram_range,
                    },
                    "vectorizer": self._vectorizer,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "AlertTfidfVectorizer":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        obj = cls(**payload["config"])
        obj._vectorizer = payload["vectorizer"]
        return obj


class IpInfoTfidfVectorizer:
    """TF-IDF vectorizer for IP info (as_name and country)."""
    def __init__(
        self,
        *,
        max_features: int = 64,
        min_df: int = 2,
        max_df: float = 0.95,
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self._vectorizer: Any = None
        self._ip_cache: dict[str, str] = {}

    @property
    def output_dim(self) -> int:
        if self._vectorizer is None:
            return self.max_features
        return len(self._vectorizer.get_feature_names_out())

    def fit(self, records: Sequence[AlertRecord], ip_info_map: dict[str, IpInfo | None]) -> "IpInfoTfidfVectorizer":
        """Fit TF-IDF on IP info from training records."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        ips = collect_ips_from_records(records)

        # Build cache and corpus
        corpus = []
        for ip in ips:
            text = ip_info_text(ip, ip_info_map)
            self._ip_cache[ip] = text
            corpus.append(text)

        # Handle empty corpus case
        if not corpus or all(not t.strip() for t in corpus):
            corpus = ["__empty_ip_info__"]

        min_df = self.min_df # min(self.min_df, max(len(corpus), 1))
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=min_df,
            max_df=self.max_df,
            token_pattern=r"(?u)[^\s]+",
        )
        self._vectorizer.fit(corpus)
        return self

    def transform_ip(self, ip: str, ip_info_map: dict[str, IpInfo | None] | None = None) -> np.ndarray:
        """Transform a single IP to TF-IDF vector."""
        if self._vectorizer is None:
            raise RuntimeError("IpInfoTfidfVectorizer is not fitted")

        if ip in self._ip_cache:
            text = self._ip_cache[ip]
        elif ip_info_map is not None:
            text = ip_info_text(ip, ip_info_map)
        else:
            text = ""

        matrix = self._vectorizer.transform([text])
        return matrix.toarray().astype(np.float32, copy=False)[0]

    # def transform(self, ips: Sequence[str], ip_info_map: dict[str, IpInfo | None] | None = None) -> np.ndarray:
    #     """Transform multiple IPs to TF-IDF vectors."""
    #     if self._vectorizer is None:
    #         raise RuntimeError("IpInfoTfidfVectorizer is not fitted")

    #     vectors = []
    #     for ip in ips:
    #         vectors.append(self.transform_ip(ip, ip_info_map))
    #     return np.stack(vectors, axis=0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "config": {
                        "max_features": self.max_features,
                        "min_df": self.min_df,
                        "max_df": self.max_df,
                    },
                    "vectorizer": self._vectorizer,
                    "ip_cache": self._ip_cache,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "IpInfoTfidfVectorizer":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        obj = cls(**payload["config"])
        obj._vectorizer = payload["vectorizer"]
        obj._ip_cache = payload["ip_cache"]
        return obj


_HAS_CHINESE = re.compile(r"[一-鿿]")


def preprocess_event_text(text: str, ngram_range: tuple[int, int] = (2, 4)) -> str:
    """Split by spaces; tokens with Chinese chars are expanded to char n-grams."""
    parts: list[str] = []
    for token in text.split():
        if _HAS_CHINESE.search(token):
            for n in range(ngram_range[0], ngram_range[1] + 1):
                parts.extend(token[i : i + n] for i in range(len(token) - n + 1))
        else:
            parts.append(token)
    return " ".join(parts)


class EventTfidfVectorizer:
    """TF-IDF vectorizer for event (event_name + event_type) with Chinese-aware tokenization."""

    def __init__(self, *, max_features: int = 64, min_df: int = 1, max_df: float = 1.0, ngram_range: tuple[int, int] = (2, 4)) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self._vectorizer: Any = None

    @property
    def output_dim(self) -> int:
        if self._vectorizer is None:
            return self.max_features
        return len(self._vectorizer.get_feature_names_out())

    def fit(self, records: Sequence[AlertRecord]) -> "EventTfidfVectorizer":
        from sklearn.feature_extraction.text import TfidfVectorizer

        seen: set[tuple[str, str, str]] = set()
        corpus: list[str] = []
        for record in records:
            key = make_event_key(record)
            if key not in seen:
                seen.add(key)
                corpus.append(preprocess_event_text(" ".join(key[1:]), self.ngram_range))

        if not corpus:
            corpus = ["__empty_event__"]

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=r"(?u)[^\s]+",
            sublinear_tf=True,
            norm="l2",
        )
        self._vectorizer.fit(corpus)
        return self

    def transform_event(self, event_key: tuple[str, str, str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("EventTfidfVectorizer is not fitted")
        text = preprocess_event_text(" ".join(event_key[1:]), self.ngram_range)
        matrix = self._vectorizer.transform([text])
        return matrix.toarray().astype(np.float32, copy=False)[0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "config": {
                        "max_features": self.max_features,
                        "min_df": self.min_df,
                        "max_df": self.max_df,
                        "ngram_range": self.ngram_range,
                    },
                    "vectorizer": self._vectorizer,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "EventTfidfVectorizer":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        obj = cls(**payload["config"])
        obj._vectorizer = payload["vectorizer"]
        return obj


@dataclass
class FeatureStats:
    ip_total: Counter[tuple[str, str]]
    ip_source: Counter[tuple[str, str]]
    ip_destination: Counter[tuple[str, str]]
    ip_events: dict[tuple[str, str], set[str]]
    ip_ports: dict[tuple[str, str], set[str]]
    source_destinations: dict[tuple[str, str], set[str]]
    destination_sources: dict[tuple[str, str], set[str]]
    event_count: Counter[tuple[str, str, str]]
    event_ips: dict[tuple[str, str, str], set[str]]

    @classmethod
    def from_records(cls, records: Sequence[AlertRecord], *, isolate_platform: bool = True) -> "FeatureStats":
        ip_total: Counter[tuple[str, str]] = Counter()
        ip_source: Counter[tuple[str, str]] = Counter()
        ip_destination: Counter[tuple[str, str]] = Counter()
        ip_events: dict[tuple[str, str], set[str]] = defaultdict(set)
        ip_ports: dict[tuple[str, str], set[str]] = defaultdict(set)
        source_destinations: dict[tuple[str, str], set[str]] = defaultdict(set)
        destination_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
        event_count: Counter[tuple[str, str, str]] = Counter()
        event_ips: dict[tuple[str, str, str], set[str]] = defaultdict(set)

        for record in records:
            platform = platform_scope(record) if isolate_platform else "__global__"
            event_key = make_event_key(record) if isolate_platform else make_event_key(record, platform="__global__")
            event_text = "|".join(event_key[1:])
            ports = (*record.source_port_list, *record.destination_port_list)
            event_count[event_key] += 1
            for ip in record.source_ip_list:
                key = (platform, ip)
                ip_total[key] += 1
                ip_source[key] += 1
                ip_events[key].add(event_text)
                ip_ports[key].update(ports)
                event_ips[event_key].add(ip)
            for ip in record.destination_ip_list:
                key = (platform, ip)
                ip_total[key] += 1
                ip_destination[key] += 1
                ip_events[key].add(event_text)
                ip_ports[key].update(ports)
                event_ips[event_key].add(ip)
            for src in record.source_ip_list:
                for dst in record.destination_ip_list:
                    source_destinations[(platform, src)].add(dst)
                    destination_sources[(platform, dst)].add(src)

        return cls(
            ip_total=ip_total,
            ip_source=ip_source,
            ip_destination=ip_destination,
            ip_events=ip_events,
            ip_ports=ip_ports,
            source_destinations=source_destinations,
            destination_sources=destination_sources,
            event_count=event_count,
            event_ips=event_ips,
        )


def make_event_key(record: AlertRecord, *, platform: str | None = None) -> tuple[str, str, str]:
    return (
        platform if platform is not None else platform_scope(record),
        record.event_name or "",
        record.event_type or "",
    )


def alert_struct_features(record: AlertRecord, stats: FeatureStats, *, isolate_platform: bool = True) -> np.ndarray:
    platform = platform_scope(record) if isolate_platform else "__global__"
    src_keys = [(platform, ip) for ip in record.source_ip_list]
    dst_keys = [(platform, ip) for ip in record.destination_ip_list]
    src_seen = [stats.ip_total[key] for key in src_keys]
    dst_seen = [stats.ip_total[key] for key in dst_keys]
    src_fanout = [len(stats.source_destinations[key]) for key in src_keys]
    dst_fanin = [len(stats.destination_sources[key]) for key in dst_keys]
    source_flags = aggregate_ip_flags(record.source_ip_list)
    destination_flags = aggregate_ip_flags(record.destination_ip_list)
    q_body_empty = 0.0 if record.q_body else 1.0
    payload_empty = 0.0 if record.payload else 1.0
    r_body_empty = 0.0 if record.r_body else 1.0
    features = [
        log1p(len(record.source_ip_list)),
        log1p(len(record.destination_ip_list)),
        log1p(len(record.source_port_list)),
        log1p(len(record.destination_port_list)),
        log1p(duration_seconds(record.start_time, record.end_time)),
        q_body_empty,
        payload_empty,
        r_body_empty,
        log1p(max(src_seen, default=0)),
        log1p(max(dst_seen, default=0)),
        log1p(max(src_fanout, default=0)),
        log1p(max(dst_fanin, default=0)),
        *port_features(record.source_port_list),
        *port_features(record.destination_port_list),
        *source_flags,
        *destination_flags,
    ]
    return np.asarray(features, dtype=np.float32)


def platform_scope(record: AlertRecord) -> str:
    return record.platform_name or "__missing_platform__"


def ip_node_features(platform: str, ip: str, stats: FeatureStats, *, ip_info_tfidf: np.ndarray) -> np.ndarray:
    key = (platform, ip)
    total = stats.ip_total[key]
    source = stats.ip_source[key]
    destination = stats.ip_destination[key]
    role_denom = max(total, 1)
    features = [
        *ip_flags(ip),
        log1p(total),
        log1p(source),
        log1p(destination),
        source / role_denom,
        destination / role_denom,
        log1p(len(stats.ip_events[key])),
        log1p(len(stats.ip_ports[key])),
        *port_features(tuple(stats.ip_ports[key])),
        *ip_info_tfidf,
    ]
    return np.asarray(features, dtype=np.float32)


def event_node_features(event_key: tuple[str, str, str], stats: FeatureStats, *, event_tfidf: np.ndarray) -> np.ndarray:
    features = [
        log1p(stats.event_count[event_key]),
        log1p(len(stats.event_ips[event_key])),
        *event_tfidf,
    ]
    return np.asarray(features, dtype=np.float32)


# def ip_prefix(ip: str) -> str:
#     try:
#         parsed = ipaddress.ip_address(ip)
#     except ValueError:
#         return ip
#     if parsed.version == 4:
#         parts = ip.split(".")
#         return ".".join(parts[:3]) + ".0/24" if len(parts) == 4 else ip
#     value = int(parsed) >> 64
#     return f"{value:016x}::/64"


def aggregate_ip_flags(ips: Sequence[str]) -> list[float]:
    if not ips:
        return [0.0] * 4
    flags = [ip_flags(ip) for ip in ips]
    private = sum(row[2] for row in flags) / len(flags)
    public = sum(row[3] for row in flags) / len(flags)
    reserved = sum(row[6] for row in flags) / len(flags)
    invalid = sum(1.0 for row in flags if not any(row)) / len(flags)
    return [private, public, reserved, invalid]


def ip_flags(ip: str) -> list[float]:
    try:
        parsed = ipaddress.ip_address(ip)
    except ValueError:
        return [0.0] * 9
    return [
        1.0 if parsed.version == 4 else 0.0,
        1.0 if parsed.version == 6 else 0.0,
        1.0 if parsed.is_private else 0.0,
        1.0 if parsed.is_global else 0.0,
        1.0 if parsed.is_loopback else 0.0,
        1.0 if parsed.is_multicast else 0.0,
        1.0 if parsed.is_reserved else 0.0,
        1.0 if parsed.is_link_local else 0.0,
        1.0 if parsed.is_unspecified else 0.0,
    ]


def port_features(ports: Sequence[str]) -> list[float]:
    if not ports:
        return [0.0, 0.0, 0.0, 0.0]
    numeric: list[int] = []
    common = 0
    for port in ports:
        if port in COMMON_PORTS:
            common += 1
        try:
            numeric.append(int(port))
        except ValueError:
            pass
    if not numeric:
        return [log1p(len(ports)), 0.0, 0.0, common / len(ports)]
    privileged = sum(1 for port in numeric if port < 1024)
    ephemeral = sum(1 for port in numeric if port >= 49152)
    return [log1p(len(ports)), privileged / len(numeric), ephemeral / len(numeric), common / len(ports)]


def duration_seconds(start_time: str | None, end_time: str | None) -> float:
    start = parse_datetime(start_time)
    end = parse_datetime(end_time)
    if start is None or end is None:
        return 0.0
    return max((end - start).total_seconds(), 0.0)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def timestamp_seconds(value: str | None) -> int | None:
    parsed = parse_datetime(value)
    return int(parsed.timestamp()) if parsed is not None else None


def log1p(value: int | float | None) -> float:
    return math.log1p(max(float(value or 0.0), 0.0))


def ip_info_text(ip: str, ip_info_map: dict[str, IpInfo | None]) -> str:
    """ 
    get ip_info 
    
    eg.
        "8.8.8.8": IpInfo(as_name="Google Public DNS", country="US"),

        "asn:Google Public DNS country:US"
    """
    info = ip_info_map.get(ip)
    if info is None:
        return ""
    parts = []
    if info.as_name:
        parts.append(f"asn:{info.as_name}")
    if info.country:
        parts.append(f"country:{info.country}")
    return " ".join(parts)


def collect_ips_from_records(records: Sequence[AlertRecord]) -> list[str]:
    """Collect all unique IPs from records."""
    ips = set()
    for record in records:
        ips.update(record.source_ip_list)
        ips.update(record.destination_ip_list)
    return sorted(ips)

