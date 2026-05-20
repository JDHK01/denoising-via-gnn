"""Local IP enrichment adapter for artifacts/ip_info/ip_info.db."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "resource" / "ip_info" / "ip_info.db"


@dataclass(frozen=True, slots=True)
class IpInfo:
    ip: str
    asn: str | None
    as_name: str | None
    as_domain: str | None
    country_code: str | None
    country: str | None
    continent_code: str | None
    continent: str | None
    error: str | None


class IpEnrichment:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path or DEFAULT_DB_PATH)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def lookup(self, ip: str) -> IpInfo | None:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT ip, asn, as_name, as_domain, country_code, country, continent_code, continent, error
            FROM ip_info
            WHERE ip = ?
            """,
            (ip,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return IpInfo(*row)

    def lookup_batch(self, ips: list[str]) -> dict[str, IpInfo | None]:
        if not ips:
            return {}
        conn = self._connect()
        cursor = conn.cursor()
        placeholders = ", ".join("?" for _ in ips)
        cursor.execute(
            f"""
            SELECT ip, asn, as_name, as_domain, country_code, country, continent_code, continent, error
            FROM ip_info
            WHERE ip IN ({placeholders})
            """,
            ips,
        )
        result: dict[str, IpInfo | None] = {ip: None for ip in ips}
        for row in cursor.fetchall():
            info = IpInfo(*row)
            result[info.ip] = info
        return result

    def __enter__(self) -> IpEnrichment:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
