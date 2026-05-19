"""Data loading for the all-in alert dataset."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal


DatasetSplit = Literal["train", "test"]
SPLIT_DIRS: dict[DatasetSplit, str] = {
    "train": "train_data",
    "test": "test_data",
}
NULLISH = {"", "--", "null", "none", "nan"}


@dataclass(frozen=True, slots=True)
class AlertRecord:
    index: int
    platform_name: str | None
    event_name: str | None
    event_type: str | None
    start_time: str | None
    end_time: str | None
    source_ip_list: tuple[str, ...]
    destination_ip_list: tuple[str, ...]
    source_port_list: tuple[str, ...]
    destination_port_list: tuple[str, ...]
    q_body: str | None
    payload: str | None
    r_body: str | None
    label: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_label(op_state: int | None, rule_decide: str | None) -> int | None:
    if op_state != 8:
        return None
    return 1 if rule_decide == "real_attack" else 0


def iter_dataset_files(dataset_dir: str | Path, split: DatasetSplit) -> Iterator[Path]:
    split_dir = Path(dataset_dir) / SPLIT_DIRS[split]
    yield from sorted(split_dir.glob("*.json"))


def iter_records(
    dataset_dir: str | Path,
    split: DatasetSplit,
    *,
    max_records: int | None = None,
) -> Iterator[AlertRecord]:
    count = 0
    for path in iter_dataset_files(dataset_dir, split):
        for raw in _iter_json_records(path):
            yield _parse_record(raw, count)
            count += 1
            if max_records is not None and count >= max_records:
                return


def load_records(
    dataset_dir: str | Path,
    split: DatasetSplit,
    *,
    max_records: int | None = None,
) -> list[AlertRecord]:
    return list(iter_records(dataset_dir, split, max_records=max_records))


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = None
        for line in handle:
            if line.strip():
                first = line
                break
        if first is None:
            return
        if first.lstrip().startswith("["):
            rows = json.loads(first + handle.read())
            for row in rows:
                yield row
            return
        yield json.loads(first)
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _parse_record(row: dict[str, Any], index: int) -> AlertRecord:
    op_state = normalize_int(row.get("op_state"))
    rule_decide = normalize_text(row.get("rule_decide"))
    return AlertRecord(
        index=index,
        platform_name=normalize_text(row.get("platform_name")),
        event_name=normalize_text(row.get("event_name")),
        event_type=normalize_text(row.get("event_type")),
        start_time=normalize_text(row.get("start_time")),
        end_time=normalize_text(row.get("end_time")),
        source_ip_list=normalize_tuple(row.get("source_ip_list")),
        destination_ip_list=normalize_tuple(row.get("destination_ip_list")),
        source_port_list=normalize_tuple(row.get("source_port_list")),
        destination_port_list=normalize_tuple(row.get("destination_port_list")),
        q_body=normalize_text(row.get("q_body"), preserve_whitespace=True),
        payload=normalize_text(row.get("payload"), preserve_whitespace=True),
        r_body=normalize_text(row.get("r_body"), preserve_whitespace=True),
        label=infer_label(op_state, rule_decide),
    )


def normalize_text(value: Any, *, preserve_whitespace: bool = False) -> str | None:
    if value is None:
        return None
    text = str(value)
    stripped = text.strip()
    if stripped.lower() in NULLISH:
        return None
    return text if preserve_whitespace else stripped


def normalize_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple)) else (value,)
    out: list[str] = []
    for item in values:
        text = normalize_text(item)
        if text is not None:
            out.append(text)
    return tuple(out)


def normalize_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = normalize_text(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None
