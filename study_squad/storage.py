"""Interaction logging as JSON Lines (one record per line, append-only).

JSONL never needs the whole file rewritten, so a crash cannot corrupt earlier
records. ``normalize`` also reads the two older formats used by v1 of the
project, so old ``study_logs.json`` files can still be mined.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class LogRecord:
    query: str
    responses: dict[str, str]
    turn_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    session_id: str = ""
    intent: str = "general"
    lead_agent: str | None = None
    used_pdf: bool = False
    pages_cited: list[int] = field(default_factory=list)
    latency_ms: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    model: str = ""
    sample: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Convert any known log shape into the current schema. Returns None if unusable."""
    if not isinstance(raw, dict):
        return None
    query = raw.get("query") or raw.get("user_query") or raw.get("topic")
    if not isinstance(query, str) or not query.strip():
        return None
    responses = raw.get("responses") or raw.get("agents") or {}
    if not isinstance(responses, dict):
        responses = {}
    record = LogRecord(query=query.strip(), responses={k: str(v) for k, v in responses.items()})
    data = record.to_dict()
    data["timestamp"] = raw.get("timestamp") or ""
    for key in data:
        if key in raw and key not in {"query", "responses"}:
            data[key] = raw[key]
    if not data["errors"]:
        data["errors"] = {k: v for k, v in data["responses"].items() if str(v).startswith("Error")}
    return data


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a ``.jsonl`` log, or a legacy ``.json`` list, and normalise every record."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    raw_items: Iterable[Any]
    if text.startswith("["):
        try:
            raw_items = json.loads(text)
        except json.JSONDecodeError:
            return []
    else:
        raw_items = []
        for line in text.splitlines():
            try:
                raw_items.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a half-written line instead of losing the file
    return [rec for item in raw_items if (rec := normalize(item)) is not None]


class LogStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: LogRecord | FeedbackRecord | dict[str, Any]) -> None:
        data = record if isinstance(record, dict) else record.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")

    def load(self) -> list[dict[str, Any]]:
        return read_records(self.path)


@dataclass
class FeedbackRecord:
    """A thumbs up (1) or down (0) that a student gave one agent's answer."""

    turn_id: str
    agent: str
    rating: int
    query: str = ""
    intent: str = "general"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_feedback(path: str | Path) -> list[dict[str, Any]]:
    """Read feedback events; the latest rating per (turn, agent) wins."""
    path = Path(path)
    if not path.exists():
        return []
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and {"turn_id", "agent", "rating"} <= item.keys():
            latest[(item["turn_id"], item["agent"])] = item
    return list(latest.values())
