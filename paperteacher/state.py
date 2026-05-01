"""Track which papers have been delivered, so we don't repeat."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

DEFAULT_STATE_DIR = Path.home() / ".paperteacher"
SEEN_FILE = "seen.jsonl"


def _state_dir() -> Path:
    d = DEFAULT_STATE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_seen() -> list[dict]:
    p = _state_dir() / SEEN_FILE
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def is_seen(arxiv_id: str) -> bool:
    return any(row.get("arxiv_id") == arxiv_id for row in list_seen())


def mark_seen(arxiv_id: str, title: str = "", note: str = "") -> None:
    p = _state_dir() / SEEN_FILE
    row = {
        "arxiv_id": arxiv_id,
        "title": title,
        "note": note,
        "ts": dt.datetime.utcnow().isoformat() + "Z",
    }
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
