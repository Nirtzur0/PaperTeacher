"""Persist per-paper outlines + scripts so the pipeline stages can hand off."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

DEFAULT_DIR = Path.home() / ".paperteacher"


def _outlines_dir() -> Path:
    d = DEFAULT_DIR / "outlines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _scripts_dir() -> Path:
    d = DEFAULT_DIR / "scripts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_outline(arxiv_id: str, outline_yaml: str) -> Path:
    p = _outlines_dir() / f"{arxiv_id}.yaml"
    p.write_text(outline_yaml)
    _record(arxiv_id, "outline_saved")
    return p


def load_outline(arxiv_id: str) -> str | None:
    p = _outlines_dir() / f"{arxiv_id}.yaml"
    return p.read_text() if p.exists() else None


def save_script(arxiv_id: str, script: str) -> Path:
    p = _scripts_dir() / f"{arxiv_id}.txt"
    p.write_text(script)
    _record(arxiv_id, "script_saved")
    return p


def load_script(arxiv_id: str) -> str | None:
    p = _scripts_dir() / f"{arxiv_id}.txt"
    return p.read_text() if p.exists() else None


def _record(arxiv_id: str, event: str) -> None:
    log = DEFAULT_DIR / "pipeline.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps({
            "arxiv_id": arxiv_id,
            "event": event,
            "ts": dt.datetime.utcnow().isoformat() + "Z",
        }) + "\n")
