"""Unified persistence: seen-papers, outlines, scripts, audits, event log.

All paths come from `paths.py`. All write paths emit a structured event to
the pipeline log so failures can be traced retroactively.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from . import paths
from .models import AuditReport, Outline, parse_audit, parse_outline

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(event: str, **fields: object) -> None:
    paths.ensure_layout()
    with paths.EVENT_LOG.open("a") as f:
        f.write(json.dumps({"event": event, "ts": _now(), **fields}) + "\n")


# ---- seen-papers ----------------------------------------------------------


def list_seen() -> list[dict]:
    if not paths.SEEN_FILE.exists():
        return []
    return [json.loads(line) for line in paths.SEEN_FILE.read_text().splitlines() if line.strip()]


def is_seen(arxiv_id: str) -> bool:
    return any(row.get("arxiv_id") == arxiv_id for row in list_seen())


def mark_seen(arxiv_id: str, *, title: str = "", note: str = "") -> Path:
    paths.ensure_layout()
    row = {"arxiv_id": arxiv_id, "title": title, "note": note, "ts": _now()}
    with paths.SEEN_FILE.open("a") as f:
        f.write(json.dumps(row) + "\n")
    _emit("seen_marked", arxiv_id=arxiv_id, note=note)
    return paths.SEEN_FILE


# ---- outlines -------------------------------------------------------------


def save_outline(arxiv_id: str, outline_yaml: str) -> tuple[Path, Outline]:
    """Validate + persist. Raises ParseError on schema mismatch.

    The saved form is the *canonical* re-serialization, not the raw input —
    field order and formatting become consistent across stages.
    """
    paths.ensure_layout()
    outline = parse_outline(outline_yaml)
    p = paths.OUTLINES_DIR / f"{arxiv_id}.yaml"
    p.write_text(outline.to_yaml())
    _emit(
        "outline_saved",
        arxiv_id=arxiv_id,
        critical=len(outline.critical_ids()),
        important=len(outline.important_ids()),
        equations=len(outline.key_equations),
        concepts=len(outline.key_concepts),
    )
    log.info("saved outline for %s (%d equations, %d concepts)",
             arxiv_id, len(outline.key_equations), len(outline.key_concepts))
    return p, outline


def load_outline(arxiv_id: str) -> Outline | None:
    p = paths.OUTLINES_DIR / f"{arxiv_id}.yaml"
    if not p.exists():
        return None
    return parse_outline(p.read_text())


def load_outline_yaml(arxiv_id: str) -> str | None:
    """Raw YAML form — used when feeding back into a prompt template."""
    p = paths.OUTLINES_DIR / f"{arxiv_id}.yaml"
    return p.read_text() if p.exists() else None


# ---- scripts --------------------------------------------------------------


def save_script(arxiv_id: str, script: str) -> Path:
    paths.ensure_layout()
    p = paths.SCRIPTS_DIR / f"{arxiv_id}.txt"
    p.write_text(script)
    _emit("script_saved", arxiv_id=arxiv_id, words=len(script.split()))
    log.info("saved script for %s (%d words)", arxiv_id, len(script.split()))
    return p


def load_script(arxiv_id: str) -> str | None:
    p = paths.SCRIPTS_DIR / f"{arxiv_id}.txt"
    return p.read_text() if p.exists() else None


# ---- audits ---------------------------------------------------------------


def save_audit(arxiv_id: str, audit_yaml: str) -> tuple[Path, AuditReport]:
    paths.ensure_layout()
    audit = parse_audit(audit_yaml)
    p = paths.AUDITS_DIR / f"{arxiv_id}.yaml"
    p.write_text(audit.to_yaml())
    _emit(
        "audit_saved",
        arxiv_id=arxiv_id,
        coverage_status=audit.coverage_status,
        recommendation=audit.recommendation,
        missing=len(audit.items_missing),
        glossed=len(audit.items_glossed),
    )
    log.info("saved audit for %s: %s (%s)",
             arxiv_id, audit.recommendation, audit.coverage_status)
    return p, audit


def load_audit(arxiv_id: str) -> AuditReport | None:
    p = paths.AUDITS_DIR / f"{arxiv_id}.yaml"
    if not p.exists():
        return None
    return parse_audit(p.read_text())
