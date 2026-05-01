"""Unified persistence: seen-papers, outlines, scripts, audits, event log.

All paths come from `paths.py`. All write paths emit a structured event to
the pipeline log so failures can be traced retroactively.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from . import paths
from .domain import active_domain
from .domains._common import AuditReport

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


def seen_ids() -> set[str]:
    """One-shot set of all seen arxiv_ids. Use this when filtering a batch
    instead of calling `is_seen` per candidate (which re-reads the file each time).
    """
    return {row["arxiv_id"] for row in list_seen() if "arxiv_id" in row}


def is_seen(arxiv_id: str) -> bool:
    return arxiv_id in seen_ids()


def mark_seen(
    arxiv_id: str,
    *,
    title: str = "",
    note: str = "",
    tags: list[str] | None = None,
) -> Path:
    """Record a delivered paper. `tags` is a list of free-form topic tags
    (e.g. ["info-geometry", "optimization", "transformers"]) used by the host
    to balance topic distribution across deliveries.
    """
    paths.ensure_layout()
    row = {
        "arxiv_id": arxiv_id,
        "title": title,
        "note": note,
        "tags": list(tags or []),
        "ts": _now(),
    }
    with paths.SEEN_FILE.open("a") as f:
        f.write(json.dumps(row) + "\n")
    _emit("seen_marked", arxiv_id=arxiv_id, note=note, tags=row["tags"])
    return paths.SEEN_FILE


# ---- skipped-papers (the backlog) -----------------------------------------


def list_skipped() -> list[dict]:
    if not paths.SKIPPED_FILE.exists():
        return []
    return [
        json.loads(line)
        for line in paths.SKIPPED_FILE.read_text().splitlines()
        if line.strip()
    ]


def skipped_ids() -> set[str]:
    return {row["arxiv_id"] for row in list_skipped() if "arxiv_id" in row}


def mark_skipped(
    arxiv_id: str,
    *,
    title: str = "",
    tags: list[str] | None = None,
    reason: str = "",
) -> Path:
    """Record a candidate that was considered but NOT delivered. Lets the host
    revisit the backlog later for topic balance and avoids re-evaluating the
    same paper from scratch every day.
    """
    paths.ensure_layout()
    row = {
        "arxiv_id": arxiv_id,
        "title": title,
        "tags": list(tags or []),
        "reason": reason,
        "ts": _now(),
    }
    with paths.SKIPPED_FILE.open("a") as f:
        f.write(json.dumps(row) + "\n")
    _emit("skipped_marked", arxiv_id=arxiv_id, reason=reason, tags=row["tags"])
    return paths.SKIPPED_FILE


def topic_distribution(window: int = 30) -> dict[str, int]:
    """Tag → count over the last `window` delivered papers. Use this when
    picking the next paper to balance underrepresented topics.
    """
    rows = list_seen()[-window:]
    counts: dict[str, int] = {}
    for r in rows:
        for t in r.get("tags", []):
            counts[t] = counts.get(t, 0) + 1
    return counts


# ---- outlines -------------------------------------------------------------


def save_outline(arxiv_id: str, outline_yaml: str) -> tuple[Path, BaseModel]:
    """Validate + persist. Raises ParseError on schema mismatch.

    Outline schema and parser come from the active domain pack so different
    subjects (ml, physics, philosophy, ...) can plug in their own typed
    contracts without touching this layer. The saved form is the *canonical*
    re-serialization — field order and formatting become consistent across stages.
    """
    paths.ensure_layout()
    domain = active_domain()
    outline = domain.parse_outline(outline_yaml)
    p = paths.OUTLINES_DIR / f"{arxiv_id}.yaml"
    p.write_text(outline.to_yaml())
    # Domain-specific stat fields (equations/concepts for ML; arguments/sources
    # for humanities; etc.). Each domain's outline can implement summary helpers
    # and we surface whatever the model exposes — fall back to None if absent.
    extras = {}
    for fn in ("critical_ids", "important_ids"):
        if callable(getattr(outline, fn, None)):
            extras[fn[:-4] + "s"] = len(getattr(outline, fn)())  # critical, important
    _emit("outline_saved", arxiv_id=arxiv_id, domain=domain.name, **extras)
    log.info("saved outline for %s (domain=%s)", arxiv_id, domain.name)
    return p, outline


def load_outline(arxiv_id: str) -> BaseModel | None:
    p = paths.OUTLINES_DIR / f"{arxiv_id}.yaml"
    if not p.exists():
        return None
    return active_domain().parse_outline(p.read_text())


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
    audit = active_domain().parse_audit(audit_yaml)
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
    return active_domain().parse_audit(p.read_text())
