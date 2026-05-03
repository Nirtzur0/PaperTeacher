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
from .domain import domain_for
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


def excluded_ids() -> set[str]:
    """Union of seen + skipped — the discovery exclusion set. Use this
    instead of computing the union ad-hoc so server.py and cli.py stay
    consistent (and so the rule has one definition, not two)."""
    return seen_ids() | skipped_ids()


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

    Side effect: auto-claims the paper via `mark_seen` if it isn't already
    in the seen set. This prevents the same paper looping back tomorrow if
    a later step (plan, script, audit, render, delivery) fails. The skill's
    explicit step-7a remains a no-op duplicate; the canonical step-20
    `mark_seen` call still runs to attach final tags + audit:complete note.
    """
    paths.ensure_layout()
    domain = domain_for(arxiv_id)
    outline = domain.parse_outline(outline_yaml)
    p = paths.OUTLINES_DIR / f"{arxiv_id}.yaml"
    p.write_text(outline.to_yaml())
    if arxiv_id not in seen_ids():
        title = getattr(outline, "title", "") or ""
        mark_seen(arxiv_id, title=title, note="outline:saved")
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
    return domain_for(arxiv_id).parse_outline(p.read_text())


def load_outline_yaml(arxiv_id: str) -> str | None:
    """Raw YAML form — used when feeding back into a prompt template."""
    p = paths.OUTLINES_DIR / f"{arxiv_id}.yaml"
    return p.read_text() if p.exists() else None


# ---- plans (stage 1.5) ---------------------------------------------------
#
# The plan is an OPTIONAL macro-structure artifact between extract and teach.
# Domain packs that haven't opted into the planner stage need not implement
# `parse_plan` — save_plan raises NotImplementedError, load_plan quietly
# returns None, and the teach prompt falls back to its default arc.


def save_plan(arxiv_id: str, plan_yaml: str) -> tuple[Path, BaseModel]:
    paths.ensure_layout()
    domain = domain_for(arxiv_id)
    parser = getattr(domain, "parse_plan", None)
    if parser is None:
        raise NotImplementedError(
            f"domain {domain.name!r} does not implement parse_plan — "
            f"the planner stage is opt-in per pack"
        )
    plan = parser(plan_yaml)
    p = paths.PLANS_DIR / f"{arxiv_id}.yaml"
    p.write_text(plan.to_yaml())
    extras: dict = {}
    arc = getattr(plan, "arc", None)
    if isinstance(arc, list):
        extras["segments"] = len(arc)
    takes = getattr(plan, "takes", None)
    if isinstance(takes, list):
        extras["takes"] = len(takes)
    _emit("plan_saved", arxiv_id=arxiv_id, domain=domain.name, **extras)
    log.info("saved plan for %s (domain=%s)", arxiv_id, domain.name)
    return p, plan


def load_plan(arxiv_id: str) -> BaseModel | None:
    p = paths.PLANS_DIR / f"{arxiv_id}.yaml"
    if not p.exists():
        return None
    parser = getattr(domain_for(arxiv_id), "parse_plan", None)
    if parser is None:
        return None
    return parser(p.read_text())


def load_plan_yaml(arxiv_id: str) -> str | None:
    p = paths.PLANS_DIR / f"{arxiv_id}.yaml"
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
    audit = domain_for(arxiv_id).parse_audit(audit_yaml)
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
    return domain_for(arxiv_id).parse_audit(p.read_text())
