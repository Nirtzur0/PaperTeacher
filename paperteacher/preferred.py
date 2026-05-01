"""Preferred-authors allowlist. Off by default.

Activates when a YAML config exists at PAPERTEACHER_PREFERRED (default
~/.paperteacher/preferred.yaml). When a discovered candidate has at least
one author whose name contains a configured needle (case-insensitive
substring), the candidate's discovery score is increased by `boost` and
the candidate list is re-sorted, so the host LLM's selection step sees
preferred work near the top.

Why authors and not affiliations: arXiv RSS exposes author names but not
institution. Listing key researchers from a lab achieves the same effect
with the data we already have.

Schema (see config/preferred.example.yaml):

    authors:
      - Chris Olah
      - Neel Nanda
    boost: 100.0
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import yaml

from . import paths
from .domains._common import Candidate

log = logging.getLogger(__name__)

DEFAULT_BOOST = 100.0


@dataclass(frozen=True)
class Preferred:
    authors: tuple[str, ...] = ()
    boost: float = DEFAULT_BOOST

    def matches(self, cand: Candidate) -> bool:
        if not self.authors or not cand.authors:
            return False
        cand_lower = [a.lower() for a in cand.authors]
        for needle in self.authors:
            n = needle.lower()
            if any(n in a for a in cand_lower):
                return True
        return False


def load() -> Preferred | None:
    """Read the preferred-list config; return None if absent or empty."""
    p = paths.PREFERRED_PATH
    if not p.exists():
        return None
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        log.warning("preferred.yaml parse failed: %s — feature disabled", e)
        return None
    if not isinstance(data, dict):
        log.warning("preferred.yaml: top-level must be a mapping — feature disabled")
        return None
    authors = tuple(
        s for s in (str(x).strip() for x in (data.get("authors") or [])) if s
    )
    if not authors:
        return None
    boost = float(data.get("boost") or DEFAULT_BOOST)
    log.info("preferred-list active: %d authors, boost=%.1f", len(authors), boost)
    return Preferred(authors=authors, boost=boost)


def apply(candidates: list[Candidate], pref: Preferred) -> list[Candidate]:
    """Boost matching candidates' scores in place, then sort by score desc."""
    for c in candidates:
        if pref.matches(c):
            c.score = (c.score or 0.0) + pref.boost
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
