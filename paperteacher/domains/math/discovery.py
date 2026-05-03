"""Discovery for the math pack.

Sources:
  1. arXiv RSS for the math archives. Default categories cover the major
     teachable subfields (NT, AG, CO, PR, AP, DG, AT, CT). Users override
     via `arxiv_categories` in the profile or the discover() argument.
  2. Semantic Scholar with `Mathematics` field-of-study, ranked by
     influential citations — surfaces papers other math papers already
     built on.

We deliberately skip HuggingFace Daily (ML-shaped community) and INSPIRE
(HEP-shaped). Users who want math-physics crossovers should run with
`domains: math, physics` and let physics's INSPIRE catch them.
"""
from __future__ import annotations

import re

from ... import paths
from .._arxiv_rss import fetch_arxiv_rss
from .._common import Candidate
from .._semantic_scholar import fetch_semantic_scholar as _s2

_VERSION_RE = re.compile(r"v\d+$")


def _canonical_arxiv_id(aid: str) -> str:
    return _VERSION_RE.sub("", (aid or "").strip())


# Default arXiv math categories. Picked for breadth across the high-traffic
# teachable subfields without flooding any one. Users can replace via profile
# or env. math.MP (mathematical physics) is intentionally out — it overlaps
# heavily with the physics pack and users running both packs would get
# duplicates.
DEFAULT_MATH_CATEGORIES = [
    "math.NT",   # number theory
    "math.AG",   # algebraic geometry
    "math.CO",   # combinatorics
    "math.PR",   # probability
    "math.AP",   # analysis of PDEs
    "math.DG",   # differential geometry
    "math.AT",   # algebraic topology
    "math.CT",   # category theory
]


# Math S2 query — broad enough to catch the field, narrow enough to stay
# relevant. The fields-of-study filter ("Mathematics") is what actually narrows;
# the query gives S2 something to score against.
_S2_QUERY = (
    "number theory OR algebraic geometry OR combinatorics OR probability "
    "OR partial differential equations OR differential geometry OR "
    "algebraic topology OR category theory"
)


async def fetch_semantic_scholar(
    window_days: int = 90,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
    today=None,
) -> list[Candidate]:
    """Recent math papers from Semantic Scholar, ranked by influential
    citations. Only returns papers with an arXiv id — the math reader is
    arXiv-shaped, so journal-only items would have nowhere to go.
    """
    return await _s2(
        query=_S2_QUERY,
        fields_of_study="Mathematics",
        id_type="ArXiv",
        canonicalize=_canonical_arxiv_id,
        window_days=window_days,
        limit=limit,
        today=today,
    )


async def discover(
    arxiv_categories: list[str] | None = None,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """Combined discovery for math. Source order (earlier wins on dupes):

        1. arXiv RSS         — canonical and fresh, per-category
        2. Semantic Scholar  — citation-velocity signal for the field

    Dedup is on canonical (version-stripped) arXiv id so the same paper
    surfacing in both sources collapses to one entry.
    """
    cats = arxiv_categories or DEFAULT_MATH_CATEGORIES
    seen_ids: set[str] = set()
    out: list[Candidate] = []

    def _add(c: Candidate) -> None:
        key = _canonical_arxiv_id(c.arxiv_id)
        if not key or key in seen_ids:
            return
        seen_ids.add(key)
        c.arxiv_id = key
        out.append(c)

    for cat in cats:
        for c in await fetch_arxiv_rss(cat, limit=limit):
            _add(c)
    for c in await fetch_semantic_scholar(limit=limit):
        _add(c)
    return out
