"""Discovery for the physics pack.

Sources:
  1. arXiv RSS for the physics archives. Default categories cover the major
     subfields the pack is meant to teach (hep-th, hep-ph, gr-qc, astro-ph,
     cond-mat, quant-ph). Users override via the `arxiv_categories` profile
     setting or the discover() argument — the same mechanism the ml pack uses.
     The fetcher itself lives in `domains/_arxiv_rss.py` since multiple packs
     share it.
  2. INSPIRE-HEP REST API for the high-energy fields. INSPIRE indexes a
     much wider corpus than arXiv RSS (conference proceedings, journal-only
     papers, theses) and ships canonical metadata for HEP. Used as an
     additional source when an active category is in the HEP family — for
     non-HEP physics (cond-mat, optics, plasma) it adds nothing.

We deliberately skip HuggingFace Daily here. HF Daily is ML-shaped — its
upvote signal doesn't reflect physics-paper relevance, and the candidates
that surface there for "physics" topics are usually physics-ML crossovers
(neural ODEs for PDEs, etc.) rather than physics proper. Users who want
those should run with `domains: ml, physics` and let ml's HF Daily fetch
catch them.
"""
from __future__ import annotations

import logging
import re

import httpx

from ... import paths
from .._arxiv_rss import extract_arxiv_id, fetch_arxiv_rss
from .._common import Candidate
from .._semantic_scholar import fetch_semantic_scholar as _s2

# Strip arXiv version suffixes for cross-source dedup. Same pattern the ml
# pack uses; the rule is universal across packs that key on arXiv ids.
_VERSION_RE = re.compile(r"v\d+$")


def _canonical_arxiv_id(aid: str) -> str:
    return _VERSION_RE.sub("", (aid or "").strip())

log = logging.getLogger(__name__)

# Default arXiv physics categories. Chosen to cover the major teachable
# subfields without flooding any one of them. Users can replace via profile
# or env. cond-mat is broad enough (8 sub-archives) that the top-level RSS
# is the right granularity; the same is true of astro-ph.
DEFAULT_PHYSICS_CATEGORIES = [
    "hep-th",        # high-energy theory (formal QFT, string theory, SUSY)
    "hep-ph",        # high-energy phenomenology (Standard Model + extensions)
    "gr-qc",         # general relativity, gravitational waves, cosmology
    "astro-ph",      # astrophysics (all sub-archives)
    "cond-mat",      # condensed matter (all sub-archives)
    "quant-ph",      # quantum physics, quantum information foundations
]

# INSPIRE-HEP indexes these arXiv archives canonically. For other physics
# categories we don't query it — the coverage is patchy and we'd dedupe
# everything against arXiv RSS anyway.
INSPIRE_HEP_CATEGORIES = {"hep-th", "hep-ph", "hep-ex", "hep-lat", "gr-qc", "nucl-th", "nucl-ex"}


async def fetch_inspire_hep(
    category: str,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """INSPIRE-HEP REST API for an HEP-family arXiv category.

    INSPIRE indexes journal-only and conference-only physics papers that
    never appear in the arXiv RSS feed, so this is genuinely additive for
    hep-th/hep-ph/gr-qc/etc. We ask for the most-recent records in the
    given arXiv category and filter to papers that carry an arXiv id —
    no-arxiv-id papers can't flow through the rest of the pipeline yet.

    API contract:
      GET https://inspirehep.net/api/literature
        ?sort=mostrecent
        &q=primarch <category>         (filter on primary arXiv archive)
        &fields=titles,authors,arxiv_eprints,abstracts
        &size=<limit>

    Query syntax note: the obvious-looking `q=arxiv:<category>` form
    silently returns zero hits (verified live), and `q=arxiv_eprints.value:
    <category>` only matches the legacy "category/yymmnnn" id form so it
    misses everything post-2007. `primarch <category>` is INSPIRE's
    official short keyword for "primary arXiv archive equals X" and is the
    only form that returns fresh records with populated arxiv_eprints.

    INSPIRE returns JSON with `hits.hits[].metadata` containing the fields
    we asked for. We're conservative on parsing: any record without a
    parseable arXiv id is dropped.
    """
    url = "https://inspirehep.net/api/literature"
    params = {
        "sort": "mostrecent",
        "q": f"primarch {category}",
        "fields": "titles,authors,arxiv_eprints,abstracts",
        "size": str(limit),
    }
    try:
        async with httpx.AsyncClient(
            timeout=30, headers={"User-Agent": paths.USER_AGENT}
        ) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        log.warning("inspire-hep fetch failed for %s: %s", category, e)
        return []
    except ValueError as e:  # JSON decode
        log.warning("inspire-hep JSON decode failed for %s: %s", category, e)
        return []

    hits = (data.get("hits") or {}).get("hits") or []
    out: list[Candidate] = []
    for hit in hits[:limit]:
        meta = hit.get("metadata") or {}
        eprints = meta.get("arxiv_eprints") or []
        if not eprints:
            continue
        aid = (eprints[0] or {}).get("value")
        if not aid or not extract_arxiv_id(aid):
            continue
        title = ""
        titles = meta.get("titles") or []
        if titles:
            title = (titles[0].get("title") or "").strip()
        summary = ""
        abstracts = meta.get("abstracts") or []
        if abstracts:
            summary = (abstracts[0].get("value") or "").strip()
        authors = [
            (a.get("full_name") or "").strip()
            for a in (meta.get("authors") or [])
            if a.get("full_name")
        ]
        out.append(
            Candidate(
                arxiv_id=aid,
                title=title,
                authors=authors,
                summary=summary,
                source=f"inspire_{category}",
                url=f"https://arxiv.org/abs/{aid}",
            )
        )
    log.info("inspire-hep %s: %d candidates", category, len(out))
    return out


# Physics S2 query — broad enough to catch the major teachable subfields
# without overweighting any single one. The fields-of-study filter
# ("Physics") is what actually narrows; the query just gives S2 something
# to score against. The shared fetcher in `domains/_semantic_scholar.py`
# handles the HTTP, ranking, and id extraction.
_S2_QUERY = (
    "quantum field theory OR general relativity OR condensed matter "
    "OR cosmology OR statistical physics OR quantum information"
)


async def fetch_semantic_scholar(
    window_days: int = 90,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
    today=None,
) -> list[Candidate]:
    """Recent physics papers from Semantic Scholar, ranked by influential
    citations. Only returns papers with an arXiv id — the physics reader
    is arXiv-shaped, so journal-only items would have nowhere to go.
    """
    return await _s2(
        query=_S2_QUERY,
        fields_of_study="Physics",
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
    """Combined discovery for physics. Source order (earlier wins on dupes):

        1. arXiv RSS         — canonical and fresh, per-category
        2. INSPIRE-HEP       — for HEP-family categories only; enrichment
                              that catches journal-route papers
        3. Semantic Scholar  — citation-velocity signal for the field

    Dedup is on canonical (version-stripped) arXiv id so the same paper
    surfacing in two or three sources collapses to one entry.
    """
    cats = arxiv_categories or DEFAULT_PHYSICS_CATEGORIES
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
    for cat in cats:
        if cat not in INSPIRE_HEP_CATEGORIES:
            continue
        for c in await fetch_inspire_hep(cat, limit=limit):
            _add(c)
    for c in await fetch_semantic_scholar(limit=limit):
        _add(c)
    return out
