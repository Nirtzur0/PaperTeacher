"""Discovery for the physics pack.

Sources:
  1. arXiv RSS for the physics archives. Default categories cover the major
     subfields the pack is meant to teach (hep-th, hep-ph, gr-qc, astro-ph,
     cond-mat, quant-ph). Users override via the `arxiv_categories` profile
     setting or the discover() argument — the same mechanism the ml pack uses.
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

import feedparser
import httpx

from ... import paths
from .._common import Candidate

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

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


def _extract_arxiv_id(text: str) -> str | None:
    m = ARXIV_ID_RE.search(text or "")
    return m.group(1) if m else None


async def fetch_arxiv_rss(
    category: str,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """arXiv RSS for one physics category. Mirrors the ml pack's fetcher
    so behaviour is uniform across packs (same redirect handling, same
    user-agent, same id extraction).

    The `rss.arxiv.org` host is the re-implemented feed introduced in
    early 2024; the legacy `export.arxiv.org/rss/<cat>` URL still works
    via redirect, so we use the legacy form to match the ml pack and
    let arXiv's own redirect take care of routing.
    """
    url = f"http://export.arxiv.org/rss/{category}"
    body: str
    try:
        async with httpx.AsyncClient(
            timeout=30, follow_redirects=True, headers={"User-Agent": paths.USER_AGENT}
        ) as client:
            r = await client.get(url)
            r.raise_for_status()
            body = r.text
    except httpx.HTTPError as e:
        log.warning("arxiv RSS fetch failed for %s: %s", category, e)
        return []
    feed = feedparser.parse(body)
    out: list[Candidate] = []
    for entry in feed.entries[:limit]:
        aid = _extract_arxiv_id(entry.get("id", "")) or _extract_arxiv_id(entry.get("link", ""))
        if not aid:
            continue
        out.append(
            Candidate(
                arxiv_id=aid,
                title=(entry.get("title") or "").strip(),
                authors=[a.get("name", "") for a in entry.get("authors", [])],
                summary=re.sub(r"<[^>]+>", "", entry.get("summary", "")).strip(),
                source=f"arxiv_{category}",
                url=f"https://arxiv.org/abs/{aid}",
            )
        )
    log.info("arxiv_rss %s: %d candidates", category, len(out))
    return out


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
        &q=arxiv:<category>            (full-text search on the arxiv field)
        &fields=titles,authors,arxiv_eprints,abstracts
        &size=<limit>

    INSPIRE returns JSON with `hits.hits[].metadata` containing the fields
    we asked for. We're conservative on parsing: any record without a
    parseable arXiv id is dropped.
    """
    url = "https://inspirehep.net/api/literature"
    params = {
        "sort": "mostrecent",
        "q": f"arxiv:{category}",
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
        if not aid or not _extract_arxiv_id(aid):
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


async def discover(
    arxiv_categories: list[str] | None = None,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """Combined discovery for physics. Walks the configured arXiv physics
    categories in order, then for HEP-family categories also queries
    INSPIRE. Dedupes by arxiv_id; earlier sources win on duplicates.

    Order matters: arXiv RSS is the primary signal (canonical, fresh),
    INSPIRE is enrichment for the HEP family (catches journal-route
    papers that didn't preprint).
    """
    cats = arxiv_categories or DEFAULT_PHYSICS_CATEGORIES
    seen_ids: set[str] = set()
    out: list[Candidate] = []
    for cat in cats:
        for c in await fetch_arxiv_rss(cat, limit=limit):
            if c.arxiv_id in seen_ids:
                continue
            seen_ids.add(c.arxiv_id)
            out.append(c)
    for cat in cats:
        if cat not in INSPIRE_HEP_CATEGORIES:
            continue
        for c in await fetch_inspire_hep(cat, limit=limit):
            if c.arxiv_id in seen_ids:
                continue
            seen_ids.add(c.arxiv_id)
            out.append(c)
    return out
