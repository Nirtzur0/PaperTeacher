"""Discover candidate papers from arXiv econ/q-fin RSS and NBER's new-papers feed.

Source priority (best-first):
  1. NBER `/rss/new.xml`           — top-tier US-affiliated econ research,
                                     daily updates US weekday mornings.
  2. arXiv `econ.*` + `q-fin.*` RSS — broader, faster, includes theory and
                                     quantitative finance preprints.

SSRN and IMF were considered and dropped: SSRN is Akamai-walled to non-browser
clients and IMF's RSS endpoint 403s. Their content reaches us indirectly via
RePEc / NBER mirroring or via direct download once a user already has the id.

NBER ids look like `w31234`; arXiv ids look like `2603.20105`. Both shapes
are filesystem-safe and disjoint, so the reader can dispatch on format
without a prefix.

Why no Semantic Scholar here (yet)
-----------------------------------
The shared `domains/_semantic_scholar.py` fetcher is parametric on id_type
and the ml/physics/neuro packs wire it in. Econ doesn't, on purpose: S2's
``Economics`` field-of-study results are mostly DOI-keyed journal papers,
but this pack's reader handles only arXiv ids and NBER working-paper ids
— DOI handling would require extending the reader (PDF parsing or DOI
redirection) or filtering S2 to only items with arXiv ids, which would
yield a thin trickle that arXiv RSS already catches. Plumbing it in as
currently shaped would be net noise. To enable: extend
``domains/econ/reader.py`` to accept DOIs, then add a
``fetch_semantic_scholar_econ`` mirroring the neuro pack's wiring.
"""
from __future__ import annotations

import datetime as dt
import logging
import re

import feedparser
import httpx

from ... import paths
from .._arxiv_rss import fetch_arxiv_rss
from .._common import Candidate

log = logging.getLogger(__name__)

# Default categories when the profile doesn't override. Three econ + eight
# q-fin. econ has only GN/TH/EM; q-fin has eight subcategories.
DEFAULT_ECON_CATEGORIES = [
    "econ.GN", "econ.TH", "econ.EM",
    "q-fin.GN", "q-fin.EC", "q-fin.RM", "q-fin.MF",
    "q-fin.CP", "q-fin.PR", "q-fin.ST", "q-fin.TR",
]

NBER_RSS_URL = "https://www.nber.org/rss/new.xml"
NBER_ID_RE = re.compile(r"/papers/(w\d{4,5})")


def _extract_nber_id(text: str) -> str | None:
    """Pull the NBER working-paper id (e.g. 'w31234') out of an RSS link or guid."""
    m = NBER_ID_RE.search(text or "")
    return m.group(1) if m else None


async def fetch_nber_new(limit: int = paths.DEFAULT_DISCOVERY_LIMIT) -> list[Candidate]:
    """NBER's new-working-papers RSS. Returns at most `limit` candidates,
    ordered as the feed serves them (newest first). Item shape: title,
    link to abstract page, description = full abstract, pubDate.
    """
    out: list[Candidate] = []
    try:
        async with httpx.AsyncClient(
            timeout=30, headers={"User-Agent": paths.USER_AGENT}
        ) as client:
            r = await client.get(NBER_RSS_URL, follow_redirects=True)
            r.raise_for_status()
            body = r.text
    except httpx.HTTPError as e:
        log.warning("nber RSS fetch failed: %s", e)
        return out
    feed = feedparser.parse(body)
    for entry in feed.entries[:limit]:
        link = entry.get("link") or entry.get("id") or ""
        nid = _extract_nber_id(link) or _extract_nber_id(entry.get("guid", ""))
        if not nid:
            continue
        # NBER RSS stores authors as a single string in dc:creator; feedparser
        # may surface them under .authors with only `name`, or under .author
        # as a flat string. Handle both.
        authors_raw = entry.get("authors") or []
        authors = [a.get("name", "") for a in authors_raw if a.get("name")]
        if not authors and entry.get("author"):
            authors = [entry["author"]]
        out.append(
            Candidate(
                arxiv_id=nid,
                title=(entry.get("title") or "").strip(),
                authors=authors,
                summary=re.sub(r"<[^>]+>", "", entry.get("summary", "")).strip(),
                source="nber_new",
                url=f"https://www.nber.org/papers/{nid}",
            )
        )
    log.info("nber_new: %d candidates", len(out))
    return out


async def discover(
    arxiv_categories: list[str] | None = None,
    date: dt.date | None = None,  # accepted for parity with ML; ignored (NBER RSS has no date filter)
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """Combined econ discovery. NBER first (curated, daily), then arXiv RSS
    across the configured econ + q-fin categories. De-duplicates by id.
    """
    del date  # not supported by NBER RSS
    seen_ids: set[str] = set()
    out: list[Candidate] = []
    for c in await fetch_nber_new(limit=limit):
        if c.arxiv_id in seen_ids:
            continue
        seen_ids.add(c.arxiv_id)
        out.append(c)
    cats = arxiv_categories if arxiv_categories is not None else DEFAULT_ECON_CATEGORIES
    for cat in cats:
        for c in await fetch_arxiv_rss(cat, limit=limit):
            if c.arxiv_id in seen_ids:
                continue
            seen_ids.add(c.arxiv_id)
            out.append(c)
    return out
