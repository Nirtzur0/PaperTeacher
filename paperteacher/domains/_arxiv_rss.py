"""Shared helper: fetch one arXiv RSS feed by category.

Lifted out of `ml/discovery.py` once the second pack (econ) needed the same
helper for econ.* and q-fin.* categories. The function is mechanical XML
parsing — pure plumbing, no domain knowledge — so it lives here rather
than being imported across sibling packs.
"""
from __future__ import annotations

import logging
import re

import feedparser
import httpx

from .. import paths
from ._common import Candidate

log = logging.getLogger(__name__)

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


def extract_arxiv_id(text: str) -> str | None:
    m = ARXIV_ID_RE.search(text or "")
    return m.group(1) if m else None


async def fetch_arxiv_rss(
    category: str,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """arXiv RSS for one category. Source tag is `arxiv_<category>`."""
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
        aid = extract_arxiv_id(entry.get("id", "")) or extract_arxiv_id(entry.get("link", ""))
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
