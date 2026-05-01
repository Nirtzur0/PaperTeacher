"""Discover candidate papers from HF Daily Papers and arXiv RSS feeds.

ML-domain discovery sources. Other domain packs (physics, econ, philosophy,
biology, ...) ship their own discovery module and register a different
fetcher chain via the Domain protocol.
"""
from __future__ import annotations

import datetime as dt
import logging
import re

import feedparser
import httpx

from ... import paths
from .._common import Candidate

log = logging.getLogger(__name__)

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


def _extract_arxiv_id(text: str) -> str | None:
    m = ARXIV_ID_RE.search(text or "")
    return m.group(1) if m else None


async def fetch_hf_daily(
    date: dt.date | None = None,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """HF Daily Papers — community-upvoted, ranked. Most days have 5-15 papers."""
    date = date or dt.date.today()
    url = f"https://huggingface.co/api/daily_papers?date={date.isoformat()}"
    out: list[Candidate] = []
    try:
        async with httpx.AsyncClient(
            timeout=30, headers={"User-Agent": paths.USER_AGENT}
        ) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        log.warning("hf_daily fetch failed for %s: %s", date, e)
        return out
    except ValueError as e:  # JSON decode
        log.warning("hf_daily JSON decode failed for %s: %s", date, e)
        return out
    for item in data[:limit]:
        paper = item.get("paper") or {}
        aid = paper.get("id") or _extract_arxiv_id(paper.get("url", ""))
        if not aid:
            continue
        out.append(
            Candidate(
                arxiv_id=aid,
                title=(paper.get("title") or "").strip(),
                authors=[a.get("name", "") for a in paper.get("authors", [])],
                summary=(paper.get("summary") or "").strip(),
                source="hf_daily",
                score=float(paper.get("upvotes") or 0),
                url=f"https://huggingface.co/papers/{aid}",
            )
        )
    log.info("hf_daily %s: %d candidates", date, len(out))
    return out


async def fetch_arxiv_rss(
    category: str = "cs.LG",
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """arXiv RSS for one category. Useful for fields HF Daily under-covers."""
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


async def discover(
    arxiv_categories: list[str] | None = None,
    date: dt.date | None = None,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """Combined discovery. HF Daily first (already ranked), then arXiv RSS by category."""
    seen_ids: set[str] = set()
    out: list[Candidate] = []
    for c in await fetch_hf_daily(date=date, limit=limit):
        if c.arxiv_id in seen_ids:
            continue
        seen_ids.add(c.arxiv_id)
        out.append(c)
    for cat in arxiv_categories or []:
        for c in await fetch_arxiv_rss(cat, limit=limit):
            if c.arxiv_id in seen_ids:
                continue
            seen_ids.add(c.arxiv_id)
            out.append(c)
    return out
