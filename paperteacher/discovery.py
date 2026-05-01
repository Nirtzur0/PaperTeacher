"""Discover candidate papers from HF Daily Papers and arXiv RSS feeds."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import feedparser
import httpx

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})")


@dataclass
class Candidate:
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    source: str
    score: float = 0.0
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "summary": self.summary,
            "source": self.source,
            "score": self.score,
            "url": self.url,
        }


def _extract_arxiv_id(text: str) -> str | None:
    m = ARXIV_ID_RE.search(text or "")
    return m.group(1) if m else None


async def fetch_hf_daily(date: dt.date | None = None, limit: int = 20) -> list[Candidate]:
    """Fetch HF Daily Papers. The endpoint returns ranked, community-upvoted papers."""
    date = date or dt.date.today()
    url = f"https://huggingface.co/api/daily_papers?date={date.isoformat()}"
    out: list[Candidate] = []
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return out
    for item in data[:limit]:
        paper = item.get("paper") or {}
        aid = paper.get("id") or _extract_arxiv_id(paper.get("url", ""))
        if not aid:
            continue
        out.append(
            Candidate(
                arxiv_id=aid,
                title=paper.get("title", "").strip(),
                authors=[a.get("name", "") for a in paper.get("authors", [])],
                summary=paper.get("summary", "").strip(),
                source="hf_daily",
                score=float(paper.get("upvotes", 0)),
                url=f"https://huggingface.co/papers/{aid}",
            )
        )
    return out


async def fetch_arxiv_rss(category: str = "cs.LG", limit: int = 20) -> list[Candidate]:
    """Fetch arXiv RSS for a category. Useful for fields HF Daily under-covers."""
    url = f"http://export.arxiv.org/rss/{category}"
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        body = r.text
    feed = feedparser.parse(body)
    out: list[Candidate] = []
    for entry in feed.entries[:limit]:
        aid = _extract_arxiv_id(entry.get("id", "")) or _extract_arxiv_id(entry.get("link", ""))
        if not aid:
            continue
        out.append(
            Candidate(
                arxiv_id=aid,
                title=entry.get("title", "").strip(),
                authors=[a.get("name", "") for a in entry.get("authors", [])],
                summary=re.sub(r"<[^>]+>", "", entry.get("summary", "")).strip(),
                source=f"arxiv_{category}",
                url=f"https://arxiv.org/abs/{aid}",
            )
        )
    return out


async def discover(
    fields: list[str] | None = None,
    arxiv_categories: list[str] | None = None,
    date: dt.date | None = None,
    limit: int = 20,
) -> list[Candidate]:
    """Combined discovery. HF Daily first (already ranked), then arXiv RSS by category."""
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in await fetch_hf_daily(date=date, limit=limit):
        if c.arxiv_id in seen:
            continue
        seen.add(c.arxiv_id)
        out.append(c)
    for cat in arxiv_categories or []:
        for c in await fetch_arxiv_rss(cat, limit=limit):
            if c.arxiv_id in seen:
                continue
            seen.add(c.arxiv_id)
            out.append(c)
    return out
