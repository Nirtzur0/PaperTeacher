"""Discover candidate papers from HF Daily, arXiv RSS, and Semantic Scholar.

ML-domain discovery sources. The arXiv RSS plumbing lives in
`domains/_arxiv_rss.py` since multiple packs (ML, econ, ...) reuse it;
HF Daily and Semantic Scholar are ML-specific and stay here.

Source mix:
  - HF Daily Papers   - community-upvoted, ranked. Best for what's getting
                       traction RIGHT NOW. Biased toward applied LLM work.
  - arXiv RSS         - one feed per category. The firehose. High noise.
  - Semantic Scholar  - recent CS papers ranked by influentialCitationCount.
                       Surfaces papers other papers actually built on; lags
                       publication by a few weeks but the signal is real.

Cross-source deduplication strips arXiv version suffixes (`2401.12345v3` →
`2401.12345`) so the same paper showing up on HF Daily, arXiv RSS, AND
Semantic Scholar collapses to one canonical candidate.
"""
from __future__ import annotations

import datetime as dt
import logging
import re

import httpx

from ... import paths
from .._arxiv_rss import extract_arxiv_id, fetch_arxiv_rss
from .._common import Candidate
from .._semantic_scholar import fetch_semantic_scholar as _s2

log = logging.getLogger(__name__)

# Re-export so existing callers / tests that import from this module keep working.
__all__ = [
    "fetch_hf_daily",
    "fetch_arxiv_rss",
    "fetch_semantic_scholar",
    "discover",
    "canonical_arxiv_id",
]


# arXiv ids carry an optional `v\d+` revision suffix. For dedup we collapse
# all versions of the same paper to the bare id.
_VERSION_RE = re.compile(r"v\d+$")


def canonical_arxiv_id(aid: str) -> str:
    """`2401.12345v3` → `2401.12345`. The arxiv landing page redirects bare
    ids to the latest version, so dropping the suffix is safe and gives us
    a stable cross-source dedup key."""
    return _VERSION_RE.sub("", (aid or "").strip())


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
        aid = paper.get("id") or extract_arxiv_id(paper.get("url", ""))
        if not aid:
            continue
        out.append(
            Candidate(
                arxiv_id=canonical_arxiv_id(aid),
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


# ML's Semantic Scholar query — broad enough to catch the field, narrow
# enough to stay relevant. The shared fetcher in `domains/_semantic_scholar.py`
# does the actual HTTP work; we just configure it. The pack's reader expects
# arXiv ids, so we ask S2 for items keyed on the `ArXiv` external-id field
# and silently skip journal-only papers (their absence is by design).
_S2_QUERY = "machine learning OR neural network OR language model"


async def fetch_semantic_scholar(
    window_days: int = 90,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
    today: dt.date | None = None,
) -> list[Candidate]:
    """Recent CS papers from Semantic Scholar, ranked by influential citations."""
    return await _s2(
        query=_S2_QUERY,
        fields_of_study="Computer Science",
        id_type="ArXiv",
        canonicalize=canonical_arxiv_id,
        window_days=window_days,
        limit=limit,
        today=today,
    )


async def discover(
    arxiv_categories: list[str] | None = None,
    date: dt.date | None = None,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """Combined discovery. Order matters for dedup: earlier sources WIN
    duplicates (their `source` tag and metadata survive). The order is:

        1. HF Daily        — already-ranked-by-humans, most relevant TODAY
        2. arXiv RSS       — the firehose, by category
        3. Semantic Scholar — citation-velocity, last 90 days

    Dedup is on the version-stripped arXiv id so a paper showing up across
    multiple sources collapses to one canonical entry.
    """
    seen_ids: set[str] = set()
    out: list[Candidate] = []

    def _add(c: Candidate) -> None:
        key = canonical_arxiv_id(c.arxiv_id)
        if key in seen_ids:
            return
        seen_ids.add(key)
        c.arxiv_id = key
        out.append(c)

    for c in await fetch_hf_daily(date=date, limit=limit):
        _add(c)
    for cat in arxiv_categories or []:
        for c in await fetch_arxiv_rss(cat, limit=limit):
            _add(c)
    for c in await fetch_semantic_scholar(limit=limit):
        _add(c)
    return out
