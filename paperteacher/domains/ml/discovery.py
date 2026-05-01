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


# Semantic Scholar's `paper/search/bulk` endpoint returns up to 1000 papers
# matching a query, with rich metadata. We use it to surface recent CS
# papers that have ALREADY been cited — the citation-velocity signal that
# neither HF Daily nor arXiv RSS provides. Free tier: ~100 req/5min
# unauthenticated; we make one call per discovery, so well within budget.
_S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
_S2_FIELDS = ",".join([
    "externalIds",
    "title",
    "authors",
    "abstract",
    "citationCount",
    "influentialCitationCount",
    "publicationDate",
])
# Broad query string. S2's bulk endpoint requires a query — empty isn't
# allowed — so we use a permissive "AND" filter that still narrows to ML.
_S2_QUERY = "machine learning OR neural network OR language model"


async def fetch_semantic_scholar(
    window_days: int = 90,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
    today: dt.date | None = None,
) -> list[Candidate]:
    """Recent CS papers from Semantic Scholar, ranked by influential citations.

    Falls through silently (returns []) on network / auth / quota errors —
    discovery is additive, not load-bearing on this source.
    """
    today = today or dt.date.today()
    start = today - dt.timedelta(days=window_days)
    params = {
        "query": _S2_QUERY,
        "fields": _S2_FIELDS,
        "publicationDateOrYear": f"{start.isoformat()}:{today.isoformat()}",
        "fieldsOfStudy": "Computer Science",
    }
    out: list[Candidate] = []
    try:
        async with httpx.AsyncClient(
            timeout=30, headers={"User-Agent": paths.USER_AGENT}
        ) as client:
            r = await client.get(_S2_BULK_URL, params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        log.warning("semantic_scholar fetch failed: %s", e)
        return out
    except ValueError as e:
        log.warning("semantic_scholar JSON decode failed: %s", e)
        return out

    items = data.get("data") or []
    for item in items:
        ext = item.get("externalIds") or {}
        # Need an arxiv id to route the paper through the rest of the
        # pipeline (reader, outline, audio). S2 papers without an ArXiv
        # id (e.g. journal-only) are silently skipped.
        aid_raw = ext.get("ArXiv")
        if not aid_raw:
            continue
        aid = canonical_arxiv_id(aid_raw)
        # Score = influential citations primarily, with raw citation count
        # as a tiebreaker. Keeps the ordering meaningful even when the
        # influential count is sparse for very recent papers.
        infl = float(item.get("influentialCitationCount") or 0)
        cites = float(item.get("citationCount") or 0)
        score = infl * 10 + cites
        out.append(
            Candidate(
                arxiv_id=aid,
                title=(item.get("title") or "").strip(),
                authors=[a.get("name", "") for a in (item.get("authors") or [])],
                summary=(item.get("abstract") or "").strip(),
                source="semantic_scholar",
                score=score,
                url=f"https://arxiv.org/abs/{aid}",
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    out = out[:limit]
    log.info("semantic_scholar (last %dd): %d candidates", window_days, len(out))
    return out


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
