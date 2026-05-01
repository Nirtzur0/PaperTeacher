"""Discover neuroscience candidate papers from bioRxiv and Europe PMC.

bioRxiv's `details` JSON API is the workhorse — it lets us filter by
`category=neuroscience` over a date interval and returns the abstract,
authors, DOI, and (importantly) a link to the JATS source XML for use by
the reader. We supplement with a Europe PMC search keyed on a generic
"is this neuroscience?" query so newly published journal papers (e.g.
eLife, J. Neurosci, Nature Neuroscience) that didn't transit through
bioRxiv still surface.

Identifiers
-----------
bioRxiv DOIs look like ``10.1101/2024.05.01.591742`` (or, more recently,
``10.64898/2026.04.23.720488`` — the prefix isn't fixed). Storage uses
``arxiv_id`` as a filename suffix, and ``/`` is illegal in filenames, so we
encode DOIs as ``<prefix>_<suffix>`` for the framework. The reader's
helper undoes this when calling out to bioRxiv / Europe PMC. The
``Candidate.arxiv_id`` field is opaque at the framework level — see the
docstring in ``domains/_common.py``.
"""
from __future__ import annotations

import datetime as dt
import logging

import httpx

from ... import paths
from .._common import Candidate

log = logging.getLogger(__name__)

BIORXIV_API = "https://api.biorxiv.org/details/biorxiv"
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# bioRxiv pages this many records per request; cursor steps by 30 (NOT 100
# despite older docs). Confirmed empirically against the live API.
BIORXIV_PAGE_SIZE = 30

# How far back to look on a single discover call when no explicit window is
# given. Three days is a sweet spot: bioRxiv posts ~30-50 neuro papers/day,
# so three days yields ~100-150 candidates — enough to filter against the
# user's preferred-authors list without hammering the API.
DEFAULT_LOOKBACK_DAYS = 3


def _split_authors(s: str) -> list[str]:
    """bioRxiv returns authors as one semicolon-joined string; Europe PMC
    returns them on `authorString` in a similar trailing-period style. Both
    tolerate the same parser."""
    if not s:
        return []
    return [a.strip().rstrip(".") for a in s.split(";") if a.strip()]


def _encode_doi(doi: str) -> str:
    """DOI -> filesystem-safe id. ``10.1101/2024.05.01.591742`` becomes
    ``10.1101_2024.05.01.591742``. Single underscore separator so the
    reader can split exactly once."""
    return doi.replace("/", "_", 1)


async def fetch_biorxiv_neuro(
    *,
    days: int = DEFAULT_LOOKBACK_DAYS,
    today: dt.date | None = None,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """Pull recent bioRxiv neuroscience preprints. Paginates the cursor until
    `limit` is hit or the window is exhausted. ``today`` is injectable for
    deterministic tests.
    """
    today = today or dt.date.today()
    start = (today - dt.timedelta(days=days)).isoformat()
    end = today.isoformat()
    out: list[Candidate] = []
    cursor = 0
    async with httpx.AsyncClient(
        timeout=30, headers={"User-Agent": paths.USER_AGENT}
    ) as client:
        while len(out) < limit:
            url = (
                f"{BIORXIV_API}/{start}/{end}/{cursor}/json"
                "?category=neuroscience"
            )
            try:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPError as e:
                log.warning("biorxiv fetch failed (%s..%s cursor=%d): %s",
                            start, end, cursor, e)
                break
            except ValueError as e:  # JSON decode
                log.warning("biorxiv JSON decode failed: %s", e)
                break
            batch = data.get("collection") or []
            if not batch:
                break
            # bioRxiv may return multiple versions of the same DOI; we want
            # the latest revision per id. Iterate, dedupe by encoded id.
            for item in batch:
                doi = (item.get("doi") or "").strip()
                if not doi:
                    continue
                aid = _encode_doi(doi)
                # Skip if we've already emitted this id (older version).
                if any(c.arxiv_id == aid for c in out):
                    continue
                out.append(
                    Candidate(
                        arxiv_id=aid,
                        title=(item.get("title") or "").strip(),
                        authors=_split_authors(item.get("authors", "")),
                        summary=(item.get("abstract") or "").strip(),
                        source="biorxiv_neuro",
                        # Recency score: today = 1.0, day before = 0.9, ...
                        # Discovery layer uses score for ranking; this keeps
                        # newer papers near the top without crowding out
                        # human-curated sources like preferred-author hits.
                        score=_recency_score(item.get("date", ""), today),
                        url=f"https://www.biorxiv.org/content/{doi}",
                    )
                )
                if len(out) >= limit:
                    break
            cursor += BIORXIV_PAGE_SIZE
            # Defensive cap — windowed bioRxiv responses are bounded but the
            # cursor convention is loose. Stop if we've paged past the
            # advertised total.
            try:
                total = int((data.get("messages") or [{}])[0].get("total", "0"))
            except (TypeError, ValueError):
                total = 0
            if total and cursor >= total:
                break
    log.info("biorxiv_neuro %s..%s: %d candidates", start, end, len(out))
    return out


def _recency_score(date_str: str, today: dt.date) -> float:
    """Map bioRxiv `date` -> [0, 1]. Same-day = 1.0, then ~0.1 decay per day.
    Bounded below at 0 so very old re-uploaded versions can't go negative."""
    try:
        d = dt.date.fromisoformat(date_str)
    except (TypeError, ValueError):
        return 0.0
    age = max(0, (today - d).days)
    return max(0.0, 1.0 - 0.1 * age)


async def fetch_epmc_neuro(
    *,
    days: int = DEFAULT_LOOKBACK_DAYS,
    today: dt.date | None = None,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """Catch journal papers (eLife, J. Neurosci, Nat. Neurosci, etc.) that
    didn't go through bioRxiv. Europe PMC's MeSH/abstract index covers all
    PubMed-indexed neuroscience plus the open preprint subset.

    Query: anything with "neuroscience" or a near-synonym that has a
    machine-fetchable abstract, sorted by date. We deliberately do NOT
    filter by SRC:PPR here — that's bioRxiv's job above; this fetcher
    fills in journal coverage.
    """
    today = today or dt.date.today()
    start = (today - dt.timedelta(days=days)).isoformat()
    end = today.isoformat()
    # MESH:Neuroscience matches papers tagged in MEDLINE; ABSTRACT:Y filters
    # out abstract-less stub records that would just bloat the candidate
    # list. Date filter pins the window without needing ORDER:DATE_DESC.
    query = (
        f'(MESH:"Neuroscience" OR MESH:"Brain") '
        f"AND ABSTRACT:Y "
        f"AND FIRST_PDATE:[{start} TO {end}] "
        f"AND NOT SRC:PPR"
    )
    params = {
        "query": query,
        "format": "json",
        "resultType": "lite",
        "pageSize": str(min(limit, 25)),
    }
    out: list[Candidate] = []
    try:
        async with httpx.AsyncClient(
            timeout=30, headers={"User-Agent": paths.USER_AGENT}
        ) as client:
            r = await client.get(EPMC_SEARCH, params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        log.warning("epmc fetch failed (%s..%s): %s", start, end, e)
        return out
    except ValueError as e:
        log.warning("epmc JSON decode failed: %s", e)
        return out
    for r in (data.get("resultList") or {}).get("result", [])[:limit]:
        doi = (r.get("doi") or "").strip()
        # Prefer DOI as the durable id; if absent, fall back to EPMC's PPR/PMC id.
        if doi:
            aid = _encode_doi(doi)
        else:
            src = r.get("source", "")
            xid = r.get("id", "")
            if not (src and xid):
                continue
            aid = f"{src}_{xid}"
        out.append(
            Candidate(
                arxiv_id=aid,
                title=(r.get("title") or "").strip(),
                authors=_split_authors(r.get("authorString", "")),
                summary=(r.get("abstractText") or "").strip(),
                source="epmc_neuro",
                score=_recency_score(r.get("firstPublicationDate", ""), today),
                url=f"https://europepmc.org/article/{r.get('source','MED')}/{r.get('id','')}",
            )
        )
    log.info("epmc_neuro %s..%s: %d candidates", start, end, len(out))
    return out


async def discover(
    arxiv_categories: list[str] | None = None,    # accepted, ignored — interface compat
    date: dt.date | None = None,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[Candidate]:
    """Combined discovery: bioRxiv first (preprints, full coverage of the
    field), then Europe PMC for journal-published work.

    The signature mirrors the ML pack so `discover_all()` can call it with
    the same kwargs; ``arxiv_categories`` is silently ignored because
    neuroscience doesn't live on arXiv.
    """
    seen_ids: set[str] = set()
    out: list[Candidate] = []
    for c in await fetch_biorxiv_neuro(today=date, limit=limit):
        if c.arxiv_id in seen_ids:
            continue
        seen_ids.add(c.arxiv_id)
        out.append(c)
    for c in await fetch_epmc_neuro(today=date, limit=limit):
        if c.arxiv_id in seen_ids:
            continue
        seen_ids.add(c.arxiv_id)
        out.append(c)
    return out
