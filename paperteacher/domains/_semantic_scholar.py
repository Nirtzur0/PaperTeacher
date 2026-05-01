"""Cross-domain Semantic Scholar discovery.

S2's bulk endpoint is the citation-velocity signal that no other source we
use provides — it tells you which papers other papers ALREADY built on.
The ranking and request mechanics are universal; what differs per domain is

  - the search query string (`query`)
  - the field-of-study filter (`fields_of_study`)
  - which external-id we extract as the canonical paper id (`id_type`)
  - how we build a URL from that id (`url_template`)
  - any per-id normalization (`canonicalize`) so cross-source dedup hits

Each pack imports `fetch_semantic_scholar` and configures these kwargs.

Why parameterize id-type
------------------------
S2's `externalIds` mapping carries `ArXiv` (CS / physics / math), `DOI`
(biology / medicine / economics journal papers), `PubMed`, etc. A pack's
reader knows how to resolve exactly one of these. The ml pack expects
`ArXiv`; the neuro pack expects `DOI`; etc. We extract whatever the pack
asks for and silently skip items where that key is missing — which is how
domain isolation stays clean even though the upstream is shared.

Why not in `_common.py`
-----------------------
`_common.py` holds *types* (Candidate, PaperText, AuditReport). This file
holds a *fetcher*. Same split as `_arxiv_rss.py` and `_html.py` — utilities
shared by domain packs, not themselves a pack.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Callable, Literal

import httpx

from .. import paths
from ._common import Candidate

log = logging.getLogger(__name__)

# What we ask S2 to return. Static across packs; cheap to over-request and
# only use what we need per item.
DEFAULT_FIELDS = ",".join([
    "externalIds",
    "title",
    "authors",
    "abstract",
    "citationCount",
    "influentialCitationCount",
    "publicationDate",
])

_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

IdType = Literal["ArXiv", "DOI", "PubMed", "PubMedCentral"]

# When a pack doesn't specify a URL template, fall back to a sensible
# scheme based on id_type. Each entry is a `str.format`-style template
# whose single placeholder receives the canonicalized id.
_DEFAULT_URLS: dict[IdType, str] = {
    "ArXiv":         "https://arxiv.org/abs/{id}",
    "DOI":           "https://doi.org/{id}",
    "PubMed":        "https://pubmed.ncbi.nlm.nih.gov/{id}/",
    "PubMedCentral": "https://www.ncbi.nlm.nih.gov/pmc/articles/{id}/",
}


def _default_score(item: dict) -> float:
    """Influential-citations weighted heavily, raw citations as tiebreaker.

    Keeps ordering meaningful even when influential count is sparse for very
    recent papers. Override per pack via `score_fn=` if a domain has a better
    signal (e.g. journal-tier weights for econ).
    """
    infl = float(item.get("influentialCitationCount") or 0)
    cites = float(item.get("citationCount") or 0)
    return infl * 10 + cites


async def fetch_semantic_scholar(
    *,
    query: str,
    fields_of_study: str,
    id_type: IdType = "ArXiv",
    window_days: int = 90,
    limit: int = 20,
    today: dt.date | None = None,
    url_template: str | None = None,
    canonicalize: Callable[[str], str] | None = None,
    score_fn: Callable[[dict], float] | None = None,
    source_tag: str = "semantic_scholar",
) -> list[Candidate]:
    """Hit S2's bulk endpoint, filter for the requested id_type, rank, return.

    Args:
        query: S2 search-query string. Required — bulk endpoint rejects empty.
              Use OR'd terms broad enough to capture the field but narrow
              enough to stay relevant.
        fields_of_study: S2 fieldsOfStudy filter. Comma-separated, case-
              sensitive. Examples: "Computer Science", "Physics",
              "Biology,Medicine", "Economics". Required.
        id_type: Which external-id the pack's reader can resolve. Items
              without this id are silently dropped — that's how the per-pack
              filtering stays implicit.
        window_days: Look back this far for "recent" papers. 90 is a
              reasonable default; the picker can re-rank by recency anyway.
        limit: Max candidates to return after ranking.
        today: Override the "now" date (used in tests).
        url_template: Override the URL we attach to each Candidate. Falls
              back to a sensible default per id_type.
        canonicalize: Per-id normalization — for arxiv this strips version
              suffixes; for DOI this lowercases. Defaults to identity.
        score_fn: Per-pack ranking. Defaults to `infl*10 + cites`.
        source_tag: Override the `source` field on each Candidate. Default
              `"semantic_scholar"` — most packs want this for log clarity.
              A pack with multiple S2 calls could differentiate them.

    Returns empty list on network/auth/quota errors — discovery is additive,
    never load-bearing on this source.
    """
    today = today or dt.date.today()
    start = today - dt.timedelta(days=window_days)
    canonicalize = canonicalize or (lambda s: s)
    score_fn = score_fn or _default_score
    url_template = url_template or _DEFAULT_URLS.get(id_type) or "{id}"

    params = {
        "query": query,
        "fields": DEFAULT_FIELDS,
        "publicationDateOrYear": f"{start.isoformat()}:{today.isoformat()}",
        "fieldsOfStudy": fields_of_study,
    }

    out: list[Candidate] = []
    try:
        async with httpx.AsyncClient(
            timeout=30, headers={"User-Agent": paths.USER_AGENT}
        ) as client:
            r = await client.get(_BULK_URL, params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        log.warning("semantic_scholar fetch failed (%s): %s", source_tag, e)
        return out
    except ValueError as e:
        log.warning("semantic_scholar JSON decode failed (%s): %s", source_tag, e)
        return out

    items = data.get("data") or []
    for item in items:
        ext = item.get("externalIds") or {}
        raw_id = ext.get(id_type)
        if not raw_id:
            continue
        # `canonicalize` produces the *framework-side* id (filesystem-safe,
        # dedup key). The URL is built from the *raw* upstream id so it stays
        # fetchable — for neuro, canonicalize encodes `/` to `_` for storage,
        # but the doi.org URL needs the slash. ML's raw and canonical happen
        # to coincide for URL purposes (arxiv redirects bare→latest).
        cid = canonicalize(str(raw_id))
        if not cid:
            continue
        out.append(
            Candidate(
                arxiv_id=cid,  # opaque id — semantics depends on the pack
                title=(item.get("title") or "").strip(),
                authors=[a.get("name", "") for a in (item.get("authors") or [])],
                summary=(item.get("abstract") or "").strip(),
                source=source_tag,
                score=score_fn(item),
                url=url_template.format(id=str(raw_id)),
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    out = out[:limit]
    log.info(
        "semantic_scholar (%s, last %dd, fos=%s, id=%s): %d candidates",
        source_tag, window_days, fields_of_study, id_type, len(out),
    )
    return out


__all__ = [
    "fetch_semantic_scholar",
    "DEFAULT_FIELDS",
    "IdType",
]
