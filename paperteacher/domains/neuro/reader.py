"""Read full text for a neuroscience paper.

Fallback chain (first non-empty text wins):
  1. Europe PMC ``fullTextXML`` for the published version, when the
     paper has been accepted to a PMC OA journal. Best signal — JATS
     XML with figure captions and methods, which is what stage 1 needs.
  2. bioRxiv ``.full`` HTML at ``biorxiv.org/content/<DOI>v<N>.full``.
     Fragile: bioRxiv puts everything behind Cloudflare and a non-residential
     IP usually gets a 403 challenge. We try anyway because residential
     IPs (and some cloud egress allowlists) succeed.
  3. bioRxiv ``details`` JSON, which always returns the abstract +
     metadata. Lossy but never fails — gives stage 1 enough to extract
     the thesis and at least the most prominent finding from the
     abstract, with the outline `note` field flagging which findings
     couldn't be ground-checked against the methods.

The pack always returns a ``PaperText``; ``source == "none"`` indicates
total failure.
"""
from __future__ import annotations

import logging
import re

import httpx

from ... import paths
from .. import _html
from .._common import PaperText

log = logging.getLogger(__name__)

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest"
BIORXIV_DETAILS = "https://api.biorxiv.org/details/biorxiv"


def _decode_id(arxiv_id: str) -> str:
    """Inverse of discovery._encode_doi. ``10.1101_2024.05.01.591742`` ->
    ``10.1101/2024.05.01.591742``. Splits on the FIRST underscore only;
    DOIs may legitimately contain further underscores in their suffix."""
    return arxiv_id.replace("_", "/", 1)


def _title_from_xml(xml: str) -> str:
    """JATS: <article-title> nested inside <title-group> in <front>."""
    m = re.search(
        r"<article-title[^>]*>(.*?)</article-title>", xml, re.DOTALL | re.IGNORECASE
    )
    if not m:
        return ""
    # JATS allows inline markup inside <article-title>; strip it.
    return re.sub(r"<[^>]+>", "", m.group(1)).strip()


async def _try_epmc_pmc_fulltext(
    client: httpx.AsyncClient, doi: str
) -> tuple[str, str] | None:
    """Look up the DOI in EPMC; if it has a PMC OA full-text version
    (preprint accepted somewhere and indexed), fetch the JATS XML.

    Returns ``(text, title)`` or None. The XML carries the methods + figure
    captions, which is the highest-quality input the pipeline ever sees
    for a neuro paper.
    """
    params = {
        "query": f'DOI:"{doi}" AND SRC:PMC',
        "format": "json",
        "resultType": "lite",
        "pageSize": "1",
    }
    try:
        r = await client.get(EPMC_SEARCH, params=params)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.debug("epmc lookup failed for %s: %s", doi, e)
        return None
    results = (data.get("resultList") or {}).get("result", [])
    if not results:
        return None
    pmcid = results[0].get("pmcid") or results[0].get("id")
    if not pmcid:
        return None
    xml = await _html.fetch(f"{EPMC_FULLTEXT}/{pmcid}/fullTextXML", client)
    if not xml:
        return None
    text = _html.strip(xml)
    if len(text) < _html.MIN_USEFUL_TEXT_LEN:
        return None
    return text, _title_from_xml(xml)


async def _try_biorxiv_html(
    client: httpx.AsyncClient, doi: str
) -> tuple[str, str] | None:
    """Try the bioRxiv full HTML. Often blocked by Cloudflare from
    non-residential IPs — we fail soft so the next fallback can run."""
    # Don't know the version a priori; v1 is most common for fresh
    # candidates. If a paper is at v3 we'll usually still hit a redirect
    # to the latest, but the URL path requires *some* version.
    for vn in (1, 2, 3):
        html = await _html.fetch(f"https://www.biorxiv.org/content/{doi}v{vn}.full", client)
        if not html:
            continue
        text = _html.strip(html)
        if len(text) >= _html.MIN_USEFUL_TEXT_LEN:
            return text, _html.title_from(html)
    # Fallback: the canonical landing page (without explicit version).
    html = await _html.fetch(f"https://www.biorxiv.org/content/{doi}", client)
    if html:
        text = _html.strip(html)
        if len(text) >= _html.MIN_USEFUL_TEXT_LEN:
            return text, _html.title_from(html)
    return None


async def _try_biorxiv_abstract(
    client: httpx.AsyncClient, doi: str
) -> tuple[str, str] | None:
    """Last-resort: pull the abstract from the bioRxiv details JSON. Always
    works for a valid bioRxiv DOI (the API isn't Cloudflare-gated)."""
    try:
        r = await client.get(f"{BIORXIV_DETAILS}/{doi}/na/json")
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.debug("biorxiv details failed for %s: %s", doi, e)
        return None
    coll = data.get("collection") or []
    if not coll:
        return None
    # Pick the latest version (highest 'version' field) since bioRxiv
    # returns one row per revision.
    latest = max(coll, key=lambda r: int(r.get("version", "0") or 0))
    abstract = (latest.get("abstract") or "").strip()
    title = (latest.get("title") or "").strip()
    # Don't gate on MIN_USEFUL_TEXT_LEN here: some bioRxiv records carry
    # short structured abstracts, and any abstract beats `source=none`.
    if not abstract:
        return None
    return abstract, title


async def _try_epmc_abstract(
    client: httpx.AsyncClient, doi: str
) -> tuple[str, str] | None:
    """For non-bioRxiv sources (journal-published from EPMC discovery), pull
    the abstract from EPMC search."""
    params = {
        "query": f'DOI:"{doi}"',
        "format": "json",
        "resultType": "core",
        "pageSize": "1",
    }
    try:
        r = await client.get(EPMC_SEARCH, params=params)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.debug("epmc abstract failed for %s: %s", doi, e)
        return None
    results = (data.get("resultList") or {}).get("result", [])
    if not results:
        return None
    rec = results[0]
    abstract = (rec.get("abstractText") or "").strip()
    title = (rec.get("title") or "").strip()
    if not abstract:
        return None
    return abstract, title


async def read_paper(
    arxiv_id: str,
    max_chars: int = paths.DEFAULT_MAX_PAPER_CHARS,
) -> PaperText:
    """Try EPMC PMC full text -> bioRxiv full HTML -> EPMC abstract -> bioRxiv
    abstract. Always returns; ``source == "none"`` indicates total failure.

    The caller treats `arxiv_id` as opaque — discovery emits encoded DOIs
    (slashes -> underscores) and this reader is the boundary that maps
    back to live URLs.
    """
    doi = _decode_id(arxiv_id)
    async with httpx.AsyncClient(
        timeout=45, headers={"User-Agent": paths.USER_AGENT}
    ) as client:
        # Order matters: EPMC PMC full text > bioRxiv full HTML > abstracts.
        # PMC has the methods section we actually need; bioRxiv HTML
        # sometimes has it; abstracts never do.
        for label, fetcher in [
            ("epmc_pmc_fulltext", _try_epmc_pmc_fulltext),
            ("biorxiv_html", _try_biorxiv_html),
            ("epmc_abstract", _try_epmc_abstract),
            ("biorxiv_abstract", _try_biorxiv_abstract),
        ]:
            result = await fetcher(client, doi)
            if result is None:
                continue
            text, title = result
            log.info("read_paper %s: %s (%d chars)", arxiv_id, label, len(text))
            return _html.truncate(PaperText(arxiv_id, title, text, label), max_chars)

    log.warning("read_paper %s: all sources failed", arxiv_id)
    return PaperText(arxiv_id, "", "", "none")
