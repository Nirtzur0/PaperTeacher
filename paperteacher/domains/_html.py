"""Shared HTML-fetching helpers for domain pack readers.

Lifted out of `ml/reader.py` once the second pack (econ) needed the same
plumbing: fetch a URL, strip nav/script/style/etc., extract a title, and
honour a max-chars budget. The `walk_sources` helper hosts the
fetch-strip-validate-truncate loop every URL-based reader does the same
way; pack-specific bits (alternate strippers, post-fetch sanity checks)
live in optional per-source callables.
"""
from __future__ import annotations

import logging
from typing import Callable

import httpx
from bs4 import BeautifulSoup

from .. import paths
from ._common import PaperText

log = logging.getLogger(__name__)

MIN_USEFUL_TEXT_LEN = 500


def strip(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


def title_from(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


def truncate(p: PaperText, max_chars: int) -> PaperText:
    if len(p.text) > max_chars:
        p.text = p.text[:max_chars]
        p.truncated = True
    return p


async def fetch(url: str, client: httpx.AsyncClient) -> str | None:
    try:
        r = await client.get(url, follow_redirects=True)
    except httpx.HTTPError as e:
        log.debug("fetch %s failed: %s", url, e)
        return None
    if r.status_code != 200:
        log.debug("fetch %s -> %d", url, r.status_code)
        return None
    return r.text


# A source entry: (url, label, strip_fn, post_check). `strip_fn` defaults to
# the generic `strip()` when omitted; `post_check` returns True to keep the
# extracted text or False to fall through (used by ar5iv to skip pages that
# look like LaTeXML error chrome).
StripFn = Callable[[str], str]
PostCheck = Callable[[str], bool]


async def walk_sources(
    arxiv_id: str,
    sources: list[tuple],
    *,
    max_chars: int,
    client: httpx.AsyncClient | None = None,
) -> PaperText:
    """Walk a fallback chain of URL sources and return the first one that
    yields enough text. `sources` is a list of tuples in one of these forms:

      (url, label)
      (url, label, strip_fn)
      (url, label, strip_fn, post_check)

    Each candidate is fetched, stripped, gated on `MIN_USEFUL_TEXT_LEN`, and
    optionally vetted by `post_check`. The first match wins; otherwise we
    return `PaperText(source="none")`.

    Pass an existing `client` to share connections across sources; otherwise
    we open one with the framework defaults.
    """
    own_client = client is None

    async def _walk(c: httpx.AsyncClient) -> PaperText:
        for entry in sources:
            url, label = entry[0], entry[1]
            strip_fn: StripFn = entry[2] if len(entry) > 2 else strip
            post_check: PostCheck | None = entry[3] if len(entry) > 3 else None

            html = await fetch(url, c)
            if html is None:
                continue
            text = strip_fn(html)
            if len(text) < MIN_USEFUL_TEXT_LEN:
                log.debug("fetch %s extracted text too short (%d chars)", url, len(text))
                continue
            if post_check is not None and not post_check(text):
                log.debug("fetch %s: post_check rejected — falling through", url)
                continue
            log.info("read_paper %s: %s (%d chars)", arxiv_id, label, len(text))
            return truncate(
                PaperText(arxiv_id, title_from(html), text, label), max_chars
            )

        log.warning("read_paper %s: all sources failed", arxiv_id)
        return PaperText(arxiv_id, "", "", "none")

    if own_client:
        async with httpx.AsyncClient(
            timeout=45, headers={"User-Agent": paths.USER_AGENT}
        ) as new_client:
            return await _walk(new_client)
    return await _walk(client)
