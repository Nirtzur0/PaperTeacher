"""Shared HTML-fetching helpers for domain pack readers.

Lifted out of `ml/reader.py` once the second pack (econ) needed the same
plumbing: fetch a URL, strip nav/script/style/etc., extract a title, and
honour a max-chars budget. Behaviour is byte-for-byte what the ML reader
used before extraction.
"""
from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup

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
