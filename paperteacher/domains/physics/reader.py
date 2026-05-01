"""Read full paper text for the physics pack.

Same fallback chain as the ml pack — arXiv HTML → HF papers → arXiv abs
— since physics papers on arXiv share the same rendering. The HF papers
fallback is less likely to hit for pure-physics arXiv ids than for ML
papers (HF doesn't curate physics), but the URL still resolves to the
arXiv abstract via HF's redirect when the page exists, and silently
returns 404 when it doesn't, so leaving it in the chain costs us nothing.

Carrying the reader in-pack rather than importing from the ml pack keeps
the dependency graph clean: each pack owns its full discovery + read
chain, so disabling ml doesn't take physics down with it. The
duplication here is small (~50 lines) and the readers can evolve
independently — for instance, we may want to add an INSPIRE-HEP fallback
here later for old hep-th preprints whose HTML rendering arXiv hasn't
backfilled.
"""
from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup

from ... import paths
from .._common import PaperText

log = logging.getLogger(__name__)

MIN_USEFUL_TEXT_LEN = 500


def _strip(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


def _title_from(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


def _truncate(p: PaperText, max_chars: int) -> PaperText:
    if len(p.text) > max_chars:
        p.text = p.text[:max_chars]
        p.truncated = True
    return p


async def _fetch(url: str, client: httpx.AsyncClient) -> str | None:
    try:
        r = await client.get(url, follow_redirects=True)
    except httpx.HTTPError as e:
        log.debug("fetch %s failed: %s", url, e)
        return None
    if r.status_code != 200:
        log.debug("fetch %s -> %d", url, r.status_code)
        return None
    return r.text


async def read_paper(
    arxiv_id: str,
    max_chars: int = paths.DEFAULT_MAX_PAPER_CHARS,
) -> PaperText:
    """Try arxiv HTML, then HF paper page, then arXiv abstract. Always
    returns; `source == "none"` indicates total failure.

    The arXiv HTML URL omits the version suffix — arxiv.org redirects to
    the latest revision, so this works for v1 / v2 / v3 papers alike.
    For older physics preprints whose HTML rendering arXiv hasn't
    backfilled, the abstract endpoint is the safe floor.
    """
    sources = [
        (f"https://arxiv.org/html/{arxiv_id}", "arxiv_html"),
        (f"https://huggingface.co/papers/{arxiv_id}", "hf_paper"),
        (f"https://arxiv.org/abs/{arxiv_id}", "arxiv_abs"),
    ]
    async with httpx.AsyncClient(
        timeout=45, headers={"User-Agent": paths.USER_AGENT}
    ) as client:
        for url, label in sources:
            html = await _fetch(url, client)
            if html is None:
                continue
            text = _strip(html)
            if len(text) < MIN_USEFUL_TEXT_LEN:
                log.debug("fetch %s extracted text too short (%d chars)", url, len(text))
                continue
            log.info("read_paper %s: %s (%d chars)", arxiv_id, label, len(text))
            return _truncate(PaperText(arxiv_id, _title_from(html), text, label), max_chars)

    log.warning("read_paper %s: all sources failed", arxiv_id)
    return PaperText(arxiv_id, "", "", "none")
