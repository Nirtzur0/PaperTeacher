"""Read full paper text using the fallback chain: arXiv HTML -> HF papers -> arXiv abstract."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from . import paths

log = logging.getLogger(__name__)

USER_AGENT = "PaperTeacher/0.1"
MIN_USEFUL_LEN = 500  # below this, treat as "not the paper"


@dataclass
class PaperText:
    arxiv_id: str
    title: str
    text: str
    source: str  # arxiv_html | hf_paper | arxiv_abs | none
    truncated: bool = False


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


async def _try(url: str, client: httpx.AsyncClient) -> str | None:
    try:
        r = await client.get(url, follow_redirects=True)
    except httpx.HTTPError as e:
        log.debug("fetch %s failed: %s", url, e)
        return None
    if r.status_code != 200:
        log.debug("fetch %s -> %d", url, r.status_code)
        return None
    if len(r.text) < MIN_USEFUL_LEN:
        log.debug("fetch %s too short (%d)", url, len(r.text))
        return None
    return r.text


async def read_paper(
    arxiv_id: str,
    max_chars: int = paths.DEFAULT_MAX_PAPER_CHARS,
) -> PaperText:
    """Try arxiv HTML, then HF paper page, then arXiv abstract. Always returns;
    `source == "none"` indicates total failure."""
    sources = [
        (f"https://arxiv.org/html/{arxiv_id}v1", "arxiv_html"),
        (f"https://huggingface.co/papers/{arxiv_id}", "hf_paper"),
        (f"https://arxiv.org/abs/{arxiv_id}", "arxiv_abs"),
    ]
    async with httpx.AsyncClient(timeout=45, headers={"User-Agent": USER_AGENT}) as client:
        for url, label in sources:
            html = await _try(url, client)
            if html is None:
                continue
            text = _strip(html)
            log.info("read_paper %s: %s (%d chars)", arxiv_id, label, len(text))
            return _truncate(PaperText(arxiv_id, _title_from(html), text, label), max_chars)

    log.warning("read_paper %s: all sources failed", arxiv_id)
    return PaperText(arxiv_id, "", "", "none")
