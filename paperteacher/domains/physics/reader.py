"""Read full paper text for the physics pack.

Same fallback chain as the ml pack — arXiv HTML → HF papers → arXiv abs
— since physics papers on arXiv share the same rendering. The HF papers
fallback is less likely to hit for pure-physics arXiv ids than for ML
papers (HF doesn't curate physics), but the URL still resolves to the
arXiv abstract via HF's redirect when the page exists, and silently
returns 404 when it doesn't, so leaving it in the chain costs us nothing.

HTML plumbing (fetch, strip, title, truncate) lives in `domains/_html.py`
so each pack's reader stays focused on the source list. We may add an
INSPIRE-HEP fallback here later for old hep-th preprints whose HTML
rendering arXiv hasn't backfilled.
"""
from __future__ import annotations

import logging

import httpx

from ... import paths
from .. import _html
from .._common import PaperText

log = logging.getLogger(__name__)


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
            html = await _html.fetch(url, client)
            if html is None:
                continue
            text = _html.strip(html)
            if len(text) < _html.MIN_USEFUL_TEXT_LEN:
                log.debug("fetch %s extracted text too short (%d chars)", url, len(text))
                continue
            log.info("read_paper %s: %s (%d chars)", arxiv_id, label, len(text))
            return _html.truncate(
                PaperText(arxiv_id, _html.title_from(html), text, label), max_chars
            )

    log.warning("read_paper %s: all sources failed", arxiv_id)
    return PaperText(arxiv_id, "", "", "none")
