"""Read full paper text. Dispatches on id format.

ID conventions in this pack:
  - arXiv econ/q-fin papers: `2603.20105`, `2603.20105v2`, etc. The dotted
    pattern is disjoint from NBER's, so we route by regex.
  - NBER working papers: `w31234`. The single-letter prefix is part of the
    canonical NBER id and is safe to use as a filesystem-friendly key.

For arXiv ids we walk arxiv.org/html → huggingface.co/papers → arxiv.org/abs.
The ML pack also tries ar5iv first; we don't here because econ papers have
much less LaTeX density than ML/math papers and the ar5iv layer adds latency
without much payoff.

For NBER ids we fetch the abstract landing page. The full PDF is openly
available at `/papers/{id}.pdf` but parsing PDFs would add a heavy
dependency and the abstract page already carries title, authors, full
abstract, and (for many recent papers) a non-technical summary — enough
signal for stage 1 outline extraction.

The fetch/strip/validate loop lives in `domains/_html.walk_sources`; this
file just dispatches by id and supplies the source list.
"""
from __future__ import annotations

import logging
import re

from ... import paths
from .. import _html
from .._common import PaperText

log = logging.getLogger(__name__)

ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
NBER_ID_RE = re.compile(r"^w\d{4,5}$")


def _arxiv_sources(arxiv_id: str) -> list[tuple[str, str]]:
    return [
        (f"https://arxiv.org/html/{arxiv_id}", "arxiv_html"),
        (f"https://huggingface.co/papers/{arxiv_id}", "hf_paper"),
        (f"https://arxiv.org/abs/{arxiv_id}", "arxiv_abs"),
    ]


def _nber_sources(nber_id: str) -> list[tuple[str, str]]:
    return [
        (f"https://www.nber.org/papers/{nber_id}", "nber_abstract"),
    ]


async def read_paper(
    arxiv_id: str,
    max_chars: int = paths.DEFAULT_MAX_PAPER_CHARS,
) -> PaperText:
    """Dispatch by id format and walk the matching source chain. Returns
    `PaperText(source="none")` on total failure — same contract as the ML
    pack.
    """
    if ARXIV_ID_RE.match(arxiv_id):
        sources = _arxiv_sources(arxiv_id)
    elif NBER_ID_RE.match(arxiv_id):
        sources = _nber_sources(arxiv_id)
    else:
        log.warning("read_paper %s: unrecognized id format", arxiv_id)
        return PaperText(arxiv_id, "", "", "none")

    return await _html.walk_sources(arxiv_id, sources, max_chars=max_chars)
