"""Read full paper text for the physics pack.

Same fallback chain as the ml pack — arXiv HTML → HF papers → arXiv abs
— since physics papers on arXiv share the same rendering. The HF papers
fallback is less likely to hit for pure-physics arXiv ids than for ML
papers (HF doesn't curate physics), but the URL still resolves to the
arXiv abstract via HF's redirect when the page exists, and silently
returns 404 when it doesn't, so leaving it in the chain costs us nothing.

The fetch/strip/validate loop lives in `domains/_html.walk_sources`; this
file just owns the source list. We may add an INSPIRE-HEP fallback here
later for old hep-th preprints whose HTML rendering arXiv hasn't backfilled.
"""
from __future__ import annotations

from ... import paths
from .. import _html
from .._common import PaperText


async def read_paper(
    arxiv_id: str,
    max_chars: int = paths.DEFAULT_MAX_PAPER_CHARS,
) -> PaperText:
    """Try arxiv HTML, then HF paper page, then arXiv abstract. Always
    returns; `source == "none"` indicates total failure.

    The arXiv HTML URL omits the version suffix — arxiv.org redirects to
    the latest revision, so this works for v1 / v2 / v3 papers alike.
    """
    sources = [
        (f"https://arxiv.org/html/{arxiv_id}", "arxiv_html"),
        (f"https://huggingface.co/papers/{arxiv_id}", "hf_paper"),
        (f"https://arxiv.org/abs/{arxiv_id}", "arxiv_abs"),
    ]
    return await _html.walk_sources(arxiv_id, sources, max_chars=max_chars)
