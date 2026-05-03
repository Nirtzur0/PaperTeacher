"""Read full paper text for the math pack.

Same fallback chain as the physics pack — arXiv HTML → HF papers → arXiv
abs — since math papers on arXiv share the same rendering. The HF papers
fallback rarely hits for math (HF doesn't curate math) but the URL still
resolves to the arXiv abstract via HF's redirect when the page exists, so
leaving it in the chain costs nothing.
"""
from __future__ import annotations

from ... import paths
from .. import _html
from .._common import PaperText


async def read_paper(
    arxiv_id: str,
    max_chars: int = paths.DEFAULT_MAX_PAPER_CHARS,
) -> PaperText:
    sources = [
        (f"https://arxiv.org/html/{arxiv_id}", "arxiv_html"),
        (f"https://huggingface.co/papers/{arxiv_id}", "hf_paper"),
        (f"https://arxiv.org/abs/{arxiv_id}", "arxiv_abs"),
    ]
    return await _html.walk_sources(arxiv_id, sources, max_chars=max_chars)
