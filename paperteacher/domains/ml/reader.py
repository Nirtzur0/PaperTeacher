"""Read full paper text using a fallback chain optimized for math-heavy ML papers.

Source order (best-first):
  1. ar5iv.labs.arxiv.org/html/{id}  — LaTeXML-rendered HTML, broadest coverage
                                      (back to ~1990s); emits LaTeX source
                                      verbatim alongside MathML, which we
                                      extract so equations arrive as `$...$`
                                      instead of MathML or symbol-salad.
  2. arxiv.org/html/{id}             — arXiv's official HTML, post-2023 papers
                                      mostly. Reasonable LaTeX preservation.
  3. huggingface.co/papers/{id}      — abstract + comments; useful when no
                                      HTML rendering exists at all.
  4. arxiv.org/abs/{id}              — abstract-only, the floor.

The ar5iv-specific strip extracts each `<annotation encoding="application/x-tex">`
payload (the original LaTeX source LaTeXML keeps alongside the rendered
MathML) and surfaces equation/theorem tags as `[(5)]` markers — frontier
LLMs reason over LaTeX-in-Markdown more fluently than over MathML or
plain-text symbol salad, so this is the highest-leverage step in the reader.

ML-domain reader. Other domain packs ship their own reader (e.g. DOI
redirection for journals); HTML plumbing (fetch, title, truncate) lives in
`domains/_html.py` so each pack's reader stays focused on the source list.
"""
from __future__ import annotations

import logging
import re

import httpx
from bs4 import BeautifulSoup

from ... import paths
from .. import _html
from .._common import PaperText

log = logging.getLogger(__name__)

# Fragment ar5iv emits when LaTeXML hits a fatal parse error on a paper.
# When the bulk of the page is error markers we'd rather fall through to
# arxiv.org/html, even if the overall length passes the size floor.
_AR5IV_FATAL_FRAGMENTS = (
    "Couldn’t fetch latex source",  # noqa: RUF001 - exact ar5iv string
    "Couldn't fetch latex source",
)


def _ar5iv_strip(html: str) -> str:
    """ar5iv-specific HTML → text. Preserves LaTeX math + equation numbers.

    Replaces every `<math>` element with its `<annotation
    encoding="application/x-tex">` payload — that's the original LaTeX
    source LaTeXML keeps alongside the rendered MathML. Inline math is
    wrapped `$...$`; display math (`display="block"` or inside an
    `ltx_equation` block) wraps `$$...$$`. ar5iv's `ltx_tag` spans (which
    carry the human-visible "(5)" or "Theorem 2.1") are surfaced as
    `[Tag]` markers so the outline can reference equations by number.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    for math in soup.find_all("math"):
        ann = math.find("annotation", attrs={"encoding": "application/x-tex"})
        latex = (ann.get_text(strip=True) if ann else math.get_text(" ", strip=True)) or ""
        # Display vs inline. ar5iv sets display="block" on the <math> for
        # numbered equations; the surrounding container is also marked
        # `ltx_equation` / `ltx_equationgroup`.
        is_display = math.get("display") == "block"
        if not is_display:
            for ancestor in math.parents:
                cls = ancestor.get("class") if hasattr(ancestor, "get") else None
                if cls and any("ltx_equation" in c for c in cls):
                    is_display = True
                    break
        delim = "$$" if is_display else "$"
        math.replace_with(f" {delim}{latex}{delim} ")

    for tag in soup.find_all(class_=re.compile(r"\bltx_tag\b")):
        text = tag.get_text(strip=True)
        if text:
            tag.replace_with(f" [{text}] ")

    return "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )


def _looks_like_failed_render(text: str) -> bool:
    """Heuristic: ar5iv pages that hit fatal LaTeXML errors still return 200
    but the body is mostly error chrome. When the fragment shows up more than
    once we're better off falling through to arxiv.org/html."""
    return any(text.count(frag) > 1 for frag in _AR5IV_FATAL_FRAGMENTS)


async def read_paper(
    arxiv_id: str,
    max_chars: int = paths.DEFAULT_MAX_PAPER_CHARS,
) -> PaperText:
    """Try ar5iv, arxiv HTML, HF paper, then arXiv abstract. Always returns;
    `source == "none"` indicates total failure.

    The arXiv URLs omit the version suffix — arxiv redirects to the latest
    revision, so this works for v1 / v2 / v3 papers alike.

    `source` values:
      ar5iv      - LaTeX-extracted, math-rich (best)
      arxiv_html - arXiv official HTML (good)
      hf_paper   - HuggingFace abstract page (limited)
      arxiv_abs  - arXiv abstract only (worst)
      none       - all sources failed
    """
    sources: list[tuple[str, str, callable]] = [
        (f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}", "ar5iv", _ar5iv_strip),
        (f"https://arxiv.org/html/{arxiv_id}", "arxiv_html", _html.strip),
        (f"https://huggingface.co/papers/{arxiv_id}", "hf_paper", _html.strip),
        (f"https://arxiv.org/abs/{arxiv_id}", "arxiv_abs", _html.strip),
    ]
    async with httpx.AsyncClient(
        timeout=45, headers={"User-Agent": paths.USER_AGENT}
    ) as client:
        for url, label, strip in sources:
            html = await _html.fetch(url, client)
            if html is None:
                continue
            text = strip(html)
            if len(text) < _html.MIN_USEFUL_TEXT_LEN:
                log.debug("fetch %s extracted text too short (%d chars)", url, len(text))
                continue
            if label == "ar5iv" and _looks_like_failed_render(text):
                log.debug("fetch %s: ar5iv render looks broken — falling through", url)
                continue
            log.info("read_paper %s: %s (%d chars)", arxiv_id, label, len(text))
            return _html.truncate(
                PaperText(arxiv_id, _html.title_from(html), text, label), max_chars
            )

    log.warning("read_paper %s: all sources failed", arxiv_id)
    return PaperText(arxiv_id, "", "", "none")
