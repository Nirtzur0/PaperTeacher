"""Read full paper text using the fallback chain: arXiv HTML -> HF papers -> arXiv abstract."""
from __future__ import annotations

from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup


@dataclass
class PaperText:
    arxiv_id: str
    title: str
    text: str
    source: str
    truncated: bool = False


def _strip(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


async def _try(url: str, client: httpx.AsyncClient) -> str | None:
    try:
        r = await client.get(url, follow_redirects=True)
        if r.status_code == 200 and len(r.text) > 500:
            return r.text
    except httpx.HTTPError:
        pass
    return None


async def read_paper(arxiv_id: str, max_chars: int = 120_000) -> PaperText:
    """Try arxiv html, then HF paper page, then arXiv abstract. Always returns something."""
    async with httpx.AsyncClient(timeout=45, headers={"User-Agent": "PaperTeacher/0.1"}) as client:
        html = await _try(f"https://arxiv.org/html/{arxiv_id}v1", client)
        if html:
            text = _strip(html)
            return _truncate(PaperText(arxiv_id, _title_from(html), text, "arxiv_html"), max_chars)

        html = await _try(f"https://huggingface.co/papers/{arxiv_id}", client)
        if html:
            text = _strip(html)
            return _truncate(PaperText(arxiv_id, _title_from(html), text, "hf_paper"), max_chars)

        html = await _try(f"https://arxiv.org/abs/{arxiv_id}", client)
        if html:
            text = _strip(html)
            return _truncate(
                PaperText(arxiv_id, _title_from(html), text, "arxiv_abs"), max_chars
            )

    return PaperText(arxiv_id, "", "", "none")


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
