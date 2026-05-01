"""ML discovery sources: HF Daily, arXiv RSS, Semantic Scholar, and the
combined `discover` with cross-source dedup.

The discovery layer was rebuilt as part of the world-class upgrade to:
  - add Semantic Scholar as a third source (citation-velocity signal)
  - dedup on canonical (version-stripped) arXiv id, so the same paper
    showing up in two or three sources collapses to one entry

These tests pin both behaviours so a refactor can't silently regress.
"""
from __future__ import annotations

import datetime as dt
import json

import httpx
import pytest
import respx

from paperteacher.domains import _semantic_scholar
from paperteacher.domains.ml import discovery

# The S2 endpoint URL lives in the shared cross-domain fetcher; alias here
# so the tests below stay readable.
_S2_BULK_URL = _semantic_scholar._BULK_URL


# ---- helpers --------------------------------------------------------------


def _s2_payload(papers: list[dict]) -> dict:
    """Mimic the shape of api.semanticscholar.org/.../paper/search/bulk."""
    return {"total": len(papers), "data": papers}


def _s2_paper(arxiv_id: str, *, title: str, infl: int = 0, cites: int = 0) -> dict:
    return {
        "externalIds": {"ArXiv": arxiv_id},
        "title": title,
        "authors": [{"name": "A. Researcher"}],
        "abstract": "An abstract.",
        "citationCount": cites,
        "influentialCitationCount": infl,
        "publicationDate": "2026-04-15",
    }


def _hf_payload(papers: list[dict]) -> list[dict]:
    return [{"paper": p} for p in papers]


def _hf_paper(arxiv_id: str, *, title: str, upvotes: int = 0) -> dict:
    return {
        "id": arxiv_id,
        "title": title,
        "authors": [{"name": "B. Researcher"}],
        "summary": "",
        "upvotes": upvotes,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }


# ---- canonical_arxiv_id --------------------------------------------------


def test_canonical_arxiv_id_strips_version():
    """`2401.12345v3` → `2401.12345`. The bare id is the dedup key because
    arxiv redirects bare→latest, so two sources surfacing v1 and v3 are
    the same paper."""
    assert discovery.canonical_arxiv_id("2401.12345v3") == "2401.12345"
    assert discovery.canonical_arxiv_id("2401.12345") == "2401.12345"
    assert discovery.canonical_arxiv_id("  2401.12345v12  ") == "2401.12345"
    assert discovery.canonical_arxiv_id("") == ""


# ---- fetch_semantic_scholar ----------------------------------------------


@pytest.mark.asyncio
async def test_semantic_scholar_returns_arxiv_papers_ranked_by_influence():
    """Papers ordered by influentialCitationCount * 10 + citationCount.
    Papers without an ArXiv externalId are silently dropped."""
    payload = _s2_payload([
        _s2_paper("2604.00001", title="medium impact", infl=2, cites=5),
        _s2_paper("2604.00002", title="high impact", infl=10, cites=50),
        # No ArXiv id — should be dropped.
        {
            "externalIds": {"DOI": "10.1234/foo"},
            "title": "journal only",
            "authors": [], "abstract": "", "citationCount": 100,
            "influentialCitationCount": 50, "publicationDate": "2026-04-01",
        },
        _s2_paper("2604.00003", title="low impact", infl=0, cites=1),
    ])
    with respx.mock(assert_all_called=False) as router:
        router.get(_S2_BULK_URL).respond(
            status_code=200, json=payload
        )
        cands = await discovery.fetch_semantic_scholar(limit=10)

    titles = [c.title for c in cands]
    assert titles == ["high impact", "medium impact", "low impact"]
    # All entries are tagged with the source.
    assert all(c.source == "semantic_scholar" for c in cands)
    # The journal-only paper without an ArXiv id was dropped.
    assert len(cands) == 3


@pytest.mark.asyncio
async def test_semantic_scholar_failure_returns_empty():
    """S2 outages must not break discovery — it's additive, not load-bearing."""
    with respx.mock(assert_all_called=False) as router:
        router.get(_S2_BULK_URL).mock(
            side_effect=httpx.ConnectError("boom")
        )
        cands = await discovery.fetch_semantic_scholar(limit=10)
    assert cands == []


@pytest.mark.asyncio
async def test_semantic_scholar_uses_recent_window():
    """The publicationDateOrYear param is set to a recent window relative
    to `today`, so we get NEW papers, not all papers ever."""
    today = dt.date(2026, 5, 1)
    captured: dict = {}

    def _capture(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=_s2_payload([]))

    with respx.mock(assert_all_called=False) as router:
        router.get(_S2_BULK_URL).mock(side_effect=_capture)
        await discovery.fetch_semantic_scholar(window_days=90, today=today)

    # 90 days back from 2026-05-01 is 2026-01-31.
    assert captured["params"]["publicationDateOrYear"] == "2026-01-31:2026-05-01"


# ---- discover() integration ---------------------------------------------


@pytest.mark.asyncio
async def test_discover_dedups_across_sources_by_canonical_id():
    """The same paper appearing on HF Daily, arXiv RSS (with v2 suffix), and
    Semantic Scholar collapses to ONE candidate. Earlier sources win
    duplicates — HF Daily entry survives.
    """
    # HF Daily returns 2604.00001 (no version) with upvote score.
    hf_url = (
        f"https://huggingface.co/api/daily_papers?date={dt.date.today().isoformat()}"
    )
    arxiv_url = "http://export.arxiv.org/rss/cs.LG"

    # arXiv RSS returns the SAME paper with a v2 suffix and a different paper.
    arxiv_rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns="http://purl.org/rss/1.0/">
  <channel><title>cs.LG</title><link>https://arxiv.org/list/cs.LG/recent</link>
    <description>cs.LG</description></channel>
  <item>
    <title>Paper A v2</title>
    <link>https://arxiv.org/abs/2604.00001v2</link>
    <description>desc A</description>
  </item>
  <item>
    <title>Paper Unique to arXiv</title>
    <link>https://arxiv.org/abs/2604.00099</link>
    <description>desc U</description>
  </item>
</rdf:RDF>"""

    s2_payload = _s2_payload([
        _s2_paper("2604.00001", title="Paper A from S2", infl=10),
        _s2_paper("2604.00500", title="Unique to S2", infl=3),
    ])

    with respx.mock(assert_all_called=False) as router:
        router.get(hf_url).respond(
            status_code=200,
            json=_hf_payload([_hf_paper("2604.00001", title="Paper A", upvotes=42)]),
        )
        router.get(arxiv_url).respond(status_code=200, text=arxiv_rss_xml)
        router.get(_S2_BULK_URL).respond(status_code=200, json=s2_payload)

        cands = await discovery.discover(arxiv_categories=["cs.LG"], limit=20)

    # All ids should be canonical (no version suffix).
    ids = [c.arxiv_id for c in cands]
    assert "2604.00001" in ids
    # Version-stripped: no v2 leaks through.
    assert not any("v" in c.arxiv_id for c in cands)
    # Three unique canonical ids: 00001 (3 sources, deduped), 00099 (arxiv-only),
    # 00500 (S2-only).
    assert sorted(ids) == ["2604.00001", "2604.00099", "2604.00500"]
    # The 00001 candidate should keep HF Daily's metadata since HF runs first.
    paper_a = next(c for c in cands if c.arxiv_id == "2604.00001")
    assert paper_a.source == "hf_daily"
    assert paper_a.score == 42  # HF upvote count, not S2 score


@pytest.mark.asyncio
async def test_discover_handles_partial_source_failures():
    """If one source 5xxs the others still produce candidates."""
    hf_url = (
        f"https://huggingface.co/api/daily_papers?date={dt.date.today().isoformat()}"
    )
    with respx.mock(assert_all_called=False) as router:
        router.get(hf_url).respond(status_code=500)
        router.get(_S2_BULK_URL).respond(
            status_code=200,
            json=_s2_payload([_s2_paper("2604.00777", title="Survivor", infl=5)]),
        )
        # No arxiv categories requested.
        cands = await discovery.discover(arxiv_categories=[], limit=10)

    ids = [c.arxiv_id for c in cands]
    assert "2604.00777" in ids
