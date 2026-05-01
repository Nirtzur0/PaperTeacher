"""Reader fallback chain. Mocks httpx via respx so the test is hermetic."""
from __future__ import annotations

import httpx
import pytest
import respx

from paperteacher import reader


PAPER_HTML = (
    "<html><head><title>A Real Paper Title</title></head>"
    "<body><h1>A Real Paper Title</h1>"
    + "<p>This is a paragraph of substantive paper content. " * 50
    + "</p></body></html>"
)


@pytest.mark.asyncio
async def test_arxiv_html_succeeds_first():
    """If arxiv HTML is reachable, we never fall through."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://arxiv.org/html/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )
        # These should NOT be hit; if they are, the test still passes but
        # we'd want to know — assert via call counts below.
        hf = router.get("https://huggingface.co/papers/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )
        abs_ = router.get("https://arxiv.org/abs/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )

        result = await reader.read_paper("2603.20105")

    assert result.source == "arxiv_html"
    assert "substantive paper content" in result.text
    assert hf.called is False
    assert abs_.called is False


@pytest.mark.asyncio
async def test_falls_through_to_hf_when_arxiv_404s():
    with respx.mock(assert_all_called=False) as router:
        router.get("https://arxiv.org/html/2603.20105").respond(status_code=404)
        router.get("https://huggingface.co/papers/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )

        result = await reader.read_paper("2603.20105")

    assert result.source == "hf_paper"


@pytest.mark.asyncio
async def test_falls_through_when_extracted_text_too_short():
    """A 200 OK with bloated HTML but tiny actual text should NOT count.
    Previously the reader checked raw HTML length; now it checks extracted text.
    """
    bloated_html = "<html><body>" + ("<script>" + "x" * 800 + "</script>") + "<p>hi</p></body></html>"
    with respx.mock(assert_all_called=False) as router:
        router.get("https://arxiv.org/html/2603.20105").respond(
            status_code=200, text=bloated_html
        )
        router.get("https://huggingface.co/papers/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )

        result = await reader.read_paper("2603.20105")

    # Bloated HTML stripped to ~"hi" — should fall through to HF.
    assert result.source == "hf_paper"


@pytest.mark.asyncio
async def test_arxiv_html_url_has_no_version_suffix():
    """v1 hardcoding was a bug — the URL must work for v2 / v3 papers too."""
    with respx.mock(assert_all_called=False) as router:
        # Match WITHOUT a version suffix; arxiv handles redirect to latest.
        route = router.get("https://arxiv.org/html/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )
        # Explicitly NOT registering a v1 route — if reader still requests v1,
        # respx will raise on unmatched.
        await reader.read_paper("2603.20105")
        assert route.called


@pytest.mark.asyncio
async def test_all_sources_fail_returns_none_source():
    with respx.mock(assert_all_called=False) as router:
        router.get("https://arxiv.org/html/2603.20105").respond(status_code=500)
        router.get("https://huggingface.co/papers/2603.20105").respond(status_code=500)
        router.get("https://arxiv.org/abs/2603.20105").respond(status_code=500)

        result = await reader.read_paper("2603.20105")

    assert result.source == "none"
    assert result.text == ""


@pytest.mark.asyncio
async def test_truncates_long_text():
    long_html = "<html><body><p>" + ("a" * 1000) + "</p></body></html>"
    with respx.mock(assert_all_called=False) as router:
        router.get("https://arxiv.org/html/2603.20105").respond(
            status_code=200, text=long_html
        )
        result = await reader.read_paper("2603.20105", max_chars=500)

    assert result.truncated is True
    assert len(result.text) == 500


@pytest.mark.asyncio
async def test_network_error_falls_through():
    with respx.mock(assert_all_called=False) as router:
        router.get("https://arxiv.org/html/2603.20105").mock(
            side_effect=httpx.ConnectError("boom")
        )
        router.get("https://huggingface.co/papers/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )
        result = await reader.read_paper("2603.20105")

    assert result.source == "hf_paper"
