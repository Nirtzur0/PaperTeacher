"""Reader fallback chain. Mocks httpx via respx so the test is hermetic.

Source order (best-first): ar5iv → arxiv.org/html → hf paper → arxiv abstract.
ar5iv goes first because it has the broadest LaTeXML coverage and exposes
the original LaTeX source via `<annotation encoding="application/x-tex">`,
which we extract for clean math input.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from paperteacher.domains.ml import reader


PAPER_HTML = (
    "<html><head><title>A Real Paper Title</title></head>"
    "<body><h1>A Real Paper Title</h1>"
    + "<p>This is a paragraph of substantive paper content. " * 50
    + "</p></body></html>"
)


# A miniature ar5iv page: LaTeXML wraps each <math> in a presentation MathML
# tree with an <annotation encoding="application/x-tex"> sibling carrying the
# raw LaTeX source. Equation numbers are emitted as <span class="ltx_tag">.
AR5IV_HTML = """
<!doctype html>
<html><head><title>Score-based diffusion paper</title></head>
<body>
<h1>Score-based diffusion paper</h1>
<p>The score is a vector field defined as
   <math display="inline">
     <semantics>
       <mrow><mo>∇</mo><mi>x</mi></mrow>
       <annotation encoding="application/x-tex">\\nabla_x \\log p(x)</annotation>
     </semantics>
   </math>
   over the data distribution.</p>
<table class="ltx_equation">
  <tr>
    <td>
      <math display="block">
        <semantics>
          <mrow></mrow>
          <annotation encoding="application/x-tex">L = \\mathbb{E}\\bigl[\\| s_\\theta(x) - \\nabla_x \\log p(x) \\|^2 \\bigr]</annotation>
        </semantics>
      </math>
    </td>
    <td><span class="ltx_tag ltx_tag_equation">(5)</span></td>
  </tr>
</table>
<p>And we recover the score-matching loss above. """ + ("Body content. " * 50) + """</p>
</body></html>
"""


@pytest.mark.asyncio
async def test_ar5iv_succeeds_first():
    """If ar5iv is reachable, we never fall through. Math from <annotation>
    arrives as `$...$` (inline) or `$$...$$` (display)."""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://ar5iv.labs.arxiv.org/html/2603.20105").respond(
            status_code=200, text=AR5IV_HTML
        )
        # Fallbacks should not be touched.
        arx_html = router.get("https://arxiv.org/html/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )
        hf = router.get("https://huggingface.co/papers/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )
        abs_ = router.get("https://arxiv.org/abs/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )

        result = await reader.read_paper("2603.20105")

    assert result.source == "ar5iv"
    # Inline LaTeX from <annotation> survives intact, wrapped in $...$.
    assert "$\\nabla_x \\log p(x)$" in result.text
    # Display LaTeX wrapped in $$...$$ for the boxed equation.
    assert "$$L = \\mathbb{E}" in result.text
    # Equation tag preserved as `[(5)]`.
    assert "[(5)]" in result.text
    # Fallbacks were not consulted.
    assert arx_html.called is False
    assert hf.called is False
    assert abs_.called is False


@pytest.mark.asyncio
async def test_falls_through_to_arxiv_html_when_ar5iv_404s():
    with respx.mock(assert_all_called=False) as router:
        router.get("https://ar5iv.labs.arxiv.org/html/2603.20105").respond(status_code=404)
        router.get("https://arxiv.org/html/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )

        result = await reader.read_paper("2603.20105")

    assert result.source == "arxiv_html"
    assert "substantive paper content" in result.text


@pytest.mark.asyncio
async def test_falls_through_to_hf_when_html_sources_404():
    with respx.mock(assert_all_called=False) as router:
        router.get("https://ar5iv.labs.arxiv.org/html/2603.20105").respond(status_code=404)
        router.get("https://arxiv.org/html/2603.20105").respond(status_code=404)
        router.get("https://huggingface.co/papers/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )

        result = await reader.read_paper("2603.20105")

    assert result.source == "hf_paper"


@pytest.mark.asyncio
async def test_falls_through_when_extracted_text_too_short():
    """A 200 OK with bloated HTML but tiny actual text should NOT count.
    Both ar5iv and arxiv_html should fall through, not just one of them.
    """
    bloated_html = "<html><body>" + ("<script>" + "x" * 800 + "</script>") + "<p>hi</p></body></html>"
    with respx.mock(assert_all_called=False) as router:
        router.get("https://ar5iv.labs.arxiv.org/html/2603.20105").respond(
            status_code=200, text=bloated_html
        )
        router.get("https://arxiv.org/html/2603.20105").respond(
            status_code=200, text=bloated_html
        )
        router.get("https://huggingface.co/papers/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )

        result = await reader.read_paper("2603.20105")

    # Both ar5iv and arxiv_html stripped to "hi" → fall through to HF.
    assert result.source == "hf_paper"


@pytest.mark.asyncio
async def test_ar5iv_render_with_fatal_errors_falls_through():
    """ar5iv occasionally returns a 200 page that's mostly LaTeXML error
    chrome. We detect repeated fatal-fragment markers and fall through."""
    broken_ar5iv = (
        "<html><body>"
        + "<p>Couldn't fetch latex source</p>" * 5
        + "<p>" + ("Body content. " * 80) + "</p>"
        + "</body></html>"
    )
    with respx.mock(assert_all_called=False) as router:
        router.get("https://ar5iv.labs.arxiv.org/html/2603.20105").respond(
            status_code=200, text=broken_ar5iv
        )
        router.get("https://arxiv.org/html/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )

        result = await reader.read_paper("2603.20105")

    assert result.source == "arxiv_html"


@pytest.mark.asyncio
async def test_arxiv_html_url_has_no_version_suffix():
    """v1 hardcoding was a bug — the URL must work for v2 / v3 papers too.
    Now also enforced for ar5iv."""
    with respx.mock(assert_all_called=False) as router:
        ar5iv_route = router.get(
            "https://ar5iv.labs.arxiv.org/html/2603.20105"
        ).respond(status_code=200, text=PAPER_HTML)
        # Don't register a v1 route — respx will raise on unmatched.
        await reader.read_paper("2603.20105")
        assert ar5iv_route.called


@pytest.mark.asyncio
async def test_all_sources_fail_returns_none_source():
    with respx.mock(assert_all_called=False) as router:
        router.get("https://ar5iv.labs.arxiv.org/html/2603.20105").respond(status_code=500)
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
        # ar5iv returns the long HTML; it'll succeed and we expect truncation.
        router.get("https://ar5iv.labs.arxiv.org/html/2603.20105").respond(
            status_code=200, text=long_html
        )
        result = await reader.read_paper("2603.20105", max_chars=500)

    assert result.truncated is True
    assert len(result.text) == 500


@pytest.mark.asyncio
async def test_network_error_falls_through():
    with respx.mock(assert_all_called=False) as router:
        router.get("https://ar5iv.labs.arxiv.org/html/2603.20105").mock(
            side_effect=httpx.ConnectError("boom")
        )
        router.get("https://arxiv.org/html/2603.20105").respond(
            status_code=200, text=PAPER_HTML
        )
        result = await reader.read_paper("2603.20105")

    assert result.source == "arxiv_html"
