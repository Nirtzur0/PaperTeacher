"""Cross-domain Semantic Scholar fetcher.

Pins the abstraction contract so future domain packs can wire S2 in
without re-implementing it: the shared fetcher must (1) honor the
configured id_type, dropping items missing that external-id, (2) apply
canonicalize ONLY to the candidate's id field — not the URL, since some
packs encode their canonical id (e.g. neuro replaces `/` with `_` for
filesystem safety) and the URL has to remain fetchable, and (3) fall
through silently to an empty list on any network/auth error.
"""
from __future__ import annotations

import datetime as dt

import httpx
import pytest
import respx

from paperteacher.domains._semantic_scholar import (
    _BULK_URL,
    fetch_semantic_scholar,
)


def _payload(items: list[dict]) -> dict:
    return {"total": len(items), "data": items}


def _item(*, ext_ids: dict, title: str, infl: int = 0, cites: int = 0) -> dict:
    return {
        "externalIds": ext_ids,
        "title": title,
        "authors": [{"name": "Author One"}],
        "abstract": "abs",
        "citationCount": cites,
        "influentialCitationCount": infl,
        "publicationDate": "2026-04-15",
    }


# ---- contract: id_type filtering ----------------------------------------


@pytest.mark.asyncio
async def test_drops_items_missing_the_requested_id_type():
    """Configured id_type=ArXiv → items keyed only on DOI are silently
    dropped. The opposite case (id_type=DOI dropping arXiv-only) is the
    same path — one test pins the policy."""
    payload = _payload([
        _item(ext_ids={"ArXiv": "2604.00001"}, title="has arxiv"),
        _item(ext_ids={"DOI": "10.1234/foo"}, title="doi only — drop"),
        _item(ext_ids={}, title="no ids — drop"),
    ])
    with respx.mock(assert_all_called=False) as router:
        router.get(_BULK_URL).respond(status_code=200, json=payload)
        cands = await fetch_semantic_scholar(
            query="x", fields_of_study="Computer Science", id_type="ArXiv",
        )
    assert [c.title for c in cands] == ["has arxiv"]


# ---- contract: canonicalize is for id only, NOT URL ---------------------


@pytest.mark.asyncio
async def test_canonicalize_does_not_touch_url():
    """The neuro use case: DOI `10.1101/foo` becomes `10.1101_foo` for
    filesystem-safe storage, but the doi.org URL must keep the slash."""
    payload = _payload([
        _item(ext_ids={"DOI": "10.1101/2026.04.15.abc"}, title="paper"),
    ])
    with respx.mock(assert_all_called=False) as router:
        router.get(_BULK_URL).respond(status_code=200, json=payload)
        cands = await fetch_semantic_scholar(
            query="x",
            fields_of_study="Biology",
            id_type="DOI",
            canonicalize=lambda d: d.replace("/", "_", 1),
            url_template="https://doi.org/{id}",
        )
    assert len(cands) == 1
    c = cands[0]
    assert c.arxiv_id == "10.1101_2026.04.15.abc"        # encoded for storage
    assert c.url == "https://doi.org/10.1101/2026.04.15.abc"  # raw for HTTP


# ---- contract: ranking is influential * 10 + cites ----------------------


@pytest.mark.asyncio
async def test_ranks_by_influence_then_citations():
    payload = _payload([
        _item(ext_ids={"ArXiv": "a"}, title="medium", infl=2, cites=5),
        _item(ext_ids={"ArXiv": "b"}, title="high", infl=10, cites=50),
        _item(ext_ids={"ArXiv": "c"}, title="low", infl=0, cites=1),
    ])
    with respx.mock(assert_all_called=False) as router:
        router.get(_BULK_URL).respond(status_code=200, json=payload)
        cands = await fetch_semantic_scholar(
            query="x", fields_of_study="Computer Science",
        )
    assert [c.title for c in cands] == ["high", "medium", "low"]


# ---- contract: tie-broken across packs by source_tag --------------------


@pytest.mark.asyncio
async def test_source_tag_is_configurable():
    """Default `source_tag` is `semantic_scholar` so logs and dedup all see
    the same string regardless of pack. A pack with multiple S2 calls (or
    one that wants to differentiate) overrides."""
    payload = _payload([_item(ext_ids={"ArXiv": "a"}, title="t")])
    with respx.mock(assert_all_called=False) as router:
        router.get(_BULK_URL).respond(status_code=200, json=payload)
        cands = await fetch_semantic_scholar(
            query="x",
            fields_of_study="Physics",
            source_tag="s2_physics",
        )
    assert cands[0].source == "s2_physics"


# ---- contract: silent failure ------------------------------------------


@pytest.mark.asyncio
async def test_network_error_returns_empty():
    with respx.mock(assert_all_called=False) as router:
        router.get(_BULK_URL).mock(side_effect=httpx.ConnectError("boom"))
        cands = await fetch_semantic_scholar(
            query="x", fields_of_study="Computer Science",
        )
    assert cands == []


@pytest.mark.asyncio
async def test_5xx_returns_empty():
    with respx.mock(assert_all_called=False) as router:
        router.get(_BULK_URL).respond(status_code=503, json={"error": "down"})
        cands = await fetch_semantic_scholar(
            query="x", fields_of_study="Computer Science",
        )
    assert cands == []


# ---- contract: window plumbing -----------------------------------------


@pytest.mark.asyncio
async def test_window_days_passed_to_api():
    today = dt.date(2026, 5, 1)
    captured: dict = {}

    def _capture(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=_payload([]))

    with respx.mock(assert_all_called=False) as router:
        router.get(_BULK_URL).mock(side_effect=_capture)
        await fetch_semantic_scholar(
            query="x",
            fields_of_study="Physics",
            window_days=30,
            today=today,
        )

    # 30 days back from 2026-05-01 → 2026-04-01.
    assert captured["params"]["publicationDateOrYear"] == "2026-04-01:2026-05-01"
    assert captured["params"]["fieldsOfStudy"] == "Physics"
    assert captured["params"]["query"] == "x"


# ---- contract: per-pack wiring stays correct ---------------------------


@pytest.mark.asyncio
async def test_ml_pack_wiring_returns_arxiv_keyed_candidates():
    """The ml pack's fetch_semantic_scholar configures id_type=ArXiv +
    canonical_arxiv_id. Acts as a smoke test that the wiring through the
    shared module produces the same shape ML always returned."""
    from paperteacher.domains.ml import discovery as ml_disc
    payload = _payload([
        _item(ext_ids={"ArXiv": "2604.00001v3"}, title="A", infl=5),
    ])
    with respx.mock(assert_all_called=False) as router:
        router.get(_BULK_URL).respond(status_code=200, json=payload)
        cands = await ml_disc.fetch_semantic_scholar(limit=10)
    assert len(cands) == 1
    # Version stripped (canonicalize), URL fetchable (raw).
    assert cands[0].arxiv_id == "2604.00001"
    assert cands[0].url == "https://arxiv.org/abs/2604.00001v3"
    assert cands[0].source == "semantic_scholar"


@pytest.mark.asyncio
async def test_physics_pack_wiring_uses_physics_fields_of_study():
    from paperteacher.domains.physics import discovery as phys_disc
    captured: dict = {}

    def _capture(request):
        captured["fos"] = request.url.params.get("fieldsOfStudy")
        return httpx.Response(200, json=_payload([]))

    with respx.mock(assert_all_called=False) as router:
        router.get(_BULK_URL).mock(side_effect=_capture)
        await phys_disc.fetch_semantic_scholar(limit=5)
    assert captured["fos"] == "Physics"


@pytest.mark.asyncio
async def test_neuro_pack_wiring_uses_doi_id_and_encodes_for_storage():
    from paperteacher.domains.neuro import discovery as neuro_disc
    payload = _payload([
        _item(ext_ids={"DOI": "10.1101/2026.04.15.foo"}, title="N1", infl=3),
        # arXiv-only paper from a Biology query: no DOI → silently dropped.
        _item(ext_ids={"ArXiv": "2604.00001"}, title="should drop", infl=20),
    ])
    captured: dict = {}

    def _respond(request):
        captured["fos"] = request.url.params.get("fieldsOfStudy")
        return httpx.Response(200, json=payload)

    with respx.mock(assert_all_called=False) as router:
        router.get(_BULK_URL).mock(side_effect=_respond)
        cands = await neuro_disc.fetch_semantic_scholar_neuro(limit=10)

    assert captured["fos"] == "Biology,Medicine"
    assert [c.arxiv_id for c in cands] == ["10.1101_2026.04.15.foo"]
    assert cands[0].url == "https://doi.org/10.1101/2026.04.15.foo"
