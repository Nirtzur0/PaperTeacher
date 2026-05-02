"""Smoke + behaviour tests for the physics domain pack.

Covers:
  - registry: pack loads on `active_domain()` and resolves by name
  - models: outline + plan parse cleanly, critical/important id helpers
    surface the right counts
  - prompts: all four templates render without unfilled placeholders, and
    the physics-specific anti-glossing instructions actually appear in
    the rendered text (so a future refactor can't accidentally drop them)
  - dispatcher: a paper stamped `physics` routes to PhysicsDomain on
    domain_for() lookup
  - INSPIRE-HEP: query string is the working `primarch <cat>` form (the
    obvious-looking `arxiv:<cat>` returns 0 hits live), and parsing
    handles the live response shape.
"""
from __future__ import annotations

import importlib
import re

import httpx
import pytest
import respx


def _reset(monkeypatch):
    import paperteacher.domain as d
    d.reset_active()


def _reload_with_home(monkeypatch, tmp_path):
    """Profile.load() is @cache'd — isolate by reloading after env-var setup."""
    monkeypatch.setenv("PAPERTEACHER_HOME", str(tmp_path))
    monkeypatch.setenv("PAPERTEACHER_PROFILE", str(tmp_path / "profile.md"))
    import paperteacher.paths as paths_mod
    importlib.reload(paths_mod)
    import paperteacher.profile as profile_mod
    importlib.reload(profile_mod)
    import paperteacher.domains.physics.prompts as prompts_mod
    importlib.reload(prompts_mod)
    return prompts_mod


# ---- registry -----------------------------------------------------------


def test_physics_domain_registered(monkeypatch):
    monkeypatch.delenv("PAPERTEACHER_DOMAIN", raising=False)
    monkeypatch.delenv("PAPERTEACHER_PROFILE", raising=False)
    _reset(monkeypatch)
    from paperteacher.domain import active_domain, list_domains
    active_domain()  # trigger lazy bundled load
    assert "physics" in list_domains()


def test_active_domain_from_env_physics(monkeypatch):
    monkeypatch.setenv("PAPERTEACHER_DOMAIN", "physics")
    monkeypatch.delenv("PAPERTEACHER_PROFILE", raising=False)
    _reset(monkeypatch)
    from paperteacher.domain import active_domain
    pack = active_domain()
    assert pack.name == "physics"
    # planner is opt-in; physics pack opted in
    assert hasattr(pack, "PlanModel")
    assert callable(pack.parse_plan)


# ---- model parsing ------------------------------------------------------


_MIN_OUTLINE_YAML = """\
paper_id: 2603.99999
type: theoretical
core_thesis: Two sentences naming the claim. Period.
gap_filled: One sentence on what was missing before.
regime_and_assumptions:
  - non-relativistic v much less than c
  - weak coupling alpha small
key_concepts:
  - id: C1
    name: Test concept
    plain_english: Plain version.
    why_it_matters: Why it matters.
    teaching_priority: critical
key_equations:
  - id: E1
    english_name: The test equation
    what_it_solves: The test problem.
    structure_in_words: Balance of two terms.
    components:
      - role: kinetic-like piece
        intuition: spreads things out
        what_if_removed: it stops spreading
    dimensional_check: both sides are energy density
    symmetries:
      - Lorentz invariant
    conservation_law: time-translation gives energy conservation
    limiting_case: in v/c -> 0 it recovers Newton
    key_trick: cancellation between two terms
    geometric_picture: a level set in field space
    fermi_estimate: plug in a kilogram and you get joules
    bridge_to_next: this leads to the next derivation
    teaching_priority: critical
  - id: E2
    english_name: A simpler equation
    what_it_solves: A simpler problem.
    structure_in_words: One term times another.
    teaching_priority: important
observables_and_predictions:
  - id: O1
    name: a measurable cross section
    predicted_value: about 1e-7 picobarns
    how_measured: at the next collider run
    falsifiability: a null result at 10x sensitivity rules it out
limitations_and_open_questions:
  - the result depends on the regularization scheme
banned_glosses:
  - the standard machinery
acronyms_to_spell_out:
  QCD: quantum chromodynamics
hard_pronunciations:
  Schrodinger: SHRO-ding-er
"""


def test_outline_parses_and_id_helpers():
    from paperteacher.domains.physics import models

    outline = models.parse_outline(_MIN_OUTLINE_YAML)
    assert outline.paper_id == "2603.99999"
    assert outline.type == "theoretical"
    # one critical concept (C1) + one critical equation (E1) = 2 critical ids;
    # one important equation (E2) = 1 important id.
    assert sorted(outline.critical_ids()) == ["C1", "E1"]
    assert outline.important_ids() == ["E2"]
    assert outline.observables_and_predictions[0].name == "a measurable cross section"
    assert "non-relativistic v much less than c" in outline.regime_and_assumptions


def test_outline_round_trips_via_yaml():
    from paperteacher.domains.physics import models

    outline = models.parse_outline(_MIN_OUTLINE_YAML)
    re_emitted = outline.to_yaml()
    again = models.parse_outline(re_emitted)
    assert again.critical_ids() == outline.critical_ids()
    assert again.observables_and_predictions[0].predicted_value == "about 1e-7 picobarns"


def test_plan_parses():
    from paperteacher.domains.physics import models

    plan_yaml = """\
paper_id: 2603.99999
arc:
  - id: seg_01
    role: opening
    covers: []
    purpose: hook the listener with the puzzle
  - id: seg_02
    role: derivation
    covers: [E1]
    callbacks: [seg_01]
    purpose: walk the equation step by step including the dimensional check
takes:
  - claim: the bound is sharp but the assumption set is doing a lot of work
    evidence: footnote 7 quietly assumes a Markov approximation
sits_alongside:
  - Bekenstein-Hawking entropy
why_now: lattice precision finally caught up
"""
    plan = models.parse_plan(plan_yaml)
    assert plan.paper_id == "2603.99999"
    assert plan.covered_ids() == {"E1"}
    assert plan.takes[0].claim.startswith("the bound is sharp")


def test_outline_invalid_yaml_raises_parse_error():
    from paperteacher.domains.physics import models
    from paperteacher.domains._common import ParseError

    with pytest.raises(ParseError):
        models.parse_outline("not: valid: yaml: at all: [")


def test_outline_missing_optional_field_still_parses():
    from paperteacher.domains.physics import models

    # The schema is intentionally lenient: missing content fields default
    # to empty strings rather than raising, so a partial outline always
    # round-trips and stage 2 can work with whatever was extracted.
    partial = "paper_id: x\ntype: theoretical\ngap_filled: y\n"
    out = models.parse_outline(partial)
    assert out.paper_id == "x"
    assert out.core_thesis == ""


# ---- prompts ------------------------------------------------------------


_UNFILLED = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def _assert_clean(rendered: str, *must_contain: str) -> None:
    leftover = _UNFILLED.findall(rendered)
    assert not leftover, f"unfilled placeholders remain: {leftover}"
    for needle in must_contain:
        assert needle in rendered, f"expected substituted content {needle!r} not found"


def test_render_extract_contains_physics_sanity_gates(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_extract(
        arxiv_id="2603.12345",
        title="A new effective field theory for dense matter",
        taste_profile="domain: physics",
        paper_text="The paper introduces a chiral effective Lagrangian.",
    )
    _assert_clean(out, "2603.12345", "A new effective field theory for dense matter")
    # The physics-specific extraction rules must appear — these are what make
    # the schema worth its weight.
    assert "dimensional_check" in out
    assert "limiting_case" in out
    assert "fermi_estimate" in out
    assert "observables_and_predictions" in out
    assert "regime_and_assumptions" in out


def test_render_audit_substitutes_all(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_audit(
        outline_yaml="paper_id: 2603.12345\nkey_equations: []\n",
        script="Person1: hi. Person2: hello.",
    )
    _assert_clean(out, "2603.12345", "Person1: hi")
    # The audit still flags physics-specific glosses, even after the global
    # banned-phrase lists were dropped.
    assert "in the appropriate limit" in out
    assert "by symmetry" in out


def test_render_teach_default_structure(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_teach(
        arxiv_id="2603.12345",
        title="t",
        taste_profile="domain: physics",
        paper_text="paper",
        outline_yaml="paper_id: 2603.12345\n",
        mode="two_host",
    )
    _assert_clean(out, "2603.12345", "two_host")
    # Default (no plan) must NOT contain the plan-section header.
    assert "EPISODE PLAN" not in out
    # Voice-first physics rules must appear — these are mandatory teaching guards.
    assert "tensor" in out.lower()
    assert "dimensional analysis" in out.lower()


def test_render_teach_with_plan_uses_plan_structure(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_teach(
        arxiv_id="2603.12345",
        title="t",
        taste_profile="domain: physics",
        paper_text="paper",
        outline_yaml="paper_id: 2603.12345\n",
        mode="single_host",
        plan_yaml="arc:\n  - role: derivation\n    purpose: walk the math\n",
    )
    _assert_clean(out, "EPISODE PLAN", "FOLLOW THE PLAN")


def test_render_plan_substitutes_all(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_plan(
        arxiv_id="2603.12345",
        title="t",
        taste_profile="domain: physics",
        outline_yaml="paper_id: 2603.12345\n",
    )
    _assert_clean(out, "2603.12345")
    # Physics arc roles should be suggested in the prompt.
    assert "regime_setup" in out or "derivation" in out


def test_render_teach_explicit_targets_override_profile(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_teach(
        arxiv_id="x",
        title="t",
        taste_profile="",
        paper_text="",
        outline_yaml="",
        target_words=999,
        target_minutes=42,
    )
    _assert_clean(out, "999", "42")


# ---- per-paper routing --------------------------------------------------


def test_domain_for_physics_paper(tmp_path, monkeypatch):
    """A paper stamped `physics` resolves back to PhysicsDomain on lookup."""
    monkeypatch.setenv("PAPERTEACHER_HOME", str(tmp_path))
    monkeypatch.delenv("PAPERTEACHER_DOMAIN", raising=False)
    monkeypatch.delenv("PAPERTEACHER_PROFILE", raising=False)
    import paperteacher.paths
    importlib.reload(paperteacher.paths)
    _reset(monkeypatch)

    from paperteacher.domain import record_domain, domain_for
    record_domain("2603.77777", "physics")
    assert domain_for("2603.77777").name == "physics"


# ---- INSPIRE-HEP --------------------------------------------------------

# Canonical INSPIRE response for one hit. The real API returns this shape
# under hits.hits[].metadata; the parsing has to pull out arxiv_eprints[0]
# .value as the canonical id and the first abstract / first title.
_INSPIRE_BODY = """{
  "hits": {
    "total": 1,
    "hits": [
      {
        "metadata": {
          "titles": [{"title": "Non-Gaussian hydrodynamic fluctuations"}],
          "authors": [
            {"full_name": "Smith, A."},
            {"full_name": "Jones, B."}
          ],
          "arxiv_eprints": [
            {"categories": ["hep-th"], "value": "2604.27730"}
          ],
          "abstracts": [{"value": "An abstract about hydrodynamic fluctuations."}]
        }
      }
    ]
  }
}"""


@pytest.mark.asyncio
async def test_fetch_inspire_uses_primarch_query():
    """The bare `arxiv:<cat>` form returns 0 hits live; only `primarch <cat>`
    works. Pin the correct query so a future regression that silently breaks
    the source surfaces immediately."""
    from paperteacher.domains.physics import discovery

    with respx.mock(assert_all_called=False) as router:
        route = router.get(
            "https://inspirehep.net/api/literature",
            params={"q": "primarch hep-th"},
        ).respond(status_code=200, text=_INSPIRE_BODY)
        cands = await discovery.fetch_inspire_hep("hep-th", limit=5)

    assert route.called, "INSPIRE was queried with the wrong q parameter"
    assert [c.arxiv_id for c in cands] == ["2604.27730"]
    assert cands[0].title.startswith("Non-Gaussian")
    assert cands[0].authors == ["Smith, A.", "Jones, B."]
    assert cands[0].source == "inspire_hep-th"
    assert cands[0].url == "https://arxiv.org/abs/2604.27730"


@pytest.mark.asyncio
async def test_fetch_inspire_returns_empty_on_http_error():
    """Live INSPIRE has been bumpy — a 5xx must not crash the pipeline."""
    from paperteacher.domains.physics import discovery

    with respx.mock(assert_all_called=False) as router:
        router.get("https://inspirehep.net/api/literature").respond(status_code=503)
        cands = await discovery.fetch_inspire_hep("hep-th", limit=5)
    assert cands == []


@pytest.mark.asyncio
async def test_fetch_inspire_drops_records_without_arxiv_id():
    """A record with no arxiv_eprints (journal-only with no preprint) can't
    flow through the rest of the pipeline — drop, don't crash."""
    from paperteacher.domains.physics import discovery

    body_no_arxiv = """{"hits": {"total": 1, "hits": [
      {"metadata": {"titles": [{"title": "T"}], "arxiv_eprints": []}}
    ]}}"""
    with respx.mock(assert_all_called=False) as router:
        router.get("https://inspirehep.net/api/literature").respond(
            status_code=200, text=body_no_arxiv
        )
        cands = await discovery.fetch_inspire_hep("hep-th", limit=5)
    assert cands == []


@pytest.mark.asyncio
async def test_fetch_inspire_network_error():
    from paperteacher.domains.physics import discovery

    with respx.mock(assert_all_called=False) as router:
        router.get("https://inspirehep.net/api/literature").mock(
            side_effect=httpx.ConnectError("boom")
        )
        cands = await discovery.fetch_inspire_hep("hep-th", limit=5)
    assert cands == []
