"""Econ domain pack: schema, prompts, discovery, reader."""
from __future__ import annotations

import importlib
import re

import httpx
import pytest
import respx

from paperteacher.domains._common import ParseError


# ---- schema --------------------------------------------------------------


VALID_OUTLINE_YAML = """\
paper_id: w31234
genre: empirical_causal
core_thesis: Two sentences here. Really two.
gap_filled: One thing was unidentified before.
identification:
  strategy: DiD
  source_of_variation: staggered Medicaid expansion
  key_assumption: parallel trends
  assumption_defense: pre-trends are flat for 8 quarters
  what_breaks_if_violated: time-varying state confounders
  teaching_priority: critical
specifications:
  - id: SP1
    purpose: baseline DiD
    outcome: log household consumption
    treatment_or_regressor: post-expansion indicator
    fixed_effects: state and quarter
    cluster_level: state
    sample: low-income households 2010-2022
    voice_description: regress log consumption on a treatment dummy after sweeping out state and quarter effects
    teaching_priority: critical
estimates:
  - id: ES1
    parameter_name: ATT
    point_estimate: 0.034 (0.012)
    unit: log_points
    economic_translation: a one-sd rise in eligibility raises consumption ~3% of mean
    teaching_priority: critical
robustness_checks:
  - id: RC1
    check_type: placebo
    what_it_rules_out: pre-trend selection
    result_summary: placebo coefficients null
    headline_survives: true
"""


def test_outline_round_trip():
    from paperteacher.domains.econ.models import Outline, parse_outline

    outline = parse_outline(VALID_OUTLINE_YAML)
    assert outline.paper_id == "w31234"
    assert outline.genre == "empirical_causal"
    # `ID` (identification) + SP1 + ES1 are all critical.
    assert outline.critical_ids() == ["ID", "SP1", "ES1"]
    assert outline.important_ids() == []

    redumped = outline.to_yaml()
    again = parse_outline(redumped)
    assert again.critical_ids() == outline.critical_ids()
    assert again.specifications[0].id == "SP1"
    assert again.estimates[0].unit == "log_points"
    assert isinstance(again, Outline)


def test_outline_coerces_unquoted_arxiv_id():
    """`coerce_numbers_to_str` lets the LLM emit `2603.20105` unquoted —
    same guarantee the ML pack relies on."""
    from paperteacher.domains.econ.models import Outline

    outline = Outline.model_validate({
        "paper_id": 2603.20105,
        "genre": "empirical_causal",
        "core_thesis": "x.",
        "gap_filled": "y.",
    })
    assert outline.paper_id == "2603.20105"


def test_outline_invalid_strategy_raises_parse_error():
    bad = VALID_OUTLINE_YAML.replace("strategy: DiD", "strategy: vibes-based")
    from paperteacher.domains.econ.models import parse_outline

    with pytest.raises(ParseError):
        parse_outline(bad)


def test_outline_invalid_unit_raises_parse_error():
    bad = VALID_OUTLINE_YAML.replace("unit: log_points", "unit: meters")
    from paperteacher.domains.econ.models import parse_outline

    with pytest.raises(ParseError):
        parse_outline(bad)


def test_outline_extra_fields_ignored():
    """Lenient model: unknown LLM-emitted fields don't break validation."""
    from paperteacher.domains.econ.models import parse_outline

    yaml_text = VALID_OUTLINE_YAML + "\nllm_invented_field: whatever\n"
    outline = parse_outline(yaml_text)
    assert outline.paper_id == "w31234"


def test_outline_structural_paper_no_identification():
    """Structural / theory papers can omit identification + specifications.
    The schema must accept them."""
    from paperteacher.domains.econ.models import parse_outline

    yaml_text = """\
paper_id: 2604.01234
genre: structural
core_thesis: One sentence. Two sentence.
gap_filled: closing one gap.
structural_model:
  agents: households and firms
  preferences_or_objective: CRRA utility
  technology_or_constraints: Cobb-Douglas
  equilibrium_concept: rational expectations
  teaching_priority: critical
structural_equations:
  - id: SE1
    name: Euler equation
    what_it_says: marginal utility today equals discounted expected marginal utility tomorrow
    voice_picture: a tightrope between consuming today and saving for tomorrow
    role_in_argument: this is the moment condition we estimate against
    teaching_priority: critical
"""
    outline = parse_outline(yaml_text)
    assert outline.identification is None
    assert outline.specifications == []
    assert outline.structural_equations[0].name == "Euler equation"
    assert outline.critical_ids() == ["SE1"]


def test_plan_round_trip():
    from paperteacher.domains.econ.models import parse_plan

    yaml_text = """\
paper_id: w31234
arc:
  - id: seg_01
    role: opening
    purpose: hook on the headline
    covers: [ES1]
takes:
  - claim: the design is clever but the population is narrow
    evidence: section 2 limits the sample to single mothers in four states
why_now: identification is finally mature enough to handle staggered rollouts
"""
    plan = parse_plan(yaml_text)
    assert plan.paper_id == "w31234"
    assert plan.arc[0].id == "seg_01"
    assert plan.covered_ids() == {"ES1"}


# ---- prompts -------------------------------------------------------------


_UNFILLED = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def _assert_clean(rendered: str, *must_contain: str) -> None:
    leftover = _UNFILLED.findall(rendered)
    assert not leftover, f"unfilled placeholders remain: {leftover}"
    for needle in must_contain:
        assert needle in rendered, f"expected {needle!r} not found"


def _reload_with_home(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERTEACHER_HOME", str(tmp_path))
    monkeypatch.setenv("PAPERTEACHER_PROFILE", str(tmp_path / "profile.md"))
    import paperteacher.paths as paths_mod
    importlib.reload(paths_mod)
    import paperteacher.profile as profile_mod
    importlib.reload(profile_mod)
    import paperteacher.domains.econ.prompts as prompts_mod
    importlib.reload(prompts_mod)
    return prompts_mod


def test_render_extract_substitutes_all(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_extract(
        arxiv_id="w31234",
        title="Medicaid expansion and consumption",
        taste_profile="domain: econ",
        paper_text="abstract goes here",
    )
    _assert_clean(out, "w31234", "Medicaid expansion and consumption")
    # Voice-first econ-specific guardrails should be present in the prompt.
    assert "identification" in out.lower()
    assert "economic_translation" in out


def test_render_audit_substitutes_all(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_audit(
        outline_yaml="paper_id: w31234\nidentification: {}\n",
        script="<Person1>hi</Person1><Person2>hello</Person2>",
    )
    _assert_clean(out, "w31234", "Person1")
    # The audit must explicitly ask the model to flag identification glosses.
    assert "identification" in out.lower()


def test_render_teach_default_structure(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_teach(
        arxiv_id="w31234",
        title="t",
        taste_profile="domain: econ",
        paper_text="paper",
        outline_yaml="paper_id: w31234\n",
        mode="two_host",
    )
    _assert_clean(out, "w31234", "two_host")
    assert "EPISODE PLAN" not in out


def test_render_teach_with_plan_uses_plan_structure(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_teach(
        arxiv_id="w31234",
        title="t",
        taste_profile="domain: econ",
        paper_text="paper",
        outline_yaml="paper_id: w31234\n",
        mode="single_host",
        plan_yaml="arc:\n  - role: identification\n    purpose: name the variation\n",
    )
    _assert_clean(out, "EPISODE PLAN", "FOLLOW THE PLAN")


def test_render_teach_uses_profile_target_words(tmp_path, monkeypatch):
    """Profile-driven target — same contract as the ML pack."""
    (tmp_path / "profile.md").write_text(
        "domain: econ\nlength_target_minutes: 20\nspeaking_rate: 1.0\n"
    )
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_teach(
        arxiv_id="w31234",
        title="t",
        taste_profile="domain: econ",
        paper_text="paper",
        outline_yaml="paper_id: w31234\n",
    )
    assert "3300" in out  # 20 × 165 × 1.0
    assert "20 minutes" in out or "~20 min" in out


def test_render_plan_substitutes_all(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_plan(
        arxiv_id="w31234",
        title="t",
        taste_profile="domain: econ",
        outline_yaml="paper_id: w31234\n",
    )
    _assert_clean(out, "w31234")


# ---- discovery -----------------------------------------------------------


# Minimal RSS body that feedparser will parse. Two NBER-shaped items.
NBER_RSS_BODY = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>NBER New Working Papers</title>
  <link>https://www.nber.org/</link>
  <description>x</description>
  <item>
    <title>Medicaid Expansion and Household Consumption</title>
    <link>https://www.nber.org/papers/w31234</link>
    <guid>https://www.nber.org/papers/w31234</guid>
    <description>Abstract: We exploit the staggered rollout of Medicaid expansion...</description>
    <author>Some Author</author>
    <pubDate>Mon, 28 Apr 2026 09:00:00 +0000</pubDate>
  </item>
  <item>
    <title>Another Working Paper</title>
    <link>https://www.nber.org/papers/w31235</link>
    <guid>https://www.nber.org/papers/w31235</guid>
    <description>Abstract: Yet another. </description>
    <pubDate>Mon, 28 Apr 2026 10:00:00 +0000</pubDate>
  </item>
</channel></rss>
"""


# Minimal arXiv RSS body — same shape as the ML pack consumes.
ARXIV_RSS_BODY = """<?xml version="1.0"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
<channel>
  <title>arXiv econ.GN</title>
  <link>http://arxiv.org/</link>
  <description>x</description>
  <item>
    <title>An Identification Strategy</title>
    <link>https://arxiv.org/abs/2604.12345</link>
    <description>arXiv:2604.12345v1 Abstract: Some abstract.</description>
    <dc:creator>An Author</dc:creator>
  </item>
</channel></rss>
"""


@pytest.mark.asyncio
async def test_fetch_nber_new_parses_items():
    from paperteacher.domains.econ import discovery

    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.nber.org/rss/new.xml").respond(
            status_code=200, text=NBER_RSS_BODY
        )
        cands = await discovery.fetch_nber_new(limit=10)

    assert [c.arxiv_id for c in cands] == ["w31234", "w31235"]
    assert cands[0].source == "nber_new"
    assert cands[0].url == "https://www.nber.org/papers/w31234"
    assert "Medicaid" in cands[0].title


@pytest.mark.asyncio
async def test_fetch_nber_new_returns_empty_on_http_error():
    """A flaky upstream must not crash the pipeline."""
    from paperteacher.domains.econ import discovery

    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.nber.org/rss/new.xml").respond(status_code=500)
        cands = await discovery.fetch_nber_new(limit=10)
    assert cands == []


@pytest.mark.asyncio
async def test_discover_combines_nber_and_arxiv():
    from paperteacher.domains.econ import discovery

    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.nber.org/rss/new.xml").respond(
            status_code=200, text=NBER_RSS_BODY
        )
        # Stub every default category. Return the same body — de-dup means
        # we only get one new arxiv id regardless of how many categories
        # serve it. The test is really about NBER-then-arXiv ordering.
        for cat in discovery.DEFAULT_ECON_CATEGORIES:
            router.get(f"http://export.arxiv.org/rss/{cat}").respond(
                status_code=200, text=ARXIV_RSS_BODY
            )

        cands = await discovery.discover(limit=5)

    ids = [c.arxiv_id for c in cands]
    # NBER first, arXiv second, no dupes.
    assert ids[0] == "w31234"
    assert ids[1] == "w31235"
    assert "2604.12345" in ids
    # Single arXiv id even though many categories returned it.
    assert ids.count("2604.12345") == 1


# ---- reader --------------------------------------------------------------


PAPER_HTML = (
    "<html><head><title>Working Paper Title</title></head>"
    "<body><h1>Working Paper Title</h1>"
    + "<p>This is a paragraph of substantive paper content. " * 50
    + "</p></body></html>"
)


@pytest.mark.asyncio
async def test_reader_dispatches_arxiv_id_to_arxiv_html():
    from paperteacher.domains.econ import reader

    with respx.mock(assert_all_called=False) as router:
        router.get("https://arxiv.org/html/2604.12345").respond(
            status_code=200, text=PAPER_HTML
        )
        # NBER endpoint should NOT be hit for an arxiv id.
        nber = router.get(re.compile(r"https://www\.nber\.org/papers/.*")).respond(
            status_code=200, text=PAPER_HTML
        )
        result = await reader.read_paper("2604.12345")

    assert result.source == "arxiv_html"
    assert "substantive" in result.text
    assert nber.called is False


@pytest.mark.asyncio
async def test_reader_dispatches_nber_id_to_nber_abstract():
    from paperteacher.domains.econ import reader

    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.nber.org/papers/w31234").respond(
            status_code=200, text=PAPER_HTML
        )
        # arXiv endpoints should NOT be hit for an NBER id.
        arx = router.get(re.compile(r"https://arxiv\.org/.*")).respond(
            status_code=200, text=PAPER_HTML
        )
        result = await reader.read_paper("w31234")

    assert result.source == "nber_abstract"
    assert "substantive" in result.text
    assert arx.called is False


@pytest.mark.asyncio
async def test_reader_unrecognized_id_returns_none():
    from paperteacher.domains.econ import reader

    result = await reader.read_paper("not-a-real-id-format")
    assert result.source == "none"
    assert result.text == ""


@pytest.mark.asyncio
async def test_reader_falls_through_arxiv_chain_on_404():
    from paperteacher.domains.econ import reader

    with respx.mock(assert_all_called=False) as router:
        router.get("https://arxiv.org/html/2604.12345").respond(status_code=404)
        router.get("https://huggingface.co/papers/2604.12345").respond(
            status_code=200, text=PAPER_HTML
        )
        result = await reader.read_paper("2604.12345")

    assert result.source == "hf_paper"


@pytest.mark.asyncio
async def test_reader_network_error_falls_through():
    from paperteacher.domains.econ import reader

    with respx.mock(assert_all_called=False) as router:
        router.get("https://arxiv.org/html/2604.12345").mock(
            side_effect=httpx.ConnectError("boom")
        )
        router.get("https://huggingface.co/papers/2604.12345").respond(
            status_code=200, text=PAPER_HTML
        )
        result = await reader.read_paper("2604.12345")

    assert result.source == "hf_paper"


@pytest.mark.asyncio
async def test_reader_truncates_long_text():
    from paperteacher.domains.econ import reader

    long_html = "<html><body><p>" + ("a" * 1000) + "</p></body></html>"
    with respx.mock(assert_all_called=False) as router:
        router.get("https://www.nber.org/papers/w31234").respond(
            status_code=200, text=long_html
        )
        result = await reader.read_paper("w31234", max_chars=500)

    assert result.truncated is True
    assert len(result.text) == 500


# ---- domain registration -------------------------------------------------


def test_econ_domain_registered(monkeypatch):
    monkeypatch.delenv("PAPERTEACHER_DOMAIN", raising=False)
    monkeypatch.delenv("PAPERTEACHER_PROFILE", raising=False)
    import paperteacher.domain as d
    d.reset_active()
    d.active_domain()  # trigger lazy bundled imports
    assert "econ" in d.list_domains()


def test_econ_domain_via_env(monkeypatch):
    monkeypatch.setenv("PAPERTEACHER_DOMAIN", "econ")
    monkeypatch.delenv("PAPERTEACHER_PROFILE", raising=False)
    import paperteacher.domain as d
    d.reset_active()
    assert d.active_domain().name == "econ"
