"""Smoke tests for the ml-domain prompt renderers.

The whole pipeline depends on these four templates rendering without
KeyError and without leaving raw `{placeholder}` markers behind. They're
big enough that a typo (a missed kwarg, a forgotten substitution) is
hard to spot by eye — and a regression here means stage 2 just fails
at runtime in production. Cheap test, very high signal.
"""
from __future__ import annotations

import importlib
import re

import pytest


def _reload_with_home(monkeypatch, tmp_path):
    """Profile.load() is @cache'd, so isolate by reloading after env-var setup."""
    monkeypatch.setenv("PAPERTEACHER_HOME", str(tmp_path))
    monkeypatch.setenv("PAPERTEACHER_PROFILE", str(tmp_path / "profile.md"))
    import paperteacher.paths as paths_mod
    importlib.reload(paths_mod)
    import paperteacher.profile as profile_mod
    importlib.reload(profile_mod)
    import paperteacher.domains.ml.prompts as prompts_mod
    importlib.reload(prompts_mod)
    return prompts_mod


# Any `{name}` left in the rendered output means a placeholder didn't
# get filled. Anchored to identifier-style names so we don't false-flag
# YAML braces inside paper_text/outline_yaml that the user passed in
# verbatim.
_UNFILLED = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def _assert_clean(rendered: str, *must_contain: str) -> None:
    leftover = _UNFILLED.findall(rendered)
    assert not leftover, f"unfilled placeholders remain: {leftover}"
    for needle in must_contain:
        assert needle in rendered, f"expected substituted content {needle!r} not found"


def test_render_extract_substitutes_all(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_extract(
        arxiv_id="2501.12345",
        title="Score-based diffusion in curved spaces",
        taste_profile="domain: ml",
        paper_text="The paper introduces a Riemannian score model.",
    )
    _assert_clean(out, "2501.12345", "Score-based diffusion in curved spaces")


def test_render_audit_substitutes_all(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_audit(
        outline_yaml="paper_id: 2501.12345\nkey_equations: []\n",
        script="Person1: hi. Person2: hello.",
    )
    _assert_clean(out, "2501.12345", "Person1: hi")


def test_render_teach_default_structure(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_teach(
        arxiv_id="2501.12345",
        title="t",
        taste_profile="domain: ml",
        paper_text="paper",
        outline_yaml="paper_id: 2501.12345\n",
        mode="two_host",
    )
    _assert_clean(out, "2501.12345", "two_host")
    # Default (no plan) must NOT contain the plan-section header.
    assert "EPISODE PLAN" not in out


def test_render_teach_with_plan_uses_plan_structure(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_teach(
        arxiv_id="2501.12345",
        title="t",
        taste_profile="domain: ml",
        paper_text="paper",
        outline_yaml="paper_id: 2501.12345\n",
        mode="single_host",
        plan_yaml="arc:\n  - role: opening\n    purpose: hook\n",
    )
    _assert_clean(out, "EPISODE PLAN", "FOLLOW THE PLAN")


def test_render_plan_substitutes_all(tmp_path, monkeypatch):
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_plan(
        arxiv_id="2501.12345",
        title="t",
        taste_profile="domain: ml",
        outline_yaml="paper_id: 2501.12345\n",
    )
    _assert_clean(out, "2501.12345")


def test_render_teach_uses_profile_target_words(tmp_path, monkeypatch):
    """The teach prompt's word/minute target should come from the profile —
    not a hardcoded constant. Setting length_target_minutes in profile.md
    must be reflected in the rendered prompt."""
    (tmp_path / "profile.md").write_text(
        "domain: ml\nlength_target_minutes: 20\nspeaking_rate: 1.0\n"
    )
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_teach(
        arxiv_id="2501.12345",
        title="t",
        taste_profile="domain: ml",
        paper_text="paper",
        outline_yaml="paper_id: 2501.12345\n",
    )
    # 20 min × 165 wpm × 1.0 = 3300 words. The exact number being in the
    # output is the contract: profile drives the target, not a constant.
    assert "3300" in out
    assert "20 minutes" in out or "~20 min" in out


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


# ---- v0.3 prompt-content invariants ------------------------------------
#
# The prompts encode pedagogical and voice-first rules that the upstream
# LLM is supposed to follow. The rules themselves only "exist" if they
# show up in the rendered prompt — these tests pin specific invariants so
# someone can't quietly remove (say) the anti-anthropomorphism block in a
# refactor and have the regression go unnoticed.


def test_extract_prompt_asks_for_pedagogy_fields(tmp_path, monkeypatch):
    """EXTRACT must request the new outline fields the audit relies on."""
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_extract(
        arxiv_id="x", title="t", taste_profile="", paper_text="",
    )
    for needle in (
        "stake_claim",
        "first_concrete_instance",
        "prior_attempts",
        "ablations",
        "assumption_boundaries",
        "benchmark_caveats",
        "compute_envelope",
        "common_misreadings",
        "symbol_glossary",
    ):
        assert needle in out, f"EXTRACT prompt is missing the {needle!r} section"


def test_teach_prompt_includes_voice_guide_and_invariants(tmp_path, monkeypatch):
    """TEACH must carry the pronunciation guide, anti-anthropomorphism,
    concrete-first, and result-with-baseline rules. These are the
    voice-first contract the AUDIT stage will check."""
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_teach(
        arxiv_id="x", title="t", taste_profile="",
        paper_text="", outline_yaml="",
    )
    # Voice guide is included.
    assert "PRONUNCIATION GUIDE" in out
    assert "ELBO" in out  # specific ML acronym phonetic
    assert "i.i.d." in out
    # Hard rules.
    assert "ANTI-ANTHROPOMORPHISM" in out
    assert "CONCRETE-FIRST" in out
    assert "RESULT-WITH-BASELINE" in out
    # Symbol glossary is referenced as the substitution table.
    assert "symbol_glossary" in out


def test_audit_prompt_runs_six_checks(tmp_path, monkeypatch):
    """AUDIT must explicitly run coverage + glossing + faithfulness +
    anthropomorphism + name-drop + voice-first checks. Each check has a
    structured output field so the host can branch on it."""
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_audit(
        outline_yaml="paper_id: x\n",
        script="hi",
    )
    for needle in (
        "CHECK 1 — COVERAGE",
        "CHECK 2 — GLOSSING",
        "CHECK 3 — FAITHFULNESS",
        "CHECK 4 — ANTHROPOMORPHISM",
        "CHECK 5 — NAME-DROPPING",
        "CHECK 6 — VOICE-FIRST",
    ):
        assert needle in out, f"AUDIT prompt is missing {needle!r}"
    # Output schema additions for the new checks.
    for field in (
        "faithfulness_violations",
        "anthropomorphism_violations",
        "name_drop_violations",
    ):
        assert field in out, f"AUDIT prompt is missing the {field!r} output field"


def test_plan_prompt_requires_methodological_take(tmp_path, monkeypatch):
    """PLAN must require at least one methodological take and the
    `appears_in` field that forces takes across multiple segments."""
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_plan(
        arxiv_id="x", title="t", taste_profile="", outline_yaml="",
    )
    assert "METHODOLOGICAL" in out
    assert "appears_in" in out
    # naive_attempt is now a first-class suggested role.
    assert "naive_attempt" in out
