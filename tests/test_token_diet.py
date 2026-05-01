"""Regression tests for the prompt token diet.

These pin the contract:
  1. The plan-stage prompt does NOT inline the listener profile.
  2. The plan-stage prompt does NOT inline the full paper text.
  3. The teach-stage prompt does NOT inline the listener profile (the host
     reads `profile://taste` once at stage 0).
  4. The teach-stage prompt CAN drop the voice guide for ML when called
     with `inline_voice_guide=False` — replaced by a one-line resource
     reference. CLI default keeps it inlined.
  5. `save_audit` returns counts, not the full per-item dump.
  6. `extract_outline` still inlines profile + paper text — the extractor
     IS the bottleneck of the pipeline, no diet here.

These pin the diet across all four bundled packs so a future template edit
can't quietly re-add the bloat in one of them.
"""
from __future__ import annotations

import importlib

import pytest

PACKS = ("ml", "physics", "neuro", "econ")


def _prompts_module(name: str):
    return importlib.import_module(f"paperteacher.domains.{name}.prompts")


# ---- 1 + 2: plan stage drops profile and paper_text ---------------------


@pytest.mark.parametrize("pack", PACKS)
def test_plan_does_not_inline_profile(pack):
    mod = _prompts_module(pack)
    rendered = mod.render_plan(
        arxiv_id="x",
        title="t",
        outline_yaml="dummy: outline",
        # Pass these to confirm they're back-compat-accepted but not inlined.
        taste_profile="SECRET-PROFILE-MARKER-PROFILE",
        paper_text="SECRET-PAPER-MARKER-PAPER",
    )
    assert "SECRET-PROFILE-MARKER" not in rendered, (
        f"{pack}.render_plan leaked the listener profile back into the prompt"
    )


@pytest.mark.parametrize("pack", PACKS)
def test_plan_does_not_inline_paper_text(pack):
    mod = _prompts_module(pack)
    rendered = mod.render_plan(
        arxiv_id="x",
        title="t",
        outline_yaml="dummy: outline",
        paper_text="SECRET-PAPER-MARKER-PAPER",
    )
    assert "SECRET-PAPER-MARKER" not in rendered, (
        f"{pack}.render_plan leaked paper_text back into the prompt — that "
        f"undoes the ~30K-token saving"
    )


# ---- 3: teach drops profile -------------------------------------------


@pytest.mark.parametrize("pack", PACKS)
def test_teach_does_not_inline_profile(pack):
    mod = _prompts_module(pack)
    rendered = mod.render_teach(
        arxiv_id="x",
        title="t",
        paper_text="paper goes here",
        outline_yaml="dummy",
        taste_profile="SECRET-PROFILE-MARKER",
    )
    assert "SECRET-PROFILE-MARKER" not in rendered, (
        f"{pack}.render_teach leaked the listener profile back into the prompt"
    )


# ---- 4: voice guide is droppable in the ML pack -----------------------


def test_ml_teach_inlines_voice_guide_by_default():
    """CLI path: no MCP host, voice guide must be inline."""
    ml = _prompts_module("ml")
    rendered = ml.render_teach(
        arxiv_id="x",
        title="t",
        paper_text="abc",
        outline_yaml="dummy",
    )
    # The pronunciation table is the most distinctive line in _VOICE_GUIDE.
    assert "PRONUNCIATION GUIDE" in rendered


def test_ml_teach_drops_voice_guide_when_resource_path():
    """MCP path: server passes inline_voice_guide=False; the teach prompt
    references `voice-guide://ml` instead of re-shipping the table."""
    ml = _prompts_module("ml")
    rendered = ml.render_teach(
        arxiv_id="x",
        title="t",
        paper_text="abc",
        outline_yaml="dummy",
        inline_voice_guide=False,
    )
    assert "PRONUNCIATION GUIDE" not in rendered
    assert "voice-guide://ml" in rendered


# ---- 5: save_audit response is decision-shaped, not a full dump -------


def test_save_audit_response_does_not_echo_full_items(paperteacher_home):
    """The host just sent the YAML — re-shipping every item via model_dump()
    in the response was 95% redundant."""
    from paperteacher import server  # imports MCP wiring; safe under tmp HOME

    audit_yaml = """\
coverage_status: partial
items_missing:
  - id: E1
    name: a long equation name
    what_was_said: a bunch of detail the host already has
    what_is_missing: a long explanation of what is missing
    severity: critical
items_glossed:
  - id: C2
    quote: "calls it 'just MSE' which the outline forbade"
    why_its_a_gloss: "MSE is in banned_glosses for this paper"
banned_phrases_used: []
voice_first_violations: []
overall_assessment: needs work
recommendation: regenerate_with_gaps
"""
    # FastMCP's @mcp.tool() registers the function but leaves it callable
    # directly — no `.fn` wrapper.
    result = server.save_audit(arxiv_id="diet-test", audit_yaml=audit_yaml)

    # Decision-shaped: counts only, not the full items.
    assert result["ok"] is True
    assert result["recommendation"] == "regenerate_with_gaps"
    assert result["coverage_status"] == "partial"
    assert result["missing_count"] == 1
    assert result["glossed_count"] == 1
    # The redundant per-item dumps must not be in the response.
    assert "items_missing" not in result
    assert "items_glossed" not in result


# ---- 6: extract is the one stage where the full diet doesn't apply ----


def test_extract_still_inlines_paper_text_and_profile():
    """The extractor IS the contract for stages 2 and 3 — undertraining it
    propagates everywhere. So this stage retains the full body."""
    ml = _prompts_module("ml")
    rendered = ml.render_extract(
        arxiv_id="x",
        title="t",
        taste_profile="EXTRACT-PROFILE-MARKER",
        paper_text="EXTRACT-PAPER-MARKER",
    )
    assert "EXTRACT-PROFILE-MARKER" in rendered
    assert "EXTRACT-PAPER-MARKER" in rendered
