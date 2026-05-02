"""Regression tests for the prompt token diet — and for what we deliberately
left rich.

Contracts:
  1. plan does NOT inline paper_text. The outline carries the structural
     claims; saving ~30K tokens per plan call is worth it.
  2. plan DOES inline taste_profile. The listener voice anchors the takes —
     earlier diet had pulled it out and the script visibly collapsed.
  3. teach DOES inline taste_profile. Same reason — voice anchor.
  4. teach DOES inline the per-pack voice guide (the ml pack's
     `_VOICE_GUIDE` block with its pronunciation table). Resource-pointer
     mode was tried and reverted — Pro 2.5 doesn't auto-fetch resources.
  5. save_audit returns counts, not full per-item dumps.
  6. extract inlines profile + paper text — that stage is the contract for
     everything downstream, so no diet there.

These pin the contract across all four bundled packs so a future template
edit can't quietly drop the listener-voice context (the symptom that
produced "Welcome to Machine Learning Frontiers, I'm Alex" 4-min slop).
"""
from __future__ import annotations

import importlib

import pytest

PACKS = ("ml", "physics", "neuro", "econ")


def _prompts_module(name: str):
    return importlib.import_module(f"paperteacher.domains.{name}.prompts")


# ---- 1: plan does NOT inline paper_text -------------------------------


@pytest.mark.parametrize("pack", PACKS)
def test_plan_does_not_take_paper_text(pack):
    """render_plan signature should not accept paper_text — that param was
    a back-compat shim from the over-aggressive diet, now removed."""
    import inspect
    mod = _prompts_module(pack)
    params = inspect.signature(mod.render_plan).parameters
    assert "paper_text" not in params, (
        f"{pack}.render_plan still accepts paper_text — the param should be "
        f"removed entirely, not silently dropped"
    )


# ---- 2: plan DOES inline taste_profile --------------------------------


@pytest.mark.parametrize("pack", PACKS)
def test_plan_inlines_profile(pack):
    mod = _prompts_module(pack)
    rendered = mod.render_plan(
        arxiv_id="x",
        title="t",
        taste_profile="LISTENER-VOICE-ANCHOR-MARKER",
        outline_yaml="dummy: outline",
    )
    assert "LISTENER-VOICE-ANCHOR-MARKER" in rendered, (
        f"{pack}.render_plan dropped taste_profile — the listener voice "
        f"is what makes the takes specific instead of generic praise"
    )


# ---- 3: teach DOES inline taste_profile --------------------------------


@pytest.mark.parametrize("pack", PACKS)
def test_teach_inlines_profile(pack):
    mod = _prompts_module(pack)
    rendered = mod.render_teach(
        arxiv_id="x",
        title="t",
        taste_profile="LISTENER-VOICE-ANCHOR-MARKER",
        paper_text="paper goes here",
        outline_yaml="dummy",
    )
    assert "LISTENER-VOICE-ANCHOR-MARKER" in rendered, (
        f"{pack}.render_teach dropped taste_profile — last time we did this "
        f"the script collapsed into 'Welcome to Machine Learning Frontiers'"
    )


# ---- 4: ml teach inlines the voice guide ------------------------------


def test_ml_teach_inlines_voice_guide():
    """The pronunciation table / numerical-rewrite block / anti-anthropomorphism
    rules MUST be in every rendered teach prompt for ml. Resource-pointer
    mode was tried (saved ~1K tokens by referencing voice-guide://ml) and
    reverted — Pro 2.5 with thinking=low doesn't auto-fetch the resource and
    the constraint block silently disappeared from its context."""
    ml = _prompts_module("ml")
    rendered = ml.render_teach(
        arxiv_id="x",
        title="t",
        taste_profile="",
        paper_text="abc",
        outline_yaml="dummy",
    )
    # Distinctive lines from _VOICE_GUIDE.
    assert "PRONUNCIATION GUIDE" in rendered
    assert "ELBO" in rendered  # ML acronym phonetic table
    # Must NOT silently swap to a resource pointer.
    assert "voice-guide://" not in rendered


# ---- 5: save_audit response is decision-shaped, not a full dump -------


def test_save_audit_response_does_not_echo_full_items(paperteacher_home):
    """The host just sent the YAML — re-shipping every item via model_dump()
    in the response was 95% redundant. Audit YAML on disk has the full
    breakdown when the host actually wants it."""
    from paperteacher import server

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
    result = server.save_audit(arxiv_id="diet-test", audit_yaml=audit_yaml)
    assert result["ok"] is True
    assert result["recommendation"] == "regenerate_with_gaps"
    assert result["coverage_status"] == "partial"
    assert result["missing_count"] == 1
    assert result["glossed_count"] == 1
    assert "items_missing" not in result
    assert "items_glossed" not in result


# ---- 6: extract still inlines paper_text and profile ------------------


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


# ---- 7: extract requires a "≥3 critical" floor in the prompt body -----
#
# This is the Stage-1 fix for "outline came back with 0 criticals so the
# script collapsed into a generic abstract paraphrase." The prompt's RULES
# block has to TELL the model that the critical tier exists for a reason
# and refuse to let it punt by marking everything `mention`.


@pytest.mark.parametrize("pack", PACKS)
def test_extract_pins_minimum_critical_count(pack):
    mod = _prompts_module(pack)
    rendered = mod.render_extract(
        arxiv_id="x", title="t", taste_profile="", paper_text="",
    )
    assert "MINIMUM 3 ITEMS MARKED `critical`" in rendered, (
        f"{pack}.EXTRACT_OUTLINE no longer enforces a critical-count floor; "
        f"without this the model can punt every item to `mention` and the "
        f"teach stage loses its coverage contract"
    )
