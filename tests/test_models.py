"""Pydantic round-trip + ParseError surface."""
from __future__ import annotations

import pytest

from paperteacher.domains._common import AuditReport, ParseError, parse_audit
from paperteacher.domains.ml.models import Outline, parse_outline


VALID_OUTLINE_YAML = """\
paper_id: 2603.20105
type: theoretical
core_thesis: Two sentences here. Really two.
gap_filled: One thing was unjustified before.
key_concepts:
  - id: C1
    name: thing
    plain_english: a thing
    why_it_matters: matters because
    teaching_priority: critical
key_equations:
  - id: E1
    english_name: loss
    what_it_solves: pain
    structure_in_words: stuff
    components:
      - role: a role
        intuition: like a spring
        what_if_removed: it explodes
    key_trick: cancel
    geometric_picture: a sphere
    numerical_walkthrough: 1+1=2
    bridge_to_next: leads to E2
    teaching_priority: critical
results_to_highlight: []
"""


def test_outline_round_trip():
    outline = parse_outline(VALID_OUTLINE_YAML)
    assert outline.paper_id == "2603.20105"
    assert outline.critical_ids() == ["C1", "E1"]

    redumped = outline.to_yaml()
    again = parse_outline(redumped)
    assert again.critical_ids() == outline.critical_ids()
    assert again.key_equations[0].id == "E1"


def test_outline_coerces_unquoted_arxiv_id():
    """`coerce_numbers_to_str` lets the LLM emit `2603.20105` unquoted."""
    yaml_text = VALID_OUTLINE_YAML.replace("paper_id: 2603.20105", "paper_id: 2603.20105")
    # The above is a no-op, but the canonical YAML emitted from a float by
    # PyYAML is what the model usually writes. Validate the unquoted form
    # parses by constructing it directly:
    outline = Outline.model_validate({"paper_id": 2603.20105, "type": "theoretical",
                                      "core_thesis": "x.", "gap_filled": "y."})
    assert outline.paper_id == "2603.20105"


def test_outline_invalid_priority_raises_parse_error():
    bad = VALID_OUTLINE_YAML.replace("teaching_priority: critical",
                                     "teaching_priority: nonsense")
    with pytest.raises(ParseError) as excinfo:
        parse_outline(bad)
    assert "teaching_priority" in str(excinfo.value) or "nonsense" in str(excinfo.value)
    # The ParseError carries the raw input so the caller can re-prompt with it.
    assert excinfo.value.raw == bad


def test_outline_yaml_syntax_error_raises_parse_error():
    with pytest.raises(ParseError):
        parse_outline("paper_id: [unbalanced")


def test_outline_extra_fields_ignored():
    """Lenient: unknown LLM-emitted fields don't break validation."""
    yaml_text = VALID_OUTLINE_YAML + "\nllm_invented_field: whatever\n"
    outline = parse_outline(yaml_text)
    assert outline.paper_id == "2603.20105"


VALID_AUDIT_YAML = """\
coverage_status: complete
items_missing: []
items_glossed: []
banned_phrases_used: []
voice_first_violations: []
overall_assessment: Looks good. Ship it.
recommendation: ship
"""


def test_audit_round_trip_and_passed_helper():
    audit = parse_audit(VALID_AUDIT_YAML)
    assert audit.recommendation == "ship"
    assert audit.passed() is True

    bad = VALID_AUDIT_YAML.replace("recommendation: ship",
                                   "recommendation: regenerate_with_gaps")
    audit2 = parse_audit(bad)
    assert audit2.passed() is False


def test_audit_invalid_recommendation_raises_parse_error():
    bad = VALID_AUDIT_YAML.replace("recommendation: ship",
                                   "recommendation: maybe")
    with pytest.raises(ParseError):
        parse_audit(bad)


def test_outline_critical_vs_important():
    outline = parse_outline(VALID_OUTLINE_YAML)
    # Add an "important" concept and check filtering.
    extra = outline.model_dump()
    extra["key_concepts"].append({
        "id": "C2", "name": "x", "plain_english": "x",
        "why_it_matters": "x", "teaching_priority": "important",
    })
    outline2 = Outline.model_validate(extra)
    assert outline2.critical_ids() == ["C1", "E1"]
    assert outline2.important_ids() == ["C2"]


# ---- v0.3 outline extensions: pedagogy + provenance --------------------
#
# These were added as part of the world-class upgrade. All fields are
# OPTIONAL so older outlines (pre-v0.3) still parse — the LLM may not
# populate them on every paper. The schema's job here is just to give the
# fields a typed home when they ARE present.


def test_outline_accepts_extension_fields():
    """Stake claim, prior attempts, ablations, assumptions, benchmark
    caveats, compute envelope, common misreadings, symbol glossary — all
    optional, all parse cleanly when the LLM emits them."""
    yaml_text = VALID_OUTLINE_YAML + """
stake_claim: If this is right, attention isn't quadratic in any meaningful sense.
prior_attempts:
  - name: vanilla softmax attention
    what_failed: O(n^2) memory blew up past 8k context
ablations:
  - component_removed: the gating
    metric_delta: drops 4.2 points on GLUE
    implies: the gating, not the depth, carries the gain
assumption_boundaries:
  - assumption: tokens are i.i.d. given the prefix
    where_it_breaks: heavy-tailed code or repeated boilerplate
benchmark_caveats:
  GLUE: heavily contaminated post-2022; tells you about memorization more than reasoning
compute_envelope: 64 H100s, 3 weeks, ~$120k
common_misreadings:
  - readers conflate "linear attention" with "as good as softmax"; the paper is explicit it isn't
symbol_glossary:
  pi_theta: the policy parameterized by theta
  log p(x|y): the log-likelihood of x given y
"""
    outline = parse_outline(yaml_text)
    assert outline.stake_claim.startswith("If this is right")
    assert outline.prior_attempts[0].what_failed.startswith("O(n^2)")
    assert outline.ablations[0].implies.startswith("the gating")
    assert outline.assumption_boundaries[0].where_it_breaks.startswith("heavy-tailed")
    assert "GLUE" in outline.benchmark_caveats
    assert "$120k" in outline.compute_envelope
    assert "linear attention" in outline.common_misreadings[0]
    assert outline.symbol_glossary["pi_theta"] == "the policy parameterized by theta"


def test_outline_extension_fields_are_optional():
    """The minimal pre-v0.3 outline still parses — extensions default to empty."""
    outline = parse_outline(VALID_OUTLINE_YAML)
    assert outline.stake_claim == ""
    assert outline.prior_attempts == []
    assert outline.ablations == []
    assert outline.assumption_boundaries == []
    assert outline.benchmark_caveats == {}
    assert outline.compute_envelope == ""
    assert outline.common_misreadings == []
    assert outline.symbol_glossary == {}
    # Concept's first_concrete_instance also defaults empty.
    assert outline.key_concepts[0].first_concrete_instance == ""


def test_concept_first_concrete_instance():
    yaml_text = VALID_OUTLINE_YAML.replace(
        "    why_it_matters: matters because",
        "    why_it_matters: matters because\n"
        "    first_concrete_instance: imagine a 3D point [1,0,0] and ask where the score points",
    )
    outline = parse_outline(yaml_text)
    assert "imagine a 3D point" in outline.key_concepts[0].first_concrete_instance


def test_result_provenance_fields():
    yaml_text = VALID_OUTLINE_YAML.replace(
        "results_to_highlight: []",
        """results_to_highlight:
  - id: R1
    claim: 87.3 on GLUE
    what_it_demonstrates: the gating mechanism transfers
    benchmark: GLUE
    baseline: prior best 84.1; obvious baseline 70.0
    variance_note: 3 seeds, std 0.4
    compute: 8x A100 for 12 hours
""",
    )
    outline = parse_outline(yaml_text)
    r = outline.results_to_highlight[0]
    assert r.benchmark == "GLUE"
    assert "84.1" in r.baseline
    assert "std 0.4" in r.variance_note
    assert "A100" in r.compute
