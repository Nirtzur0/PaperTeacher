"""Pydantic round-trip + ParseError surface."""
from __future__ import annotations

import pytest

from paperteacher.models import (
    AuditReport,
    Outline,
    ParseError,
    parse_audit,
    parse_outline,
)


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
