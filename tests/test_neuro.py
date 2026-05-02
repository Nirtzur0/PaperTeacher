"""Neuroscience domain pack — schema, prompt rendering, and identifier round-trip.

Network-touching paths (discovery, reader) aren't exercised here for the
same reason the ML pack tests don't hit arXiv: live HTTP makes the suite
flaky. The contract pieces — schema validates, parser surfaces ParseError,
prompts substitute every placeholder, encoded DOIs round-trip cleanly —
are what regress hardest and are what we pin.
"""
from __future__ import annotations

import importlib

import pytest


def _reload_with_home(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERTEACHER_HOME", str(tmp_path))
    monkeypatch.setenv("PAPERTEACHER_PROFILE", str(tmp_path / "profile.md"))
    import paperteacher.paths as paths_mod
    importlib.reload(paths_mod)
    import paperteacher.profile as profile_mod
    importlib.reload(profile_mod)
    import paperteacher.domains.neuro.prompts as prompts_mod
    importlib.reload(prompts_mod)
    return prompts_mod


VALID_OUTLINE_YAML = """\
paper_id: 10.1101_2024.05.01.591742
type: experimental
core_thesis: Place cells in CA1 split into two populations on reward shifts. One follows reward; the other stays anchored to the prior location.
gap_filled: Prior work assumed remapping was uniform across the population.
subjects:
  organism: C57BL/6J mice
  sample_size: n=8 mice, 412 cells
  brain_regions:
    - hippocampal area CA1
key_concepts:
  - id: C1
    name: place field remapping
    plain_english: cells switching which location they fire at
    why_it_matters: it's the main mechanism the field uses to explain spatial memory updates
    teaching_priority: critical
key_methods:
  - id: M1
    name: in-vivo two-photon calcium imaging
    what_it_measures: fluorescence as a proxy for spiking averaged over hundreds of milliseconds
    spatial_temporal_resolution: single cells, ~100ms
    typical_confounds:
      - GCaMP saturates at high firing rates
      - calcium signals lag spiking by tens of ms
    teaching_priority: critical
behavioral_tasks:
  - id: T1
    name: virtual linear track with movable reward
    what_subjects_did: ran for water reward at one of three locations on a treadmill in VR
    what_is_varied: which location dispenses water across blocks
    why_this_design: dissociates spatial location from reward identity
    teaching_priority: critical
key_findings:
  - id: F1
    name: roughly half of CA1 place cells follow reward; the other half stay anchored
    what_it_shows: place cells are not a homogeneous population
    effect_in_words: when reward moves, about half the cells move their preferred location with it; the rest keep firing at the old location even on probe trials with no reward
    concrete_picture: two clouds of points on a tuning-curve shift histogram, one centered on the reward shift, one on zero
    key_control: ruled out passive sensory tuning by running probe trials with no reward; the anchored cells still fired at the old location
    numerical_anchor: 412 cells across 8 mice; the bimodal split was visible in 7 of 8 animals
    bridge_to_next: this leaves open whether the two populations are stable subpopulations or trial-by-trial assignments
    teaching_priority: critical
results_to_highlight:
  - id: R1
    claim: the bimodal split was robust across animals
    what_it_demonstrates: the population-level finding is not driven by one outlier mouse
limitations_and_open_questions:
  - generalization to freely-moving rodents is untested
  - the cells are recorded over a single session
banned_glosses:
  - they recorded from neurons
  - they imaged the brain
acronyms_to_spell_out:
  CA1: cornu ammonis area 1
hard_pronunciations:
  hippocampus: hip-oh-CAM-pus
"""


def test_outline_round_trip():
    """Schema parses a complete neuro outline and round-trips through YAML."""
    from paperteacher.domains.neuro.models import Outline, parse_outline

    outline = parse_outline(VALID_OUTLINE_YAML)
    assert outline.paper_id == "10.1101_2024.05.01.591742"
    assert outline.subjects.organism == "C57BL/6J mice"
    # critical_ids spans methods, tasks, findings, and concepts.
    crit = outline.critical_ids()
    assert "C1" in crit and "M1" in crit and "T1" in crit and "F1" in crit

    redumped = outline.to_yaml()
    again = parse_outline(redumped)
    assert again.critical_ids() == outline.critical_ids()
    assert again.key_findings[0].key_control.startswith("ruled out passive sensory tuning")


def test_outline_missing_required_finding_field_raises():
    """`key_control` is optional in the schema (some findings genuinely
    lack one) but `name` and `teaching_priority` are required. Drop one
    and we should see a ParseError."""
    from paperteacher.domains._common import ParseError
    from paperteacher.domains.neuro.models import parse_outline

    bad = VALID_OUTLINE_YAML.replace("teaching_priority: critical",
                                     "teaching_priority: nonsense")
    with pytest.raises(ParseError) as excinfo:
        parse_outline(bad)
    assert excinfo.value.raw == bad


def test_outline_extra_fields_ignored():
    """Lenient: future LLM-emitted fields don't break validation."""
    from paperteacher.domains.neuro.models import parse_outline

    yaml_text = VALID_OUTLINE_YAML + "\nllm_invented_field: whatever\n"
    outline = parse_outline(yaml_text)
    assert outline.paper_id == "10.1101_2024.05.01.591742"


def test_outline_handles_paper_with_no_behavior():
    """Cell-biology / connectomics papers have no behavioral_tasks. The
    schema must accept an empty list without complaint."""
    from paperteacher.domains.neuro.models import parse_outline

    # Drop the task block from the YAML.
    no_task = VALID_OUTLINE_YAML.replace(
        """behavioral_tasks:
  - id: T1
    name: virtual linear track with movable reward
    what_subjects_did: ran for water reward at one of three locations on a treadmill in VR
    what_is_varied: which location dispenses water across blocks
    why_this_design: dissociates spatial location from reward identity
    teaching_priority: critical
""",
        "behavioral_tasks: []\n",
    )
    outline = parse_outline(no_task)
    assert outline.behavioral_tasks == []
    # T1 should now be absent from critical_ids since the task was dropped.
    assert "T1" not in outline.critical_ids()


def test_render_extract_substitutes_all(tmp_path, monkeypatch):
    """The extract prompt fills every placeholder with no leftover braces."""
    import re
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_extract(
        arxiv_id="10.1101_2024.05.01.591742",
        title="Place cells split into two functional populations",
        taste_profile="domain: neuro",
        paper_text="Two-photon calcium imaging in CA1 reveals...",
    )
    leftover = re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", out)
    assert not leftover, f"unfilled placeholders: {leftover}"
    assert "10.1101_2024.05.01.591742" in out
    assert "Place cells split" in out
    # Field-specific neuro vocabulary must reach the prompt.
    assert "control" in out.lower() and "method" in out.lower()


def test_render_teach_default_and_with_plan(tmp_path, monkeypatch):
    """Default structure is the prescriptive arc; passing a plan switches
    to the plan-driven structure block."""
    prompts = _reload_with_home(monkeypatch, tmp_path)
    default_out = prompts.render_teach(
        arxiv_id="10.1101_x",
        title="t",
        taste_profile="domain: neuro",
        paper_text="paper",
        outline_yaml="paper_id: 10.1101_x\n",
        mode="two_host",
    )
    assert "EPISODE PLAN" not in default_out
    # Default structure is an arc menu, not a 7-section template.
    assert "FINDING ARC" in default_out
    assert "CONTROL ARC" in default_out

    plan_out = prompts.render_teach(
        arxiv_id="10.1101_x",
        title="t",
        taste_profile="domain: neuro",
        paper_text="paper",
        outline_yaml="paper_id: 10.1101_x\n",
        mode="single_host",
        plan_yaml="arc:\n  - role: opening\n    purpose: hook\n",
    )
    assert "EPISODE PLAN" in plan_out
    assert "FOLLOW THE PLAN" in plan_out


def test_render_audit_substitutes_all(tmp_path, monkeypatch):
    import re
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_audit(
        outline_yaml="paper_id: 10.1101_x\nkey_findings: []\n",
        script="<Person1>hi</Person1>",
    )
    leftover = re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", out)
    assert not leftover
    # Audit prompt must specifically demand control coverage for findings.
    assert "key control" in out.lower()


def test_render_plan_substitutes_all(tmp_path, monkeypatch):
    import re
    prompts = _reload_with_home(monkeypatch, tmp_path)
    out = prompts.render_plan(
        arxiv_id="10.1101_x",
        title="t",
        taste_profile="domain: neuro",
        outline_yaml="paper_id: 10.1101_x\n",
    )
    leftover = re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", out)
    assert not leftover
    assert "10.1101_x" in out


def test_doi_id_round_trip():
    """DOIs encode `/` -> `_` for the framework, and the reader inverts
    that on the way out. Round-trip must be lossless for both DOI prefixes
    bioRxiv has used (10.1101 historical, 10.64898 since ~2026)."""
    from paperteacher.domains.neuro.discovery import _encode_doi
    from paperteacher.domains.neuro.reader import _decode_id

    for doi in ("10.1101/2024.05.01.591742", "10.64898/2026.04.23.720488"):
        encoded = _encode_doi(doi)
        assert "/" not in encoded                 # filesystem-safe
        assert _decode_id(encoded) == doi          # round-trip


def test_pack_registers_under_neuro(monkeypatch):
    """The bundled-loader picks up the neuro pack on first active_domain()."""
    monkeypatch.setenv("PAPERTEACHER_DOMAIN", "neuro")
    monkeypatch.delenv("PAPERTEACHER_PROFILE", raising=False)
    import paperteacher.domain as d
    d.reset_active()
    assert d.active_domain().name == "neuro"
    assert "neuro" in d.list_domains()


def test_neuro_pack_implements_protocol(monkeypatch):
    """Domain protocol contract: every callable the host needs is wired up."""
    monkeypatch.setenv("PAPERTEACHER_DOMAIN", "neuro")
    monkeypatch.delenv("PAPERTEACHER_PROFILE", raising=False)
    import paperteacher.domain as d
    d.reset_active()
    pack = d.active_domain()
    assert pack.OutlineModel is not None
    # Optional planner stage — neuro implements it.
    assert getattr(pack, "PlanModel", None) is not None
    assert callable(pack.parse_outline)
    assert callable(pack.parse_audit)
    assert callable(pack.render_extract)
    assert callable(pack.render_plan)
    assert callable(pack.render_teach)
    assert callable(pack.render_audit)
    assert callable(pack.discover)
    assert callable(pack.read)
