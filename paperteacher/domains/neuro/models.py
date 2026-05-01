"""Pydantic models for the Neuroscience domain's stage-1 outline.

The schema's primary unit is the **Finding** (mirroring ML's Equation): the
central effect the paper is built around, decomposed so a teacher can walk a
listener through what was measured, in what subjects doing what, and what
pattern fell out — with the control or alternative explanation that was ruled
out always called out by name.

Stage-3 audit + ParseError + the LenientModel base live in
`domains/_common.py` since they're cross-domain.
"""
from __future__ import annotations

from typing import Literal

import yaml
from pydantic import Field, ValidationError

from .._common import LenientModel, ParseError, Priority

PaperType = Literal[
    "experimental",   # primary data — recordings, imaging, behavior, lesion, ...
    "modeling",       # computational / theoretical — circuit models, normative theories
    "clinical",       # human patient or translational study
    "review",
    "position",
]


class Method(LenientModel):
    """A recording / imaging / perturbation / analysis technique. The unit a
    listener has to understand to evaluate the paper's claim — `calcium
    imaging in CA1` is not interchangeable with `extracellular tetrode
    recordings in CA1`, and the limits of each shape what the data can show."""
    id: str
    name: str                                       # plain-English name (not just an acronym)
    what_it_measures: str                           # the actual physical signal: fluorescence, BOLD, spike times, ...
    spatial_temporal_resolution: str = ""           # "single-cell, ~100ms" / "voxel ~2mm, ~1s" / ...
    typical_confounds: list[str] = Field(default_factory=list)
    teaching_priority: Priority


class BehaviorTask(LenientModel):
    """What the subject was doing. Often the entire claim hinges on what the
    task isolated — a memory paper using a delayed-match task is making a
    different claim from one using a free-recall task. Empty for cell-biology
    or anatomy papers without a behavioral component."""
    id: str
    name: str
    what_subjects_did: str
    what_is_varied: str = ""                        # the parametric or categorical variable the analysis hangs on
    why_this_design: str = ""                       # what the design lets the authors conclude (or rule out)
    teaching_priority: Priority


class Subjects(LenientModel):
    """Organism + region + N. Critical context: an effect in two macaques is
    not an effect in 200 mice, and a region label without coordinates can
    mask what's actually being recorded from."""
    organism: str = ""                              # "C57BL/6J mice" / "two adult rhesus macaques" / "n=24 humans, age 18-35" / ""
    sample_size: str = ""                           # free-form: "n=8 mice, 412 cells" / "n=24 humans"
    brain_regions: list[str] = Field(default_factory=list)   # spelled out: "primary visual cortex (V1)", "dorsolateral prefrontal cortex (dlPFC)"


class Finding(LenientModel):
    """The central effect. Mirrors the ML pack's Equation in role: this is
    the thing every listener should be able to retell after the episode."""
    id: str
    name: str                                       # plain-English name, not "Figure 3"
    what_it_shows: str                              # one sentence: the effect
    effect_in_words: str                            # the magnitude/direction described as a picture, no p-values
    concrete_picture: str = ""                      # a literal mental image — a tuning curve, a raster, a manifold
    key_control: str = ""                           # the control or alternative explanation ruled out and HOW
    numerical_anchor: str = ""                      # one concrete number: "the response was three times larger" / "n=8 mice"
    bridge_to_next: str = ""
    teaching_priority: Priority
    note: str | None = None                         # honesty escape hatch — "I don't have a clean intuition for this"


class Concept(LenientModel):
    """A new or load-bearing idea the paper depends on (e.g. predictive coding,
    place fields, replay). Same shape as ML — kept identical so the planner
    can treat it generically."""
    id: str
    name: str
    plain_english: str
    why_it_matters: str
    teaching_priority: Priority


class Result(LenientModel):
    """A statistical or replication-level result — not the central finding,
    but the supporting numbers (n, effect size, replicated in N labs, ...)."""
    id: str
    claim: str
    what_it_demonstrates: str
    why_surprising: str | None = None


class Outline(LenientModel):
    paper_id: str
    type: PaperType
    core_thesis: str
    gap_filled: str
    subjects: Subjects = Field(default_factory=Subjects)
    key_concepts: list[Concept] = Field(default_factory=list)
    key_methods: list[Method] = Field(default_factory=list)
    behavioral_tasks: list[BehaviorTask] = Field(default_factory=list)
    key_findings: list[Finding] = Field(default_factory=list)
    results_to_highlight: list[Result] = Field(default_factory=list)
    limitations_and_open_questions: list[str] = Field(default_factory=list)
    banned_glosses: list[str] = Field(default_factory=list)
    acronyms_to_spell_out: dict[str, str] = Field(default_factory=dict)
    hard_pronunciations: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, text: str) -> "Outline":
        return cls.model_validate(yaml.safe_load(text))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False)

    def critical_ids(self) -> list[str]:
        out: list[str] = []
        out.extend(c.id for c in self.key_concepts if c.teaching_priority == "critical")
        out.extend(m.id for m in self.key_methods if m.teaching_priority == "critical")
        out.extend(t.id for t in self.behavioral_tasks if t.teaching_priority == "critical")
        out.extend(f.id for f in self.key_findings if f.teaching_priority == "critical")
        return out

    def important_ids(self) -> list[str]:
        out: list[str] = []
        out.extend(c.id for c in self.key_concepts if c.teaching_priority == "important")
        out.extend(m.id for m in self.key_methods if m.teaching_priority == "important")
        out.extend(t.id for t in self.behavioral_tasks if t.teaching_priority == "important")
        out.extend(f.id for f in self.key_findings if f.teaching_priority == "important")
        return out


def parse_outline(text: str) -> Outline:
    try:
        return Outline.from_yaml(text)
    except ValidationError as e:
        raise ParseError(f"outline schema validation failed:\n{e}", raw=text, validation=e) from e
    except yaml.YAMLError as e:
        raise ParseError(f"outline YAML parse failed: {e}", raw=text) from e


# ---- EpisodePlan (stage 1.5) --------------------------------------------
#
# Same contract as the ML pack: macro structure + committed takes + adjacent
# work + why_now. Segments reference outline ids — for neuro those ids span
# methods, tasks, findings, and concepts (anything in `critical_ids()`).


class Segment(LenientModel):
    id: str                                                 # seg_01, seg_02, ...
    role: str                                               # free-form (opening / setup / method / task / finding / control / critique / closing)
    covers: list[str] = Field(default_factory=list)         # outline ids (M*, T*, F*, C*)
    callbacks: list[str] = Field(default_factory=list)
    purpose: str


class Take(LenientModel):
    """A committed opinion the professor holds across the whole episode.

    For neuro papers, takes often live in the methods–claim gap: the data shows
    X, but does the technique actually measure what they think it measures?
    Whose work is this rebutting? Are the controls really controls? Compute
    these once from the full paper so the persona has a coherent stance.
    """

    claim: str
    evidence: str


class EpisodePlan(LenientModel):
    paper_id: str
    arc: list[Segment]
    takes: list[Take] = Field(default_factory=list)
    sits_alongside: list[str] = Field(default_factory=list)
    why_now: str = ""

    @classmethod
    def from_yaml(cls, text: str) -> "EpisodePlan":
        return cls.model_validate(yaml.safe_load(text))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False)

    def covered_ids(self) -> set[str]:
        return {oid for seg in self.arc for oid in seg.covers}


def parse_plan(text: str) -> EpisodePlan:
    try:
        return EpisodePlan.from_yaml(text)
    except ValidationError as e:
        raise ParseError(f"plan schema validation failed:\n{e}", raw=text, validation=e) from e
    except yaml.YAMLError as e:
        raise ParseError(f"plan YAML parse failed: {e}", raw=text) from e
