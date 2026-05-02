"""Pydantic models for the ML domain's stage-1 outline.

Stage-3 audit + ParseError + the LenientModel base live in `domains/_common.py`
since they're cross-domain.
"""
from __future__ import annotations

from typing import Literal

import yaml
from pydantic import Field, ValidationError

from .._common import LenientModel, ParseError, Priority

PaperType = Literal["theoretical", "empirical", "position", "survey"]


class EquationComponent(LenientModel):
    role: str = ""
    intuition: str = ""
    what_if_removed: str = ""


class Equation(LenientModel):
    id: str
    english_name: str = ""
    what_it_solves: str = ""
    structure_in_words: str = ""
    components: list[EquationComponent] = Field(default_factory=list)
    key_trick: str = ""
    geometric_picture: str = ""
    numerical_walkthrough: str = ""
    bridge_to_next: str = ""
    teaching_priority: Priority = "mention"
    note: str | None = None


class Concept(LenientModel):
    id: str
    name: str = ""
    plain_english: str = ""
    why_it_matters: str = ""
    first_concrete_instance: str = ""
    teaching_priority: Priority = "mention"


class Result(LenientModel):
    id: str
    claim: str = ""
    what_it_demonstrates: str = ""
    why_surprising: str | None = None
    # Provenance fields. The reproducibility literature documents accuracy
    # variance up to 90% across identical runs with different seeds — every
    # reported number should arrive with baseline + variance + compute context
    # so the script can never quote a SOTA number unmoored from what it cost
    # or what it beat. All optional: not every result has a number behind it.
    benchmark: str = ""
    baseline: str = ""
    variance_note: str = ""
    compute: str = ""


class PriorAttempt(LenientModel):
    """What was tried before this paper's contribution and why it failed.

    The "naive attempt that fails" is the load-bearing pedagogical move from
    Karpathy-style explanation: motivate the contribution by letting the
    listener feel why the simpler version doesn't work. Without this, the
    paper's contribution arrives as arbitrary cleverness instead of a
    response to a specific failure mode.
    """

    name: str = ""          # "softmax attention", "vanilla policy gradient", ...
    what_failed: str = ""   # the specific failure mode this paper addresses


class Ablation(LenientModel):
    """Structured ablation evidence.

    Most ML papers' actual story is in the ablations rather than the headline
    number. Capturing them as {component_removed → metric_delta → implication}
    lets the script say "this component is doing X; we know because removing
    it costs Y" instead of vague "various ablations confirm".
    """

    component_removed: str = ""
    metric_delta: str = ""    # "drops from 87.3 to 82.1 on GLUE"
    implies: str = ""         # "the gating, not the depth, carries the gain"


class Assumption(LenientModel):
    """A theoretical assumption + where it would break in practice.

    Markov, i.i.d., Lipschitz, full-rank, bounded reward, ... these are the
    silent contracts under which the theory holds. Naming them and where
    they break is what separates an honest explanation from a marketing one.
    """

    assumption: str = ""
    where_it_breaks: str = ""


class Outline(LenientModel):
    paper_id: str
    type: PaperType = "empirical"
    core_thesis: str = ""
    gap_filled: str = ""
    # One sentence committing to a stake: "if this paper is right, X
    # changes." Forces the model out of summary-mode into having a position
    # the script can defend or push against. Empty by default for older outlines.
    stake_claim: str = ""

    key_concepts: list[Concept] = Field(default_factory=list)
    key_equations: list[Equation] = Field(default_factory=list)
    results_to_highlight: list[Result] = Field(default_factory=list)

    # Pedagogical scaffolding — all optional, but each one is a known
    # failure mode for ML explainers when missing.
    prior_attempts: list[PriorAttempt] = Field(default_factory=list)
    ablations: list[Ablation] = Field(default_factory=list)
    assumption_boundaries: list[Assumption] = Field(default_factory=list)
    benchmark_caveats: dict[str, str] = Field(default_factory=dict)
    compute_envelope: str = ""
    common_misreadings: list[str] = Field(default_factory=list)

    limitations_and_open_questions: list[str] = Field(default_factory=list)
    banned_glosses: list[str] = Field(default_factory=list)
    acronyms_to_spell_out: dict[str, str] = Field(default_factory=dict)
    hard_pronunciations: dict[str, str] = Field(default_factory=dict)
    # Pre-computed substitution table for every symbol that appears in the
    # paper. The TEACH stage reads from this to enforce role-not-symbol
    # speech: `pi_theta` → "the policy", `log p(x|y)` → "the log-likelihood
    # of x given y". The single most reliable way to prevent the script from
    # reading symbols aloud is to hand it an explicit lookup table.
    symbol_glossary: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, text: str) -> "Outline":
        return cls.model_validate(yaml.safe_load(text))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False)

    def critical_ids(self) -> list[str]:
        out: list[str] = []
        out.extend(c.id for c in self.key_concepts if c.teaching_priority == "critical")
        out.extend(e.id for e in self.key_equations if e.teaching_priority == "critical")
        return out

    def important_ids(self) -> list[str]:
        out: list[str] = []
        out.extend(c.id for c in self.key_concepts if c.teaching_priority == "important")
        out.extend(e.id for e in self.key_equations if e.teaching_priority == "important")
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
# The plan owns the *macro* shape of the episode — the realizer (stage 2)
# generates each segment with the plan in scope so it knows where it sits in
# the arc, what's already been introduced, and what stance the professor has
# committed to. References outline items by id (no duplication of content).
#
# The schema deliberately leaves a lot loose: `role` is free-form so the
# planner can invent paper-shaped roles (worked_example, historical_aside,
# comparison, vibes_check, ...), and there's no `bridge_to_next` field — the
# realizer figures out transitions from neighbouring `purpose` lines so each
# episode sounds different rather than templated.


class Segment(LenientModel):
    id: str                                                 # seg_01, seg_02, ...
    role: str = ""
    covers: list[str] = Field(default_factory=list)
    callbacks: list[str] = Field(default_factory=list)
    purpose: str = ""


class Take(LenientModel):
    """A committed opinion the professor holds across the whole episode.

    Computed once from the full paper so the persona has a coherent stance
    instead of improvising hot takes per segment. The realizer pulls from these
    when the segment role is `critique` or when context calls for an aside.
    """

    claim: str = ""
    evidence: str = ""


class EpisodePlan(LenientModel):
    paper_id: str
    arc: list[Segment] = Field(default_factory=list)
    takes: list[Take] = Field(default_factory=list)
    sits_alongside: list[str] = Field(default_factory=list)  # 1-3 adjacent works/lines of work
    why_now: str = ""                                        # what makes this paper land today

    @classmethod
    def from_yaml(cls, text: str) -> "EpisodePlan":
        return cls.model_validate(yaml.safe_load(text))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False)

    def covered_ids(self) -> set[str]:
        """All outline ids the plan claims to cover. Useful for cross-checking
        against `Outline.critical_ids()` to catch a plan that drops a critical
        equation/concept on the floor."""
        return {oid for seg in self.arc for oid in seg.covers}


def parse_plan(text: str) -> EpisodePlan:
    try:
        return EpisodePlan.from_yaml(text)
    except ValidationError as e:
        raise ParseError(f"plan schema validation failed:\n{e}", raw=text, validation=e) from e
    except yaml.YAMLError as e:
        raise ParseError(f"plan YAML parse failed: {e}", raw=text) from e
