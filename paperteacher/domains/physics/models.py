"""Pydantic models for the physics domain's stage-1 outline.

Why this schema differs from the ml pack's:

- Physics equations carry content that a pure ML loss does not: dimensional
  balance (units must match on both sides), symmetries (Lorentz, gauge,
  parity, time-reversal — what's invariant under which transformation),
  limiting cases (does it reduce to Newton in the low-velocity limit? to
  classical mechanics as ℏ→0?), and conservation laws (Noether: every
  continuous symmetry implies a conserved current). Skipping any of these
  is the standard failure mode of "popularized" physics. The schema makes
  them mandatory fields, not optional notes.

- Physics papers predict observables — quantities you can measure with a
  detector or telescope, with quoted uncertainty. Cross-sections, branching
  ratios, spectral lines, decay rates, anomaly coefficients, transition
  temperatures. ML's `results_to_highlight` is mostly benchmark numbers; the
  physics analogue is `observables_and_predictions` plus a falsifiability
  note ("what would rule this out?"). Different field; different prompt.

- Many physics papers describe an apparatus or observation campaign (ATLAS
  Run 3, LIGO/Virgo merger event, JWST NIRSpec spectrum, ultracold-atom
  trap). The script needs to ground the theory in what was actually
  measured, including the dominant systematic. ML papers don't usually
  carry this layer.

- Regime and assumptions matter and are easy to gloss: "low-energy
  effective theory valid up to Λ", "weak coupling g ≪ 1", "non-relativistic
  v ≪ c", "T < T_c". Listing them up front in the outline forces the
  teaching pass to name them rather than hand-wave with "in the appropriate
  limit".

- Historical context belongs to physics in a way it doesn't to ML. A 2026
  ML paper rarely needs to invoke Hebb or Rosenblatt; a 2026 hep-th paper
  routinely sits inside a 60-year story (gauge theory, the Standard Model,
  string theory). One field, one sentence per thread, no big essay.

Stage-3 audit + ParseError + the LenientModel base live in
`domains/_common.py` since they're cross-domain.
"""
from __future__ import annotations

from typing import Literal

import yaml
from pydantic import Field, ValidationError

from .._common import LenientModel, ParseError, Priority

# Physics paper types: theoretical (pure-math derivation), phenomenological
# (predicts measurable quantities for an existing experiment), experimental
# (lab apparatus result), observational (sky survey / telescope / detector),
# computational (lattice / N-body / fluid sim), review (taxonomy walk).
PaperType = Literal[
    "theoretical",
    "phenomenological",
    "experimental",
    "observational",
    "computational",
    "review",
]


class EquationComponent(LenientModel):
    """One term/symbol in an equation, described BY ROLE not symbol.

    `intuition` should be a physical picture (a worldline, a field
    configuration, a flow). `what_if_removed` answers "what physics breaks
    if this term is dropped?" — often the cleanest way to explain why a
    term is there at all.
    """

    role: str = ""
    intuition: str = ""
    what_if_removed: str = ""


class Equation(LenientModel):
    """A physics equation, decomposed for voice-first teaching.

    Beyond the ML schema: every critical physics equation must clear four
    sanity gates the script will read aloud — dimensional, symmetry,
    conservation, and limiting-case. These are the moves Feynman taught and
    that get skipped first under narrative pressure. Carrying them in the
    schema means the teaching prompt can demand each one.
    """

    id: str
    english_name: str = ""
    what_it_solves: str = ""
    structure_in_words: str = ""
    components: list[EquationComponent] = Field(default_factory=list)

    # --- physics-specific sanity gates ----------------------------------
    dimensional_check: str = ""
    """Units balance, expressed in plain words. e.g. "left side is energy
    per unit volume; right side is field-tensor squared, which carries the
    same dimensions in natural units once you reinstate ℏc". Empty allowed
    for `mention` items only — `critical` items must fill this in."""

    symmetries: list[str] = Field(default_factory=list)
    """Invariances. e.g. ["Lorentz invariant", "gauge symmetry under U(1)",
    "parity-odd"]. State the symmetry type, not the technical proof."""

    conservation_law: str = ""
    """What's conserved (Noether). e.g. "the time-translation symmetry of
    the Lagrangian gives energy conservation; the global U(1) phase
    rotation gives charge conservation". Empty if not applicable."""

    limiting_case: str = ""
    """A regime where the equation reduces to something already understood.
    e.g. "in the v/c → 0 limit, the kinetic term recovers (1/2)mv² —
    Newton's mechanics drops out". Mandatory for `critical` items."""

    key_trick: str = ""
    geometric_picture: str = ""
    fermi_estimate: str = ""
    """A first-principles order-of-magnitude estimate. e.g. "for a
    star-mass black hole, plug in M ≈ 10³⁰ kg and you get a Schwarzschild
    radius around 3 km — within a factor of two of the value people
    quote." Replaces ML's `numerical_walkthrough`; the spirit is the same
    but Fermi-style estimation is the physics tradition."""

    bridge_to_next: str = ""
    teaching_priority: Priority = "mention"
    note: str | None = None


class Concept(LenientModel):
    id: str
    name: str = ""
    plain_english: str = ""
    why_it_matters: str = ""
    teaching_priority: Priority = "mention"
    historical_thread: str = ""
    """Optional one-liner placing the concept in tradition. e.g. "this is
    the modern descendant of Lorentz's local-time idea, sharpened by
    Minkowski into geometry". Skip for genuinely new concepts."""


class Observable(LenientModel):
    """A quantity the paper predicts (or measures) that an experiment can
    confront. The script grounds the math by naming what falls out of it
    in measurable form.
    """

    id: str
    name: str = ""
    """e.g. "branching ratio of B → K μ⁺μ⁻", "shift in the CMB TT power
    spectrum near ℓ ≈ 200", "transition temperature of the superconducting
    phase"."""

    predicted_value: str = ""
    """Numerical value with uncertainty if quoted. Free-form string so
    units, exponent notation, and asymmetric error bars all fit. e.g.
    "(1.2 ± 0.3) × 10⁻⁷" or "T_c ≈ 90 K"."""

    how_measured: str = ""
    """Which apparatus / dataset would reach this. e.g. "LHCb full Run 3
    luminosity", "Planck 2018 low-ℓ likelihood", "muon-spin-rotation on
    YBCO crystals at 4 K"."""

    falsifiability: str = ""
    """One sentence: what value or non-detection would rule the prediction
    out? Forces the script to name the failure mode, not just the success
    case."""


class ExperimentalSetup(LenientModel):
    """For experimental / observational papers: the apparatus or
    observation campaign that produced the result. Theoretical papers
    leave this list empty.
    """

    id: str
    apparatus: str = ""
    """e.g. "ATLAS detector, Run 3 (139 fb⁻¹)", "JWST NIRSpec G395M",
    "LIGO Hanford + Livingston, O4 run"."""

    what_is_measured: str = ""
    key_systematic: str = ""
    """Dominant source of uncertainty. e.g. "luminosity calibration
    (2.4%)", "instrument PSF wavelength dependence", "calibration of the
    arms' length to picometre level"."""


class Result(LenientModel):
    """A claim the paper makes that isn't itself an observable — proofs,
    no-go theorems, scaling laws, a derived bound. Observables go in
    `observables_and_predictions`; this is for the "we showed that..."
    style of result.
    """

    id: str
    claim: str = ""
    what_it_demonstrates: str = ""
    why_surprising: str | None = None


class Outline(LenientModel):
    paper_id: str
    type: PaperType = "theoretical"
    core_thesis: str = ""
    gap_filled: str = ""

    historical_context: list[str] = Field(default_factory=list)
    """1-3 short threads placing the work in tradition. Optional but
    recommended for hep-th, gr-qc, foundations work. Skip for routine
    cond-mat measurements where the lineage isn't load-bearing."""

    regime_and_assumptions: list[str] = Field(default_factory=list)
    """The validity envelope. e.g. ["non-relativistic v ≪ c", "weak
    coupling g ≪ 1", "Markov approximation on the bath"]. Naming these
    up front makes the script say "valid in the X limit" rather than "in
    the appropriate limit"."""

    key_concepts: list[Concept] = Field(default_factory=list)
    key_equations: list[Equation] = Field(default_factory=list)
    observables_and_predictions: list[Observable] = Field(default_factory=list)
    experimental_setup: list[ExperimentalSetup] = Field(default_factory=list)
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

    # The framework's storage layer surfaces these counts in the event log
    # if they exist (see storage.save_outline). Keeping the same names as
    # the ml pack means the dashboards stay consistent.
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
# Same shape as the ml pack's planner — segments, takes, sits-alongside,
# why-now. The pack carries its own copy rather than importing across peer
# packs so it can evolve independently (e.g. add a `dimensional_audit`
# segment-level field later) without coupling physics to ml's release.


class Segment(LenientModel):
    id: str
    role: str
    """Free-form. Suggested physics-shaped roles (the prompt expands on
    these, the schema doesn't constrain them):
        opening | motivation | tradition | regime_setup | derivation
        | dimensional_check | limiting_case | symmetry_argument
        | prediction | experimental_status | comparison | critique
        | closer
    """
    covers: list[str] = Field(default_factory=list)
    callbacks: list[str] = Field(default_factory=list)
    purpose: str


class Take(LenientModel):
    """A committed opinion the professor holds across the whole episode.
    Same role as in the ml pack's planner — drives critique segments and
    asides so the persona has a coherent stance instead of improvising.
    Physics has its own flavour: a take might be "the authors are
    underselling the regime of validity — this only works for v ≪ c, and
    the solar-system tests they cite already lived there".
    """

    claim: str
    evidence: str


class EpisodePlan(LenientModel):
    paper_id: str
    arc: list[Segment]
    takes: list[Take] = Field(default_factory=list)
    sits_alongside: list[str] = Field(default_factory=list)
    """1-3 adjacent works or paradigms. For physics this often crosses
    decades — Einstein's field equations, BCS, the Standard Model, the
    Bekenstein-Hawking entropy — not just the last 18 months of arXiv."""

    why_now: str = ""
    """What changed in apparatus, computing, or theory that made this
    paper land in 2026 specifically. e.g. "the JWST early-release data is
    finally precise enough at z>10 to test models people had on the shelf
    for a decade"."""

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
