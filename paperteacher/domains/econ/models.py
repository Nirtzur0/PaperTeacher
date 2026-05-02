"""Pydantic models for the econ domain's stage-1 outline.

The schema deliberately diverges from the ML pack. ML papers are organized
around equations and components; modern econ papers are organized around
*identification* — how a regression coefficient maps to a counterfactual —
plus the specifications, estimates, and robustness checks that defend it.
Asset-pricing papers are different again: pricing kernels, factor-model
nests, alphas. So this outline carries five genres and conditional fields
per genre rather than forcing everything through a single shape.

Cross-domain bits (LenientModel, Priority, ParseError, AuditReport) live
in `domains/_common.py`.
"""
from __future__ import annotations

from typing import Literal

import yaml
from pydantic import Field, ValidationError

from .._common import LenientModel, ParseError, Priority

# ---- enums ---------------------------------------------------------------

PaperGenre = Literal[
    "empirical_causal",  # credibility-revolution applied micro / macro
    "structural",        # writes a model, estimates structurally, runs counterfactuals
    "asset_pricing",     # SDFs, factor models, anomalies (q-fin)
    "pure_theory",       # proves a proposition, no data
    "survey",            # synthesizes a literature
]

# Standard identification taxonomy. `none` is a real value — survey/theory
# papers don't have one, and forcing the LLM to pick fake strategies harms
# stage-3 audit signal. `structural` here means "identification comes from
# the structural model's restrictions," distinct from quasi-experimental
# strategies.
IDStrategy = Literal[
    "RCT", "IV", "DiD", "RDD", "event_study", "synthetic_control",
    "shift_share", "matching", "structural", "none",
]

# What units the headline number lives in. Forces the realizer to translate
# every coefficient into something the listener can hear without staring
# at a regression table.
EstimateUnit = Literal[
    "bps",                # basis points
    "log_points",         # ~percent change for small moves
    "sd_of_X",            # one-sd-of-X effect on Y
    "percent_of_mean_Y",  # effect as % of baseline outcome
    "elasticity",         # d log Y / d log X
    "level",              # raw units (dollars, jobs, etc.)
    "annualized_pct",     # for asset-pricing alphas / Sharpes
]

RobustnessType = Literal[
    "alt_specification",   # different controls / functional form / FE set
    "alt_sample",          # drop a subgroup; balanced panel; pre/post-crisis
    "placebo",             # design on population/period that shouldn't move
    "alt_instrument",      # different IV / different running variable / bandwidth
    "alt_inference",       # different cluster level, wild bootstrap, RI
    "heterogeneity",       # subgroup splits (often graduates to a contribution)
]


# ---- entities ------------------------------------------------------------


class Identification(LenientModel):
    """The identification strategy. Required for empirical_causal papers,
    omitted for theory/survey. The audit checks `key_assumption` and
    `what_breaks_if_violated` were named out loud in the script — the
    single most common gloss in econ pop-sci.
    """

    strategy: IDStrategy = "none"
    source_of_variation: str = ""
    key_assumption: str = ""
    assumption_defense: str = ""
    what_breaks_if_violated: str = ""
    teaching_priority: Priority = "mention"


class Specification(LenientModel):
    """A regression / estimating equation actually run in the paper.

    `voice_description` is the contract: the realizer reads THIS, not the
    LaTeX. Format: "We're regressing <outcome> on <regressor>, sweeping out
    <fixed effects>, with errors clustered at <level>. Beta is identified
    off <source of variation>."
    """

    id: str                           # SP1, SP2, ...
    purpose: str = ""
    outcome: str = ""
    treatment_or_regressor: str = ""
    controls: str = ""
    fixed_effects: str = ""
    cluster_level: str = ""
    sample: str = ""
    voice_description: str = ""
    teaching_priority: Priority = "mention"


class StructuralEquation(LenientModel):
    """A single equation from the structural model — Euler equation,
    no-arbitrage condition, market-clearing condition, FOC, etc.

    Voice-first move: `voice_picture` is the geometric/story analog
    (a tightrope between today and tomorrow; a vector in payoff space)
    that lets the listener hold the equation without seeing it.
    """

    id: str                           # SE1, SE2, ...
    name: str = ""
    what_it_says: str = ""
    voice_picture: str = ""
    role_in_argument: str = ""
    teaching_priority: Priority = "mention"


class StructuralModel(LenientModel):
    """The economic model behind a structural / theory / hybrid paper.
    Empirical_causal papers usually have no structural model; leave this
    field unset rather than inventing one.
    """

    agents: str = ""
    preferences_or_objective: str = ""
    technology_or_constraints: str = ""
    equilibrium_concept: str = ""
    parameters_estimated: list[str] = Field(default_factory=list)
    parameters_calibrated: list[str] = Field(default_factory=list)
    teaching_priority: Priority = "mention"


class Estimate(LenientModel):
    """A headline number. Every entry must carry both the raw point estimate
    and an `economic_translation` — the spoken sentence that puts the
    number into a unit a human can feel. "Coefficient is 0.034 (0.012)"
    is not enough; "a one-sd rise in X moves Y by 3% of its sample mean"
    is.
    """

    id: str                           # ES1, ES2, ...
    parameter_name: str = ""
    point_estimate: str = ""
    std_error_or_ci: str = ""
    unit: EstimateUnit = "level"
    economic_translation: str = ""
    comparable_benchmarks: list[str] = Field(default_factory=list)
    teaching_priority: Priority = "mention"


class RobustnessCheck(LenientModel):
    id: str                           # RC1, RC2, ...
    check_type: RobustnessType = "alt_specification"
    what_it_rules_out: str = ""
    result_summary: str = ""
    headline_survives: bool = False


class Mechanism(LenientModel):
    """The 'why' — the channel through which the headline effect runs.
    Distinct from the headline itself: a paper can credibly identify an
    effect without nailing the mechanism.
    """

    proposed_channel: str = ""
    evidence_for_channel: str = ""
    alternatives_ruled_out: str = ""


class FactorModelComparison(LenientModel):
    """Asset-pricing-specific. The standard table walks an anomaly's alpha
    across nested factor models (CAPM → FF3 → FF5 → q-factor) to see how
    far it survives. `nested_models` and `alpha_per_model` are parallel
    lists indexed together.
    """

    nested_models: list[str] = Field(default_factory=list)
    alpha_per_model: list[str] = Field(default_factory=list)
    survives_all_models: bool = False
    interpretation: str = ""


class PolicyImplication(LenientModel):
    policy_question: str = ""
    magnitude_in_policy_units: str = ""
    caveats: str = ""


# ---- outline -------------------------------------------------------------


class Outline(LenientModel):
    """Stage-1 contract for econ papers.

    Conditional fields by genre:
      - empirical_causal: `identification` + `specifications` required
      - structural / pure_theory: `structural_model` + `structural_equations` required
      - asset_pricing: `factor_model_comparison` (when applicable) +
        either structural_equations (no-arbitrage / SDF papers) or
        specifications (factor regressions)
      - survey: most fields optional; `mechanism` may carry the synthesis

    The model isn't enforced — the prompt asks for genre-appropriate
    coverage and the audit prompt rechecks. We'd rather have a partial
    outline land than reject a paper that doesn't fit one of the five
    genres cleanly.
    """

    paper_id: str
    genre: PaperGenre = "empirical_causal"
    core_thesis: str = ""
    gap_filled: str = ""

    # Empirical-causal block
    identification: Identification | None = None
    specifications: list[Specification] = Field(default_factory=list)

    # Structural / theoretical block
    structural_model: StructuralModel | None = None
    structural_equations: list[StructuralEquation] = Field(default_factory=list)

    # Asset-pricing block
    factor_model_comparison: FactorModelComparison | None = None

    # Universal
    estimates: list[Estimate] = Field(default_factory=list)
    robustness_checks: list[RobustnessCheck] = Field(default_factory=list)
    mechanism: Mechanism | None = None
    limitations_and_external_validity: list[str] = Field(default_factory=list)
    policy_implication: PolicyImplication | None = None

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
        if self.identification and self.identification.teaching_priority == "critical":
            out.append("ID")
        out.extend(s.id for s in self.specifications if s.teaching_priority == "critical")
        out.extend(e.id for e in self.structural_equations if e.teaching_priority == "critical")
        out.extend(es.id for es in self.estimates if es.teaching_priority == "critical")
        return out

    def important_ids(self) -> list[str]:
        out: list[str] = []
        if self.identification and self.identification.teaching_priority == "important":
            out.append("ID")
        out.extend(s.id for s in self.specifications if s.teaching_priority == "important")
        out.extend(e.id for e in self.structural_equations if e.teaching_priority == "important")
        out.extend(es.id for es in self.estimates if es.teaching_priority == "important")
        return out


def parse_outline(text: str) -> Outline:
    try:
        return Outline.from_yaml(text)
    except ValidationError as e:
        raise ParseError(f"outline schema validation failed:\n{e}", raw=text, validation=e) from e
    except yaml.YAMLError as e:
        raise ParseError(f"outline YAML parse failed: {e}", raw=text) from e


# ---- EpisodePlan (stage 1.5, optional) ----------------------------------
#
# Reuses the same shape as the ML pack — a list of segments referencing
# outline ids, plus committed `takes` for the persona. The realizer is the
# only consumer; it doesn't care which pack produced the plan.


class Segment(LenientModel):
    id: str
    role: str
    covers: list[str] = Field(default_factory=list)
    callbacks: list[str] = Field(default_factory=list)
    purpose: str


class Take(LenientModel):
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
