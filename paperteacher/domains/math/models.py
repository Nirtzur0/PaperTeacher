"""Pydantic models for the math domain's stage-1 outline.

Why this schema differs from the ml or physics packs:

- The spine of a math paper is theorems and definitions, not equations.
  An equation in a math paper is usually a functional equation, an
  identity, an inequality, or a recursion — what matters is what each
  piece IS and what role it plays, not units (no dimensional analysis)
  and not benchmark numbers (no SOTA).
- A theorem carries `hypotheses` (each with a `where_it_bites` field
  telling the script where in the proof that hypothesis is essential)
  and a `sharpness` field forcing the script to address whether the
  hypotheses are necessary, the bound is tight, the converse holds.
  Generality and tightness are usually the entire point of a math paper.
- A first-class `canonical_examples` list (canonical / edge / counter)
  encodes the way math is actually taught — examples illuminate the
  theorem before the abstract statement does its work.
- Definitions are their own entity. The "obvious" definitions in a math
  paper hide all the work; surfacing them up front makes the script
  walk the listener through what's actually being defined.

Stage-3 audit + ParseError + the LenientModel base live in
`domains/_common.py` since they're cross-domain.
"""
from __future__ import annotations

from typing import Literal

import yaml
from pydantic import Field, ValidationError

from .._common import LenientModel, ParseError, Priority

# Math paper types: pure-theory derivation, applied (math used in service of
# another field), survey, computational (numerics / lattice / experiment in
# the math sense), open-problem (papers framed around an unresolved question).
PaperType = Literal["theory", "applied", "survey", "computational", "open_problem"]


class Definition(LenientModel):
    """A defined object the paper introduces or relies on. The `plain_english`
    is what the listener should leave the segment knowing; the
    `canonical_example` makes the abstract definition land before any
    formal statement.
    """

    id: str
    name: str = ""
    plain_english: str = ""
    why_it_matters: str = ""
    canonical_example: str = ""
    teaching_priority: Priority = "mention"


class Hypothesis(LenientModel):
    """One assumption a theorem rests on. Tracked separately so the audit
    can check whether the script names where each hypothesis BITES — math
    proofs almost always have one or two hypotheses doing the load-bearing
    work and a few that look essential but are technical conveniences.
    """

    statement: str = ""
    where_it_bites: str = ""


class Theorem(LenientModel):
    """The centerpiece object of a math paper. `key_idea_of_proof` is the
    one substantive move (not the bookkeeping); `sharpness` forces the
    script to address whether the result is tight.
    """

    id: str
    name: str = ""
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    conclusion: str = ""
    sharpness: str = ""
    """Is the bound tight? Does the converse hold? Are the hypotheses
    necessary? e.g. "the constant 1/4 is sharp; equality holds for the
    Heisenberg uncertainty saturator". Empty allowed for `mention` items
    only — `critical` items must address sharpness."""

    key_idea_of_proof: str = ""
    """The one move that makes the proof go through, in plain English.
    "construct an auxiliary function f and show it's both subharmonic
    and bounded — Liouville does the rest". NOT the bookkeeping."""

    role_in_paper: str = ""
    """e.g. "main theorem", "the lemma everything reduces to", "corollary
    used in section 5"."""

    historical_thread: str = ""
    teaching_priority: Priority = "mention"
    note: str | None = None


class Concept(LenientModel):
    id: str
    name: str = ""
    plain_english: str = ""
    why_it_matters: str = ""
    historical_thread: str = ""
    teaching_priority: Priority = "mention"


class EquationComponent(LenientModel):
    role: str = ""
    intuition: str = ""
    what_if_removed: str = ""


class Equation(LenientModel):
    """For math papers, equations are functional equations, identities,
    inequalities, recursions. No dimensional check; what matters is what
    each piece IS and where it FORCES the answer. `worked_specialization`
    replaces the physics pack's Fermi estimate — taking the equation in
    dimension 1 / the abelian case / for n=2 is the math analog.
    """

    id: str
    english_name: str = ""
    what_it_says: str = ""
    structure_in_words: str = ""
    components: list[EquationComponent] = Field(default_factory=list)
    key_trick: str = ""
    worked_specialization: str = ""
    """The equation taken in a tractable special case. e.g. "in dimension
    1, the recursion collapses to Fibonacci"; "for the abelian group
    case, the cohomology vanishes and the formula reduces to a count"."""

    teaching_priority: Priority = "mention"


class Example(LenientModel):
    """Math is taught by examples. The canonical one shows the typical
    case; the edge / counter example shows where things break or where the
    hypotheses bite.
    """

    id: str
    kind: Literal["canonical", "edge_case", "counterexample"] = "canonical"
    description: str = ""
    what_it_illuminates: str = ""


class Outline(LenientModel):
    paper_id: str
    type: PaperType = "theory"
    core_thesis: str = ""
    gap_filled: str = ""

    historical_context: list[str] = Field(default_factory=list)
    """1-3 short threads placing the work in tradition. e.g. "extends
    Bourgain's restriction estimates from p=4 to the full Tomas-Stein
    range"; "the modern descendant of Cantor's middle-thirds construction".
    Skip for routine technical results without a clear lineage."""

    key_definitions: list[Definition] = Field(default_factory=list)
    key_theorems: list[Theorem] = Field(default_factory=list)
    key_concepts: list[Concept] = Field(default_factory=list)
    key_equations: list[Equation] = Field(default_factory=list)
    canonical_examples: list[Example] = Field(default_factory=list)
    conjectures_referenced: list[str] = Field(default_factory=list)
    """Open problems or named conjectures the paper engages with. e.g.
    "Riemann Hypothesis"; "Erdős–Ko–Rado in the chromatic setting";
    "the abc conjecture". Empty list for self-contained technical work."""

    limitations_and_open_questions: list[str] = Field(default_factory=list)
    acronyms_to_spell_out: dict[str, str] = Field(default_factory=dict)
    hard_pronunciations: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, text: str) -> "Outline":
        return cls.model_validate(yaml.safe_load(text))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False)

    def critical_ids(self) -> list[str]:
        out: list[str] = []
        out.extend(d.id for d in self.key_definitions if d.teaching_priority == "critical")
        out.extend(t.id for t in self.key_theorems if t.teaching_priority == "critical")
        out.extend(c.id for c in self.key_concepts if c.teaching_priority == "critical")
        out.extend(e.id for e in self.key_equations if e.teaching_priority == "critical")
        return out

    def important_ids(self) -> list[str]:
        out: list[str] = []
        out.extend(d.id for d in self.key_definitions if d.teaching_priority == "important")
        out.extend(t.id for t in self.key_theorems if t.teaching_priority == "important")
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


class Segment(LenientModel):
    id: str
    role: str
    """Free-form. Suggested math-shaped roles (the prompt expands on
    these, the schema doesn't constrain them):
        opening | motivation | tradition | definitions | example_first
        | theorem_statement | proof_sketch | hypothesis_bite | sharpness
        | counterexample | comparison | critique | closer
    """
    covers: list[str] = Field(default_factory=list)
    callbacks: list[str] = Field(default_factory=list)
    purpose: str


class Take(LenientModel):
    """A committed opinion the professor holds across the whole episode.
    Math takes are often about generality vs. cleanness ("the theorem is
    stated in the most general setting but the only case anyone uses is
    the smooth one"), or about whether the new contribution is the result
    or the technique ("the technique is the contribution; the headline
    theorem was already accessible to anyone willing to grind").
    """

    claim: str
    evidence: str


class EpisodePlan(LenientModel):
    paper_id: str
    arc: list[Segment]
    takes: list[Take] = Field(default_factory=list)
    sits_alongside: list[str] = Field(default_factory=list)
    """1-3 adjacent works or paradigms. For math this often crosses
    decades — Cantor's diagonal, Erdős' probabilistic method,
    Grothendieck's relative point of view, Wiles on FLT — not just the
    last 18 months of arXiv."""

    why_now: str = ""
    """What changed in technique, computation, or adjacent fields that
    made this paper land now specifically."""

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
