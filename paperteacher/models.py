"""Pydantic models for the pipeline's typed contracts.

The LLM's output for stage 1 (outline) and stage 3 (audit) must conform to
these shapes. Validating at save-time turns silent corruption into a loud,
fixable error.
"""
from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

Priority = Literal["critical", "important", "mention"]
PaperType = Literal["theoretical", "empirical", "position", "survey"]
AuditRecommendation = Literal["ship", "regenerate_with_gaps", "regenerate_from_scratch"]
AuditCoverage = Literal["complete", "partial", "poor"]


class _Lenient(BaseModel):
    """Allow unknown fields, coerce numbers to strings.

    The `coerce_numbers_to_str` matters: arXiv IDs like ``2603.20105`` look
    like floats to YAML's parser, so without coercion the LLM has to remember
    to quote them. With coercion, both ``2603.20105`` and ``"2603.20105"``
    are accepted.
    """

    model_config = ConfigDict(extra="ignore", coerce_numbers_to_str=True)


# ---- Outline (stage 1 output) ---------------------------------------------


class EquationComponent(_Lenient):
    role: str
    intuition: str
    what_if_removed: str = ""


class Equation(_Lenient):
    id: str
    english_name: str
    what_it_solves: str
    structure_in_words: str
    components: list[EquationComponent] = Field(default_factory=list)
    key_trick: str = ""
    geometric_picture: str = ""
    numerical_walkthrough: str = ""
    bridge_to_next: str = ""
    teaching_priority: Priority
    note: str | None = None


class Concept(_Lenient):
    id: str
    name: str
    plain_english: str
    why_it_matters: str
    teaching_priority: Priority


class Result(_Lenient):
    id: str
    claim: str
    what_it_demonstrates: str
    why_surprising: str | None = None


class Outline(_Lenient):
    paper_id: str
    type: PaperType
    core_thesis: str
    gap_filled: str
    key_concepts: list[Concept] = Field(default_factory=list)
    key_equations: list[Equation] = Field(default_factory=list)
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
        out.extend(e.id for e in self.key_equations if e.teaching_priority == "critical")
        return out

    def important_ids(self) -> list[str]:
        out: list[str] = []
        out.extend(c.id for c in self.key_concepts if c.teaching_priority == "important")
        out.extend(e.id for e in self.key_equations if e.teaching_priority == "important")
        return out


# ---- AuditReport (stage 3 output) -----------------------------------------


class MissingItem(_Lenient):
    id: str
    name: str = ""
    what_was_said: str = ""
    what_is_missing: str
    severity: Priority


class GlossedItem(_Lenient):
    id: str
    quote: str
    why_its_a_gloss: str


class VoiceFirstViolation(_Lenient):
    quote: str
    why: str


class AuditReport(_Lenient):
    coverage_status: AuditCoverage
    items_missing: list[MissingItem] = Field(default_factory=list)
    items_glossed: list[GlossedItem] = Field(default_factory=list)
    banned_phrases_used: list[str] = Field(default_factory=list)
    voice_first_violations: list[VoiceFirstViolation] = Field(default_factory=list)
    overall_assessment: str
    recommendation: AuditRecommendation

    @classmethod
    def from_yaml(cls, text: str) -> "AuditReport":
        return cls.model_validate(yaml.safe_load(text))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False)

    def passed(self) -> bool:
        return self.recommendation == "ship"


# ---- helpers --------------------------------------------------------------


class ParseError(ValueError):
    """Raised when an LLM payload doesn't conform to the expected schema.

    Carries the raw text so callers can show the offending payload.
    """

    def __init__(self, message: str, raw: str, validation: ValidationError | None = None):
        super().__init__(message)
        self.raw = raw
        self.validation = validation


def parse_outline(text: str) -> Outline:
    try:
        return Outline.from_yaml(text)
    except ValidationError as e:
        raise ParseError(f"outline schema validation failed:\n{e}", raw=text, validation=e) from e
    except yaml.YAMLError as e:
        raise ParseError(f"outline YAML parse failed: {e}", raw=text) from e


def parse_audit(text: str) -> AuditReport:
    try:
        return AuditReport.from_yaml(text)
    except ValidationError as e:
        raise ParseError(f"audit schema validation failed:\n{e}", raw=text, validation=e) from e
    except yaml.YAMLError as e:
        raise ParseError(f"audit YAML parse failed: {e}", raw=text) from e
