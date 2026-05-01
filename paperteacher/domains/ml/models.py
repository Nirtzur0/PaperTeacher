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
    role: str
    intuition: str
    what_if_removed: str = ""


class Equation(LenientModel):
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


class Concept(LenientModel):
    id: str
    name: str
    plain_english: str
    why_it_matters: str
    teaching_priority: Priority


class Result(LenientModel):
    id: str
    claim: str
    what_it_demonstrates: str
    why_surprising: str | None = None


class Outline(LenientModel):
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


def parse_outline(text: str) -> Outline:
    try:
        return Outline.from_yaml(text)
    except ValidationError as e:
        raise ParseError(f"outline schema validation failed:\n{e}", raw=text, validation=e) from e
    except yaml.YAMLError as e:
        raise ParseError(f"outline YAML parse failed: {e}", raw=text) from e
