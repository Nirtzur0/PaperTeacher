"""Cross-domain types: lifted out so future domain packs can reuse them.

What's domain-agnostic:
  - Priority/AuditRecommendation/AuditCoverage enums
  - The lenient base model (extra="ignore", coerce_numbers_to_str)
  - AuditReport (stage-3 output) + parse_audit
  - ParseError
  - Candidate (discovery item) + PaperText (read result)

What stays domain-specific (in domains/<name>/):
  - The Outline schema (math papers have Equations; humanities papers have Arguments)
  - The stage-1 / stage-2 / stage-3 prompt templates
  - Discovery sources
  - The reader's fallback chain
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# ---- shared enums --------------------------------------------------------

Priority = Literal["critical", "important", "mention"]
AuditRecommendation = Literal["ship", "regenerate_with_gaps", "regenerate_from_scratch"]
AuditCoverage = Literal["complete", "partial", "poor"]


class LenientModel(BaseModel):
    """Allow unknown fields, coerce numbers to strings.

    `coerce_numbers_to_str=True` matters: arXiv-style ids like ``2603.20105``
    look like floats to YAML's parser, so without coercion the LLM has to
    remember to quote them. With coercion both forms are accepted. The setting
    is harmless for non-numeric ids (DOIs, PhilPapers ids, etc.).
    """

    model_config = ConfigDict(extra="ignore", coerce_numbers_to_str=True)


# ---- AuditReport (stage 3 output) ----------------------------------------


class MissingItem(LenientModel):
    id: str
    name: str = ""
    what_was_said: str = ""
    what_is_missing: str
    severity: Priority


class GlossedItem(LenientModel):
    id: str
    quote: str
    why_its_a_gloss: str


class VoiceFirstViolation(LenientModel):
    quote: str
    why: str


class AuditReport(LenientModel):
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


# ---- error type ----------------------------------------------------------


class ParseError(ValueError):
    """Raised when an LLM payload doesn't conform to the expected schema.

    Carries the raw text so callers can re-prompt with the offending payload.
    """

    def __init__(self, message: str, raw: str, validation: ValidationError | None = None):
        super().__init__(message)
        self.raw = raw
        self.validation = validation


def parse_audit(text: str) -> AuditReport:
    try:
        return AuditReport.from_yaml(text)
    except ValidationError as e:
        raise ParseError(f"audit schema validation failed:\n{e}", raw=text, validation=e) from e
    except yaml.YAMLError as e:
        raise ParseError(f"audit YAML parse failed: {e}", raw=text) from e


# ---- discovery / reader DTOs ---------------------------------------------


@dataclass
class Candidate:
    """A discovered paper. `arxiv_id` is the paper's domain-specific identifier
    (arxiv id for ML, DOI for econ, PhilPapers id for philosophy, etc.) — the
    name is kept for backward compat across the MCP/CLI surface; treat it as
    an opaque string at the framework level. `source` is a free-form tag from
    the discovery fetcher (e.g. "hf_daily", "arxiv_cs.LG", "ssrn_econ").
    """

    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    source: str
    score: float = 0.0
    url: str = ""
    # Stamped by `discover_all()` so the host can route subsequent
    # read/extract/teach/audit calls back to the originating pack.
    domain: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaperText:
    """The read-paper result. `source` reports which fetcher succeeded
    (e.g. "arxiv_html", "hf_paper", "arxiv_abs", "doi_redirect"). `source ==
    "none"` means total failure; `text` will be empty.
    """

    arxiv_id: str
    title: str
    text: str
    source: str
    truncated: bool = False
