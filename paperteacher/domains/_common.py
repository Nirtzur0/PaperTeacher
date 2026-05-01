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


# ---- domain pack factory --------------------------------------------------


def make_domain(
    name: str,
    *,
    models,
    prompts,
    discovery,
    reader,
):
    """Build a Domain class from a pack's submodules.

    All four bundled packs (ml, physics, neuro, econ) follow the same shape:
    they expose `models`, `prompts`, `discovery`, `reader` modules and want a
    Domain class that wires the standard methods onto them. This factory does
    that wiring once.

    Required attributes on each module:
      models:    Outline (BaseModel), parse_outline(text) -> BaseModel.
                 Optional: EpisodePlan + parse_plan(text) for the planner stage.
      prompts:   render_extract / render_plan / render_teach / render_audit.
      discovery: discover(...) -> list[Candidate].
      reader:    read_paper(arxiv_id, ...) -> PaperText.
    """
    attrs: dict = {
        "name": name,
        "OutlineModel": models.Outline,
        "parse_outline": staticmethod(models.parse_outline),
        "parse_audit": staticmethod(parse_audit),
        "render_extract": staticmethod(prompts.render_extract),
        "render_teach": staticmethod(prompts.render_teach),
        "render_audit": staticmethod(prompts.render_audit),
        "discover": staticmethod(discovery.discover),
        "read": staticmethod(reader.read_paper),
    }
    # Planner stage is opt-in — copy the bits over only when the pack ships them.
    plan_model = getattr(models, "EpisodePlan", None)
    if plan_model is not None:
        attrs["PlanModel"] = plan_model
    if hasattr(models, "parse_plan"):
        attrs["parse_plan"] = staticmethod(models.parse_plan)
    if hasattr(prompts, "render_plan"):
        attrs["render_plan"] = staticmethod(prompts.render_plan)

    # Voice guide is also opt-in. Packs that have extracted their voice rules
    # into a `_VOICE_GUIDE` constant get it surfaced as a class attribute so
    # the MCP server can expose `voice-guide://<name>` and the teach renderer
    # can drop the table from the prompt body when the host has it cached.
    # Packs that haven't yet extracted their voice rules return "" — the
    # resource still resolves, just to an empty body, and the teach prompt
    # body keeps its inline rules.
    attrs["voice_guide_text"] = getattr(prompts, "_VOICE_GUIDE", "")

    return type(f"{name.capitalize()}Domain", (), attrs)
