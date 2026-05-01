"""Domain pack abstraction.

A `Domain` bundles the subject-specific bits of the pipeline:
  - the stage-1 outline schema (math papers have Equations; humanities papers
    have Arguments / Counterclaims; biology papers have Hypotheses / Methods)
  - the stage-1/2/3 prompt templates
  - discovery sources (arXiv vs SSRN vs PubMed vs PhilPapers ...)
  - the reader (arXiv HTML chain vs PDF parsing vs DOI redirect)

The orchestration framework (storage, audio, TTS, MCP server, CLI) is
domain-agnostic and queries `active_domain()` for these pieces at call time.

Active-domain resolution (first hit wins):
  1. PAPERTEACHER_DOMAIN env var
  2. `domain: <name>` line in profile.md
  3. default "ml"
"""
from __future__ import annotations

import os
import re
from typing import Awaitable, Callable, Protocol, runtime_checkable

from pydantic import BaseModel

from .domains._common import AuditReport, Candidate, ParseError, PaperText


@runtime_checkable
class Domain(Protocol):
    """The interface a domain pack exposes. Concrete domains live in
    `paperteacher/domains/<name>/__init__.py` and call `register_domain()`
    with a factory that returns an instance.
    """

    name: str
    OutlineModel: type[BaseModel]  # stage-1 typed output

    # parsers — domains may share `parse_audit` from _common
    def parse_outline(self, text: str) -> BaseModel: ...
    def parse_audit(self, text: str) -> AuditReport: ...

    # prompt templates — return the rendered string the host LLM sees
    def render_extract(self, *, arxiv_id: str, title: str, taste_profile: str, paper_text: str) -> str: ...
    def render_teach(
        self,
        *,
        arxiv_id: str,
        title: str,
        taste_profile: str,
        paper_text: str,
        outline_yaml: str,
        mode: str,
    ) -> str: ...
    def render_audit(self, *, outline_yaml: str, script: str) -> str: ...

    # discovery + reading — async since they hit network
    discover: Callable[..., Awaitable[list[Candidate]]]
    read: Callable[..., Awaitable[PaperText]]


# ---- registry ------------------------------------------------------------

_REGISTRY: dict[str, Callable[[], Domain]] = {}


def register_domain(name: str, factory: Callable[[], Domain]) -> None:
    """Register a domain pack. `factory` is a zero-arg callable returning a
    fresh instance — typically the domain class itself.
    """
    _REGISTRY[name] = factory


def list_domains() -> list[str]:
    return sorted(_REGISTRY)


def get_domain(name: str) -> Domain:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown domain: {name!r}. Registered: {sorted(_REGISTRY)!r}"
        )
    return _REGISTRY[name]()


# ---- active-domain resolution -------------------------------------------

_active: Domain | None = None


def active_domain() -> Domain:
    """Resolve once per process, then cache. Call `reset_active()` from tests
    that switch domains via env vars.
    """
    global _active
    if _active is None:
        _ensure_bundled_domains_loaded()
        name = (
            os.environ.get("PAPERTEACHER_DOMAIN")
            or _domain_from_profile()
            or "ml"
        )
        _active = get_domain(name)
    return _active


def _ensure_bundled_domains_loaded() -> None:
    """Import each bundled domain pack so its register_domain() runs.

    Done lazily (not at module load) to avoid a circular import: this module
    imports types from `domains._common`, and domain packs import
    `register_domain` from this module — if we triggered pack imports during
    our own load, that's a cycle.
    """
    from .domains import ml  # noqa: F401
    # add more bundled packs here as they're created


def reset_active() -> None:
    """Clear the cached active domain. Tests use this when switching."""
    global _active
    _active = None


def _domain_from_profile() -> str | None:
    """Look for a `domain: <name>` line in the listener profile.

    The profile is otherwise free-form markdown; this is a single-line
    convention so users don't have to switch to YAML for one field.
    """
    from . import paths

    if not paths.PROFILE_PATH.exists():
        return None
    text = paths.PROFILE_PATH.read_text()
    m = re.search(r"^\s*domain:\s*([\w-]+)\s*$", text, re.MULTILINE)
    return m.group(1) if m else None


__all__ = [
    "Domain",
    "register_domain",
    "get_domain",
    "list_domains",
    "active_domain",
    "reset_active",
    "ParseError",
    "Candidate",
    "PaperText",
    "AuditReport",
]
