"""Domain pack abstraction.

A `Domain` bundles the subject-specific bits of the pipeline:
  - the stage-1 outline schema (math papers have Equations; humanities papers
    have Arguments / Counterclaims; biology papers have Hypotheses / Methods)
  - the stage-1/2/3 prompt templates
  - discovery sources (arXiv vs SSRN vs PubMed vs PhilPapers ...)
  - the reader (arXiv HTML chain vs PDF parsing vs DOI redirect)

The orchestration framework (storage, audio, TTS, MCP server, CLI) is
domain-agnostic and queries the routing helpers below for these pieces.

Active-domain resolution (first hit wins):
  1. PAPERTEACHER_DOMAIN env var (comma-separated for multi)
  2. `domains: a, b` (list) or `domain: a` (single) line in profile.md
  3. default ["ml"]

Per-paper routing: discovery stamps `Candidate.domain`; once a paper has
been read, `meta/<id>.json` records its pack so save/load/prompt calls all
route through the same one. `domain_for(arxiv_id)` is the single lookup.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Awaitable, Callable, Protocol, runtime_checkable

from pydantic import BaseModel

from .domains._common import AuditReport, Candidate, ParseError, PaperText

log = logging.getLogger(__name__)


@runtime_checkable
class Domain(Protocol):
    """The interface a domain pack exposes. Concrete domains live in
    `paperteacher/domains/<name>/__init__.py` and call `register_domain()`
    with a factory that returns an instance.

    Optional (opt-in per pack): the planner stage. A domain may also expose
    `PlanModel`, `parse_plan(text) -> BaseModel`, and `render_plan(...)`.
    Callers (storage, CLI, server) discover these via getattr and gracefully
    skip the planner stage on packs that don't implement it. When a plan has
    been saved for a paper, `render_teach` receives it via `plan_yaml`; the
    pack decides whether and how to honor it.
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
        plan_yaml: str | None = None,
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

_active_list: list[Domain] | None = None


def active_domains() -> list[Domain]:
    """Resolve the configured set of active packs. First call triggers
    bundled-pack import + profile/env parsing; result is cached.
    """
    global _active_list
    if _active_list is None:
        _ensure_bundled_domains_loaded()
        names = (
            _names_from_env()
            or _names_from_profile()
            or ["ml"]
        )
        _active_list = [get_domain(n) for n in names]
    return _active_list


def active_domain() -> Domain:
    """First active pack — convenience for code paths that haven't been
    routed by candidate yet (single-paper CLI commands, fallbacks).
    """
    return active_domains()[0]


def _ensure_bundled_domains_loaded() -> None:
    """Import each bundled domain pack so its register_domain() runs.

    Done lazily (not at module load) to avoid a circular import: this module
    imports types from `domains._common`, and domain packs import
    `register_domain` from this module — if we triggered pack imports during
    our own load, that's a cycle.
    """
    from .domains import econ  # noqa: F401
    from .domains import ml  # noqa: F401
    from .domains import neuro  # noqa: F401
    from .domains import physics  # noqa: F401
    # add more bundled packs here as they're created


def reset_active() -> None:
    """Clear the cached active-domain list. Tests use this when switching."""
    global _active_list
    _active_list = None


def _names_from_env() -> list[str] | None:
    raw = os.environ.get("PAPERTEACHER_DOMAIN")
    if not raw:
        return None
    return [n.strip() for n in raw.split(",") if n.strip()]


def _names_from_profile() -> list[str] | None:
    """Look for `domains: a, b` (list) or `domain: a` (single) in profile.md.

    Lists win when both are present. The profile is otherwise free-form
    markdown; this is a single-line convention so users don't have to switch
    to YAML for one field.
    """
    from . import paths

    if not paths.PROFILE_PATH.exists():
        return None
    text = paths.PROFILE_PATH.read_text()
    m = re.search(r"^\s*domains:\s*(.+?)\s*$", text, re.MULTILINE)
    if m:
        names = [n.strip() for n in m.group(1).split(",") if n.strip()]
        return names or None
    m = re.search(r"^\s*domain:\s*([\w-]+)\s*$", text, re.MULTILINE)
    return [m.group(1)] if m else None


# ---- per-paper routing --------------------------------------------------


def _meta_path(arxiv_id: str):
    from . import paths
    return paths.META_DIR / f"{arxiv_id}.json"


def record_domain(arxiv_id: str, name: str) -> None:
    """Stamp which pack handled a paper. Idempotent. Subsequent storage
    and prompt calls look up the pack via `domain_for(arxiv_id)`.
    """
    from . import paths
    paths.ensure_layout()
    p = _meta_path(arxiv_id)
    if p.exists():
        try:
            existing = json.loads(p.read_text()).get("domain")
            if existing == name:
                return
        except (OSError, ValueError):
            pass
    p.write_text(json.dumps({"domain": name}))


def domain_for(arxiv_id: str) -> Domain:
    """Resolve the pack responsible for `arxiv_id`. Falls back to the first
    active pack when no metadata is recorded yet (i.e. legacy data, or a
    paper being read for the first time without a candidate hint).

    Bundled packs are loaded eagerly here so a ValueError from
    `get_domain(name)` reflects a genuinely unknown pack (e.g. a stamped
    sidecar referencing a third-party pack that hasn't been imported), not
    "the registry just hasn't booted yet". Without this, every CLI command
    that touched a meta-sidecared paper logged a noisy "Unknown domain"
    warning before silently falling through to active_domain().
    """
    _ensure_bundled_domains_loaded()
    p = _meta_path(arxiv_id)
    if p.exists():
        try:
            name = json.loads(p.read_text()).get("domain")
            if name:
                return get_domain(name)
        except (OSError, ValueError) as e:
            log.warning("meta read failed for %s: %s", arxiv_id, e)
    return active_domain()


# ---- cross-pack discovery + read ----------------------------------------


async def discover_all(
    arxiv_categories: list[str] | None = None,
    limit: int | None = None,
) -> list[Candidate]:
    """Run every active pack's `discover()` in parallel, stamp each result
    with its origin pack, and merge by arxiv_id.

    Earlier packs win on duplicates — order in `domains:` is meaningful.
    """
    packs = active_domains()
    kwargs: dict = {}
    if arxiv_categories is not None:
        kwargs["arxiv_categories"] = arxiv_categories
    if limit is not None:
        kwargs["limit"] = limit

    async def _one(pack: Domain) -> list[Candidate]:
        try:
            cands = await pack.discover(**kwargs)
        except TypeError:
            # Pack's discover may not accept arxiv_categories (e.g. PubMed).
            cands = await pack.discover(**{k: v for k, v in kwargs.items() if k != "arxiv_categories"})
        for c in cands:
            c.domain = pack.name
        return cands

    results = await asyncio.gather(*[_one(p) for p in packs])
    seen: set[str] = set()
    merged: list[Candidate] = []
    for batch in results:
        for c in batch:
            if c.arxiv_id in seen:
                continue
            seen.add(c.arxiv_id)
            merged.append(c)
    return merged


async def read_paper(
    arxiv_id: str,
    *,
    max_chars: int | None = None,
    hint: str | None = None,
) -> PaperText:
    """Route a read to the right pack and stamp `meta/<id>.json` on success.

    Routing order: explicit `hint` (e.g. from `Candidate.domain`) → recorded
    meta sidecar → first active pack. This is the single entry point so
    every caller (CLI, MCP server, prompt renderers) records uniformly.
    """
    if hint:
        pack = get_domain(hint)
    else:
        pack = domain_for(arxiv_id)
    kwargs = {} if max_chars is None else {"max_chars": max_chars}
    paper = await pack.read(arxiv_id, **kwargs)
    if paper.source != "none":
        record_domain(arxiv_id, pack.name)
    return paper


__all__ = [
    "Domain",
    "register_domain",
    "get_domain",
    "list_domains",
    "active_domain",
    "active_domains",
    "domain_for",
    "record_domain",
    "discover_all",
    "read_paper",
    "reset_active",
    "ParseError",
    "Candidate",
    "PaperText",
    "AuditReport",
]
