"""Shared prompt-rendering scaffolding for all domain packs.

Every pack ships four prompt templates (extract / plan / teach / audit) plus
four render helpers that fill placeholders. The render helpers were byte-
identical across ml/physics/neuro/econ; this module hosts them once.

Each pack now defines just the four template strings as module-level
constants (`EXTRACT_OUTLINE`, `PLAN_EPISODE`, `TEACH_FROM_OUTLINE`,
`AUDIT_COVERAGE`) plus its two structure variants (`_STRUCTURE_DEFAULT`,
`_STRUCTURE_FROM_PLAN`), and exposes a tiny `prompts.py` that delegates
its render_* calls here.

Token-diet (2026-05-02): `taste_profile` is NO LONGER inlined in plan or
teach prompts. The host agent reads `profile://taste` once at stage 0;
re-shipping the same text every stage was ~1.5K tokens of redundancy.
`paper_text` is also dropped from the plan template — the outline carries
every structurally relevant claim/equation/limitation, so the planner has
what it needs without the prose. Saves ~30K tokens per plan call. The
voice guide moves to a `voice-guide://<domain>` MCP resource for hosts
that support resource caching; CLI users still get it inlined since they
have no resource layer.

Placeholder vocabulary across packs:
  - extract: arxiv_id, title, taste_profile, paper_text  (UNCHANGED — the
             extractor needs both)
  - plan:    arxiv_id, title, outline_yaml               (no profile, no prose)
  - teach:   arxiv_id, title, paper_text, outline_yaml, mode, plan_section,
             structure_section, target_words, target_minutes,
             voice_guide_section
"""
from __future__ import annotations


# ---- extract -----------------------------------------------------------


def render_extract_template(
    template: str,
    *,
    arxiv_id: str,
    title: str,
    taste_profile: str,
    paper_text: str,
) -> str:
    """Stage 1. Profile + full paper text are both essential here — the
    extractor's output IS the contract for stages 2 and 3, so undertraining
    it would propagate everywhere downstream.
    """
    return template.format(
        arxiv_id=arxiv_id,
        title=title,
        taste_profile=taste_profile,
        paper_text=paper_text,
    )


# ---- plan --------------------------------------------------------------


def render_plan_template(
    template: str,
    *,
    arxiv_id: str,
    title: str,
    outline_yaml: str,
    # Accepted for back-compat with older callers; intentionally NOT inlined.
    # The outline carries the structural decisions; the host has the profile
    # via resource. Saves ~30K tokens per plan call.
    taste_profile: str | None = None,
    paper_text: str | None = None,
) -> str:
    del taste_profile, paper_text
    return template.format(
        arxiv_id=arxiv_id,
        title=title,
        outline_yaml=outline_yaml,
    )


# ---- teach -------------------------------------------------------------


# One-line stand-in when the prompt body shouldn't carry the full voice guide.
# The MCP server uses this so hosts that have already loaded the
# voice-guide://<domain> resource don't get the same content twice.
_VOICE_GUIDE_RESOURCE_REF = (
    "The voice guide (pronunciation table, numerical rewrites, banned "
    "phrases, anti-anthropomorphism rules) is loaded as the "
    "`voice-guide://{domain}` MCP resource — your host has it. Apply those "
    "rules to every line you emit; they are non-negotiable."
)


def render_teach_template(
    teach_template: str,
    *,
    structure_default: str,
    structure_from_plan: str,
    arxiv_id: str,
    title: str,
    paper_text: str,
    outline_yaml: str,
    mode: str = "single_host",
    plan_yaml: str | None = None,
    target_words: int | None = None,
    target_minutes: int | None = None,
    voice_guide: str = "",
    inline_voice_guide: bool = True,
    domain_name: str = "ml",
    extras: dict[str, str] | None = None,
    # Accepted for back-compat; intentionally not inlined (the host has the
    # `profile://taste` resource).
    taste_profile: str | None = None,
) -> str:
    """Render the stage-2 prompt.

    `target_words` / `target_minutes` resolve from the profile when omitted,
    so callers don't duplicate the lookup.

    `voice_guide` is the per-pack pronunciation/banned-phrase table.
    `inline_voice_guide=True` (CLI default) embeds it directly; `False` (MCP
    server) replaces it with a one-line pointer to the
    `voice-guide://<domain>` resource — the host already has the content
    loaded once and doesn't need it re-shipped every regen.
    """
    del taste_profile  # silently dropped — host has the resource
    if target_words is None or target_minutes is None:
        from .. import profile as _profile_mod

        prof = _profile_mod.load()
        if target_words is None:
            target_words = prof.target_script_words
        if target_minutes is None:
            target_minutes = prof.length_target_minutes

    if plan_yaml:
        plan_section = (
            "\nEPISODE PLAN (the macro structure — follow this arc, segment by segment):\n"
            "---\n"
            f"{plan_yaml}\n"
            "---\n"
        )
        structure_template = structure_from_plan
    else:
        plan_section = ""
        structure_template = structure_default

    structure_section = structure_template.format(
        target_words=target_words,
        target_minutes=target_minutes,
    )

    if inline_voice_guide:
        voice_guide_section = voice_guide
    else:
        voice_guide_section = _VOICE_GUIDE_RESOURCE_REF.format(domain=domain_name)

    return teach_template.format(
        arxiv_id=arxiv_id,
        title=title,
        paper_text=paper_text,
        outline_yaml=outline_yaml,
        mode=mode,
        plan_section=plan_section,
        structure_section=structure_section,
        target_words=target_words,
        target_minutes=target_minutes,
        voice_guide_section=voice_guide_section,
        **(extras or {}),
    )


# ---- audit -------------------------------------------------------------


def render_audit_template(template: str, *, outline_yaml: str, script: str) -> str:
    return template.format(outline_yaml=outline_yaml, script=script)
