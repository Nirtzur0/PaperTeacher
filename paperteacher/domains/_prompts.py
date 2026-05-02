"""Shared prompt-rendering scaffolding for all domain packs.

Every pack ships four prompt templates (extract / plan / teach / audit) plus
four render helpers that fill placeholders. The render helpers were byte-
identical across ml/physics/neuro/econ; this module hosts them once.

Each pack now defines just the four template strings as module-level
constants (`EXTRACT_OUTLINE`, `PLAN_EPISODE`, `TEACH_FROM_OUTLINE`,
`AUDIT_COVERAGE`) plus its two structure variants (`_STRUCTURE_DEFAULT`,
`_STRUCTURE_FROM_PLAN`), and exposes a tiny `prompts.py` that delegates
its render_* calls here.

Token-diet:
  - The plan body still drops paper_text — the outline already carries every
    structurally relevant claim/equation/limitation. Saves ~30K tokens per
    plan call.
  - taste_profile IS inlined in extract, plan, and teach: it's the voice
    anchor (PhD friend, geometric/physical analogies, intuition-before-
    formalism). ~500 tokens of context that actively shapes output, not
    redundant ceremony — earlier diet had pulled this from plan/teach and
    it visibly killed the script's voice. Restored.

Placeholder vocabulary across packs:
  - extract: arxiv_id, title, taste_profile, paper_text
  - plan:    arxiv_id, title, taste_profile, outline_yaml
  - teach:   arxiv_id, title, taste_profile, paper_text, outline_yaml,
             mode, plan_section, structure_section,
             target_words, target_minutes, voice_guide_section
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
    taste_profile: str,
    outline_yaml: str,
) -> str:
    return template.format(
        arxiv_id=arxiv_id,
        title=title,
        taste_profile=taste_profile,
        outline_yaml=outline_yaml,
    )


# ---- teach -------------------------------------------------------------


def render_teach_template(
    teach_template: str,
    *,
    structure_default: str,
    structure_from_plan: str,
    arxiv_id: str,
    title: str,
    taste_profile: str,
    paper_text: str,
    outline_yaml: str,
    mode: str = "single_host",
    plan_yaml: str | None = None,
    target_words: int | None = None,
    target_minutes: int | None = None,
    voice_guide: str = "",
    extras: dict[str, str] | None = None,
) -> str:
    """Render the stage-2 prompt.

    `target_words` / `target_minutes` resolve from the profile when omitted,
    so callers don't duplicate the lookup.

    `voice_guide` is the per-pack pronunciation/banned-phrase block,
    inlined into the `{voice_guide_section}` placeholder. Packs that don't
    use a separate voice guide pass empty string and `.format()` ignores
    the missing placeholder reference (Python semantics).
    """
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

    return teach_template.format(
        arxiv_id=arxiv_id,
        title=title,
        taste_profile=taste_profile,
        paper_text=paper_text,
        outline_yaml=outline_yaml,
        mode=mode,
        plan_section=plan_section,
        structure_section=structure_section,
        target_words=target_words,
        target_minutes=target_minutes,
        voice_guide_section=voice_guide,
        **(extras or {}),
    )


# ---- audit -------------------------------------------------------------


def render_audit_template(template: str, *, outline_yaml: str, script: str) -> str:
    return template.format(outline_yaml=outline_yaml, script=script)
