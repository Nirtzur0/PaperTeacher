"""Neuroscience domain pack.

Subject coverage: neuroscience preprints (bioRxiv) and recent journal
papers (Europe PMC index of MEDLINE) — recordings, imaging, behavior,
circuits, computation, clinical translational.

The Outline schema's central unit is the **Finding**, decomposed
alongside the **Method** that produced it and the **Control** that ruled
out the obvious alternative explanation. The prompts enforce voice-first
reading rules tuned for the field — spell out brain-region acronyms,
don't read p-values aloud, never describe a method as "we recorded
neurons" — and the audit at stage 3 checks for those specifically.
"""
from __future__ import annotations

from .._common import AuditReport, parse_audit
from . import discovery, models, prompts, reader
from ...domain import register_domain


class NeuroDomain:
    name = "neuro"
    OutlineModel = models.Outline
    PlanModel = models.EpisodePlan  # opt-in planner stage

    # parsers
    parse_outline = staticmethod(models.parse_outline)
    parse_plan = staticmethod(models.parse_plan)
    parse_audit = staticmethod(parse_audit)

    # prompt rendering
    render_extract = staticmethod(prompts.render_extract)
    render_plan = staticmethod(prompts.render_plan)
    render_teach = staticmethod(prompts.render_teach)
    render_audit = staticmethod(prompts.render_audit)

    # discovery + reading
    discover = staticmethod(discovery.discover)
    read = staticmethod(reader.read_paper)


register_domain("neuro", NeuroDomain)
