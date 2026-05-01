"""Econ / finance domain pack.

Subject coverage: economics + quantitative finance papers from arXiv
(econ.GN, econ.TH, econ.EM, q-fin.*) and NBER's new-working-papers RSS.
The outline schema is built around the conventions of modern empirical
econ — identification strategy, specifications, estimates with explicit
economic-magnitude translations, robustness checks — plus structural-model
fields for theory papers and a factor-model nest for asset pricing. The
prompts enforce voice-first rules tuned to econ glosses ("X causes Y"
without identification, "controlled for everything", "the result is robust"
without naming the checks).
"""
from __future__ import annotations

from .._common import AuditReport, parse_audit
from . import discovery, models, prompts, reader
from ...domain import register_domain


class EconDomain:
    """The econ / finance domain pack."""

    name = "econ"
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


register_domain("econ", EconDomain)
