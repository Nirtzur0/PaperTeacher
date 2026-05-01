"""ML domain pack.

Subject coverage: ML / CS theory and practice papers from arXiv (cs.LG,
stat.ML, math-ph) and HuggingFace Daily. The outline schema centers on
equations and concepts; the prompts enforce voice-first reading rules and
exhaustive equation coverage.
"""
from __future__ import annotations

from .._common import AuditReport, parse_audit
from . import discovery, models, prompts, reader
from ...domain import register_domain


class MLDomain:
    """The default ML domain pack."""

    name = "ml"
    OutlineModel = models.Outline

    # parsers
    parse_outline = staticmethod(models.parse_outline)
    parse_audit = staticmethod(parse_audit)

    # prompt rendering
    render_extract = staticmethod(prompts.render_extract)
    render_teach = staticmethod(prompts.render_teach)
    render_audit = staticmethod(prompts.render_audit)

    # discovery + reading
    discover = staticmethod(discovery.discover)
    read = staticmethod(reader.read_paper)


register_domain("ml", MLDomain)
