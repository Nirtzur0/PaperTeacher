"""ML domain pack.

Subject coverage: ML / CS theory and practice papers from arXiv (cs.LG,
cs.CL, cs.AI, stat.ML, math.ST, math.OC) and HuggingFace Daily, with
Semantic Scholar as a citation-weighted source. The outline schema centers
on equations and concepts; the prompts enforce voice-first reading rules
and exhaustive equation coverage.
"""
from __future__ import annotations

from .._common import make_domain
from . import discovery, models, prompts, reader
from ...domain import register_domain

MLDomain = make_domain(
    "ml", models=models, prompts=prompts, discovery=discovery, reader=reader
)

register_domain("ml", MLDomain)
