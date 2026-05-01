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

from .._common import make_domain
from . import discovery, models, prompts, reader
from ...domain import register_domain

EconDomain = make_domain(
    "econ", models=models, prompts=prompts, discovery=discovery, reader=reader
)

register_domain("econ", EconDomain)
