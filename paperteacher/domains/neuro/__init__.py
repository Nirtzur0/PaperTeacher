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

from .._common import make_domain
from . import discovery, models, prompts, reader
from ...domain import register_domain

NeuroDomain = make_domain(
    "neuro", models=models, prompts=prompts, discovery=discovery, reader=reader
)

register_domain("neuro", NeuroDomain)
