"""Physics domain pack.

Subject coverage: physics papers from the arXiv physics archives
(hep-th, hep-ph, gr-qc, astro-ph, cond-mat, quant-ph by default — users
override via `arxiv_categories` in the profile or the discover() call).

What's physics-shaped about this pack vs. the ml pack:

- The outline schema's equations carry mandatory sanity-gate fields
  (dimensional check, limiting case, symmetries, conservation law) and
  use a Fermi-style estimate in place of the ml pack's
  `numerical_walkthrough` — these are the moves a working physicist
  actually makes when reading a new equation.
- A first-class `observables_and_predictions` list separates measurable
  quantities (with units, uncertainty, falsifiability) from structural
  claims (`results_to_highlight`).
- Experimental / observational papers fill an `experimental_setup` list
  with apparatus, what-was-measured, and the dominant systematic — so
  the script can name the LIGO arms or the LHCb tracker rather than
  saying "with high precision".
- A `regime_and_assumptions` list forces every script to name its
  validity envelope, blocking the standard "in the appropriate limit"
  gloss.
- Discovery hits the physics arXiv archives plus, for HEP-family
  categories, the INSPIRE-HEP REST API for journal/conference papers
  arXiv RSS misses.
- The teach-prompt's voice-first rules add physics-specific anti-patterns:
  no reading tensor / index notation aloud, no naming generic Greek
  letters, named limits required for any "in the appropriate limit"
  phrase.
"""
from __future__ import annotations

from .._common import make_domain
from . import discovery, models, prompts, reader
from ...domain import register_domain

PhysicsDomain = make_domain(
    "physics", models=models, prompts=prompts, discovery=discovery, reader=reader
)

register_domain("physics", PhysicsDomain)
