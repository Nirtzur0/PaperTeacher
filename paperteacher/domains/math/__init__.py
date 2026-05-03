"""Math domain pack.

Subject coverage: pure and applied math papers from the arXiv math.*
archives (math.NT, math.AG, math.CO, math.PR, math.AP, math.DG, math.AT,
math.CT by default — users override via `arxiv_categories`), with
Semantic Scholar as a citation-velocity source.

What's math-shaped about this pack vs. physics or ml:

- The schema centers on theorems, definitions, and the move that makes
  a proof go through. Math papers don't have units to dimension-check or
  benchmarks to baseline against — they have hypotheses, conclusions,
  and a question of where each hypothesis BITES in the proof.
- A first-class `canonical_examples` list (canonical / edge / counter)
  encodes the way math is actually taught — the example shows what the
  theorem says before the abstract statement does the work.
- Theorems carry `hypotheses` with a `where_it_bites` field per
  hypothesis, plus a `sharpness` field forcing the script to address
  whether the hypotheses are necessary, the bound is tight, the converse
  holds. Skipping these is the standard failure mode of "popularized"
  math — generality and tightness are usually the entire point.
- The teach prompt's voice-first rules add math-specific anti-patterns:
  no reading quantifier strings or symbol-soup aloud, no "by abstract
  nonsense" / "trivially" / "an easy computation shows" without
  delivering the move, no claiming generality without taking the special
  case first.
- Discovery hits the math arXiv archives plus Semantic Scholar. No HF
  Daily (math papers don't surface there) and no INSPIRE (that's HEP
  proper).
"""
from __future__ import annotations

from .._common import make_domain
from . import discovery, models, prompts, reader
from ...domain import register_domain

MathDomain = make_domain(
    "math", models=models, prompts=prompts, discovery=discovery, reader=reader
)

register_domain("math", MathDomain)
