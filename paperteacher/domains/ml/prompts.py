"""Three-and-a-half prompts that drive the ML pipeline:
  STAGE 1   EXTRACT  - decompose every equation + concept; surface provenance.
  STAGE 1.5 PLAN     - macro arc + persona stance, opt-in.
  STAGE 2   TEACH    - write the spoken script with the outline as a contract.
  STAGE 3   AUDIT    - blunt coverage + voice-first + faithfulness audit.

The pipeline is decompose-then-execute. Stage 1 forces the model to enumerate
every equation and concept BEFORE narrative pressure makes it want to skip the
hard math. Stage 2 writes the script with explicit coverage requirements drawn
from stage 1. Stage 3 verifies and triggers regeneration if anything was glossed.

The prompts also encode pedagogy lessons distilled from world-class
explainers (3Blue1Brown, Karpathy "Zero to Hero", distill.pub, ML Street
Talk) and faithfulness research:
  - concrete-then-abstract, never the reverse
  - motivate by failing first ("here's the naive thing; here's why it breaks")
  - role-not-symbol speech, with a per-paper substitution table
  - anti-anthropomorphism ("after training, this head fires on..." not "the
    model learns to attend to...")
  - SOTA numbers must arrive with baseline + variance + compute context
  - takes must be methodological and woven across multiple segments
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------
# STAGE 1 — Extraction
# --------------------------------------------------------------------------------------

EXTRACT_OUTLINE = """Extract a structured outline of a research paper before it gets taught as a podcast. Be exhaustive — better to over-extract than miss. Anything marked `critical` gets full decomposition in the teach stage; `mention` gets one substantive sentence.

LISTENER PROFILE:
---
{taste_profile}
---

PAPER:
arxiv_id: {arxiv_id}
title: {title}
---
{paper_text}
---

OUTPUT — pure YAML, no preamble, no markdown fence:

paper_id: {arxiv_id}
type: theoretical | empirical | position | survey
core_thesis: <2 sentences naming the single most important claim>
gap_filled: <1 sentence: what was wrong, missing, or unjustified before>
stake_claim: <ONE sentence: "If this paper is right, X changes." Avoid generic praise.>

key_concepts:
  - id: C1
    name:
    plain_english: <one sentence the listener could repeat at dinner>
    why_it_matters:
    first_concrete_instance: <the example to LEAD with before any abstract definition. Concrete-then-abstract.>
    teaching_priority: critical | important | mention

key_equations:
  # INCLUDE EVERY SIGNIFICANT EQUATION. Trivial ones can be `mention` but must appear.
  - id: E1
    english_name:
    what_it_solves:
    structure_in_words: <shape WITHOUT symbols>
    components:
      # one per significant term — describe ROLE, never SYMBOL
      - role:
        intuition:
        what_if_removed:
    key_trick: <the non-obvious insight (often "X cancels Y" or "expensive A replaced by cheap B")>
    geometric_picture: <a literal mental picture the listener can SEE>
    numerical_walkthrough: <small concrete example>
    bridge_to_next:
    teaching_priority: critical | important | mention

results_to_highlight:
  - id: R1
    claim:
    what_it_demonstrates:
    why_surprising: <omit if N/A>
    benchmark: <"" if no number>
    baseline: <prior best AND obvious baseline. Never a bare SOTA. Say so if hidden.>
    variance_note: <e.g. "3 seeds, std 0.4" or "single run, no error bars">
    compute: <e.g. "8x A100 12h"; "" if N/A>

# Drives the "naive-attempt-that-fails" move. Without this, contributions land as arbitrary cleverness.
prior_attempts:
  - name:
    what_failed: <one sentence>

ablations:
  - component_removed:
    metric_delta: <concrete number from the paper>
    implies:

# Markov, i.i.d., Lipschitz, full-rank, bounded reward, smoothness, etc. Skip if purely empirical.
assumption_boundaries:
  - assumption:
    where_it_breaks:

# Per-benchmark caveats — what the eval doesn't measure, contamination, known biases.
benchmark_caveats:
  # name: caveat (one line each)

compute_envelope: <e.g. "64 H100s, 3 weeks" or "">

# 1-3 most likely confusions for THIS genre of paper.
common_misreadings:
  - <one line each>

limitations_and_open_questions:
  - <subtle issues + natural extensions>

acronyms_to_spell_out:
  - <ABBR>: <full form>

hard_pronunciations:
  - <term>: <PHO-ne-tic>

# Pre-computed substitution table. The TEACH stage uses this so the script never reads symbols aloud. Cover greek, subscripted parameters (pi_theta), notations (log p(x|y)), composite expressions (Q K^T / sqrt(d)).
symbol_glossary:
  # symbol_or_expression: spoken description

RULES:
- Be exhaustive on equations including the appendix.
- MIN 3 items marked `critical` across key_equations + key_concepts. If you can't find 3, surface that in `core_thesis` rather than faking compliance.
- For `critical` equations every component-and-trick field is mandatory.
- Components describe ROLE, not SYMBOL.
- Every critical concept has a `first_concrete_instance`.
- Every reported NUMBER carries baseline + variance + compute.
- Don't fabricate `prior_attempts` or `assumption_boundaries`.
- `symbol_glossary` exhaustive enough to substitute anything the teach stage might read aloud.
- If you don't have a clean intuition, mark `mention` with `note: "..."`. Flag, don't fake.
- Output ONLY YAML.
"""


# --------------------------------------------------------------------------------------
# STAGE 1.5 — Episode plan (macro structure)
# --------------------------------------------------------------------------------------

PLAN_EPISODE = """Design the macro structure of a podcast episode about a research paper. The arc must be SHAPED BY THE PAPER — surveys, theory, position, and empirical papers each deserve different shapes. Commit the persona's stance.

LISTENER PROFILE:
---
{taste_profile}
---

PAPER:
arxiv_id: {arxiv_id}
title: {title}

OUTLINE:
---
{outline_yaml}
---

You are NOT writing the script. Decide:
1. The arc — typically 5-9 segments, but use judgment.
2. Which outline items each segment covers (by id).
3. The persona's COMMITTED OPINIONS — takes that survive across segments.
4. Where this paper sits and what changed to make it land now.

Suggested role names (use these or invent your own):
opening | motivation | naive_attempt | prereq | setup | core | result | critique | closing

If `prior_attempts` is non-empty, give it its own segment — the "naive-attempt-that-fails" move (Karpathy/3Blue1Brown) is load-bearing.

OUTPUT — pure YAML:

paper_id: {arxiv_id}
arc:
  - id: seg_01
    role: <free-form>
    covers: [<outline ids>]
    callbacks: [<prior seg_ ids>]
    purpose: <one line>

takes:
  # 2-5 committed opinions in the professor's voice.
  # AT LEAST ONE must be METHODOLOGICAL — what the evidence actually shows vs. what the abstract claims. Without a methodological take, the critique segment will be polite. Avoid generic praise.
  - claim:
    evidence: <one line>
    appears_in: [<2-3 seg_ ids — a take in only one segment reads as an aside>]

sits_alongside:
  - <1-3 adjacent works>

why_now: <one sentence>

RULES:
- Every `critical` outline item is covered by some segment.
- Tight `covers` (1-3 items per segment).
- Don't pre-bake transitions.
- Takes are the most important field — what would a domain expert actually say at coffee?
- ≥1 methodological take; every take `appears_in` ≥ 2 segments.
- Output ONLY YAML.
"""


# --------------------------------------------------------------------------------------
# STAGE 2 — Teaching script
# --------------------------------------------------------------------------------------

# Voice-first guidance built up from TTS-quirk research (Kokoro + Vertex
# Chirp 3) and ML-jargon pronunciation conventions. Included verbatim in
# the teach prompt so the realizer has explicit rules + examples for the
# textual rewrites that prevent TTS from butchering the script.
_VOICE_GUIDE = """\
PRONUNCIATION GUIDE — apply these rewrites; neural TTS reliably mangles them:

ML acronyms (pronounce as words; spell when ambiguous):
  ELBO → "EL-bow"        VAE → "V-A-E" (letter-spell)
  LoRA → "LORE-uh"       BERT → "burt"
  GAN → "gan" (rhymes with "can")
  GPT, RLHF, DPO, PPO, TRPO, SGD, MoE → letter-spell
  ReLU → "RAY-loo"       GELU → "GEH-loo"      SiLU → "SIGH-loo"
  T-SNE → "tee-S-N-E"    UMAP → "YOU-map"      YOLO → "YO-loh"
  NeRF → "nerf"          CLIP → "clip"         DALL-E → "DAH-lee"

Forced rewrites:
  i.i.d. → "independent and identically distributed"
  e.g.   → "for example"
  i.e.   → "that is"
  et al. → "and colleagues"
  vs.    → "versus"
  w.r.t. → "with respect to"
  s.t.   → "such that"
  cf.    → "compare to"
  L2     → "L-two"       L1 → "L-one"
  KL     → "K-L"         FFN → "feed-forward network"
  SOTA   → "state of the art"

Greek letters / operators (never leave raw Unicode):
  α/β/γ/δ/θ/λ/μ/σ/ε/η/ρ/τ/φ/ψ/ω →
    alpha/beta/gamma/delta/theta/lambda/mu/sigma/epsilon/eta/rho/tau/phi/psi/omega
  ∇ → "gradient"      ∂ → "partial"      Σ → "sum"      Π → "product"

Names — phonetic guides if mentioned:
  Schmidhuber → SHMIT-hoo-ber       Hochreiter → HOKE-rye-ter
  Bengio → BEN-jee-oh                Schölkopf → SHURL-kopf
  Sutskever → SOOTS-keh-ver          Vaswani → vahs-WAH-nee
  LeCun → luh-KUN                    Krizhevsky → kriz-EFF-skee
  Wasserstein → VAH-ser-shtine       Chebyshev → CHEB-ih-shev
  Riemannian → ree-MAH-nee-un        Ricci → REE-chee
  Lévy → LAY-vee                     Itô → EE-toh
  Lipschitz → LIP-shits              Cholesky → kuh-LES-kee
  Kullback-Leibler → KULL-back LIBE-ler   Frobenius → froh-BEE-nee-us
  Fréchet → FRAY-shay                Bregman → BREG-mun

NUMERICAL REWRITES:
  1e-4 / 10^-4   → "one-times-ten-to-the-minus-four"
  82M / 7B / 1.5T → "eighty-two million" / "seven billion" / "one-point-five trillion"
  O(n log n)     → "order n log n"
  O(n²)          → "order n squared"
  p < 0.05       → "p less than zero-point-zero-five"
  95% CI         → "ninety-five percent confidence interval"
  k=5, n=1024    → "k equals five", "n equals one thousand twenty-four"
  ±2.1           → "plus or minus two-point-one"
  fp16 / bf16    → "F-P sixteen" / "B-float sixteen"
  16-bit         → "sixteen-bit"
  3×             → "three times" (not "3 X")
  layers 4–8     → "layers four through eight" (en-dash → "through")
  encoder/decoder → "encoder and decoder" (no slash for TTS)

ARCHITECTURE ACRONYMS — expand on first use, then alias is fine:
  GQA   → "grouped-query attention"
  RMSNorm → "R-M-S norm" (note: split the camel case)
  RoPE  → "rotary position embeddings"
  SwiGLU → "swiglu" (one word) or "swish-gated linear unit" first time
  FlashAttention → "flash attention"
  KV cache → "key-value cache"

SYMBOL ROLE TABLE — read symbols by their ROLE, never by their glyph. The outline's
`symbol_glossary` is the lookup for THIS paper; use those descriptions verbatim. For
any symbol not in the glossary, describe its role in one short clause first, then alias.
  pi_theta            → "the policy" (NOT "pi theta", NOT "pi sub theta")
  log p(x|y)          → "the log-likelihood of x given y"
  ∇_θ L(θ)            → "the gradient of the loss with respect to the parameters"
  Q K^T / sqrt(d)     → "queries dotted with keys, scaled by the square root of the head dimension"
  E[(...)²]           → "the expected squared difference between..."

PACING — TTS pauses at periods (~400ms), barely at commas (~150ms):
  - One idea per sentence. Max ~20 words.
  - Em-dashes for parenthetical beats — like this — not parentheses (TTS reads parentheticals flat).
  - Blank line before a hard pivot for a ~1s pause.
  - Dense math: many short sentences > one long one with semicolons.
"""


TEACH_FROM_OUTLINE = """You are a research mentor producing the spoken text of a deep-dive on a research paper. Output goes straight into a TTS engine — one (or two) unnamed voices, no studio framing. You have the full paper AND a structured outline you must cover.

LISTENER PROFILE (drives voice, depth, choice of analogies):
---
{taste_profile}
---

PAPER:
arxiv_id: {arxiv_id}
title: {title}
---
{paper_text}
---

OUTLINE (MANDATORY COVERAGE — your script will be audited against this):
---
{outline_yaml}
---
{plan_section}
DELIVERY MODE: {mode}
- single_host: one narrator using self-questioning ("you might be wondering why this term
  cancels — here's the trick...") for internal dialogue. The 3Blue1Brown / Karpathy move.
- two_host: <Person1>...</Person1><Person2>...</Person2> alternating. Person2 is a peer
  interlocutor; every Person2 turn is exactly one of: clarifying ("wait, doesn't that
  mean..."), challenging ("the obvious worry is..."), or connecting ("this reminds me
  of..."). Person2 may be wrong and get corrected — pedagogically valuable. The two
  MUST disagree substantively at least once (pull from plan `takes` when present). NO
  cheerleader phrases from either speaker; same banned list as below applies.

THE CONTRACT — your script will be audited against this. Failure modes named here are
audit failures, not stylistic preferences:

For every `critical` equation, the full chain in spoken form:
  what-it-solves → structure-in-words → role of each major term (the ROLE, never the
  symbol) → key trick → geometric picture → numerical walkthrough → bridge to next.
  Naming the equation is failure. Vibes-only intuition is failure.

For every `critical` concept: lead with its `first_concrete_instance` BEFORE any
  abstract definition. Concrete-then-abstract. The outline pre-computed the example.

For every `important` item: at minimum what-it-solves, the key trick, the connection
  to the next idea. For every `mention` item: at least one substantive sentence (no
  "and there's also some other stuff").

For every reported number: baseline AND claim-it-is-evidence-for, in the same beat.
  "87.3 on GLUE, up from 84.1, which is evidence that the gating transfers" — never
  the bare SOTA.

`prior_attempts` (when non-empty): walk through at least the first — let the listener
  feel the simpler thing breaking before the contribution arrives. Load-bearing.

`assumption_boundaries` (when present): name the one most consequential assumption
  AND where it breaks. Don't bury it in the closer.

Every entry in `limitations_and_open_questions` is addressed by name. If the outline
  carries a `note` for something you didn't fully understand, acknowledge that subtlety
  honestly — don't fake confidence.

PERSONA — what makes this not NotebookLM:
The voice is a working researcher with a STANCE. Bring lineage (where ideas come from
historically), connections to adjacent work, opinions on what's actually new vs. clever
reframing. Generic praise like "an important contribution" is failure. Every architectural
choice gets motivation within ~2 sentences of being introduced (what failed without it,
what it buys). Every named technique (DPO, FlashAttention, GQA, SwiGLU…) gets an
operational definition in plain English before its second use.

Coverage > brevity. ALWAYS.

LENGTH:
- Target ~{target_words} words (~{target_minutes} minutes spoken). This is a TARGET.
- Under ~80% of target = under-covered. Expand a `critical` item with more decomposition,
  numerical walkthrough, geometric picture, or lineage until you hit target.
- 10–15% over target is fine when the math earns it. Cut filler before coverage; never
  cut equation decomposition, `first_concrete_instance` examples, or numerical anchors.

VARIETY — this is a script for THIS paper, not a template:
- Two of your episodes about different papers must NOT have the same shape. If yours
  could be transposed onto a different paper unchanged, you're templating, not teaching.
  Let the paper's argument decide the arc.
- Pacing is uneven on purpose. Spend the most words on whatever this paper deserves —
  sometimes the math, sometimes one ablation, sometimes the historical thread. There
  are no fixed proportions.
- Asides and callbacks earn their place. "Wait, this connects back to what we said
  about the gating...", "side note: this is the same trick as in [adjacent work]",
  "let me reframe what I just said". Real talks have these; templates don't.
- The ending lands for THIS paper. Default: one sentence to remember + one concrete
  10-minute follow-up (specific section, figure, or repo). Vary if the paper calls
  for it — a question that opens up the field, an anecdote about who's working on
  this, a stance you're committing to.

{structure_section}

VOICE-FIRST RULES (HARD):
- NEVER read equations symbolically. Use the outline's `symbol_glossary` as
  your substitution table. If a symbol appears that isn't in the glossary,
  describe its role in one short clause the first time, then alias.
- No LaTeX, no symbols in output, no "the equation states".
- Spell out acronyms on first use per `acronyms_to_spell_out`. After first use, alias is fine
  if the listener will retain it. Use whichever pronunciation is natural to the field
  (letters vs. spoken-word).
- Use phonetic guidance per `hard_pronunciations` on first mention.
- Short sentences beat comma-laden ones. TTS pauses at periods, barely at commas.
- Numbers become words: see the NUMERICAL REWRITES section below.

{voice_guide_section}

ANTI-ANTHROPOMORPHISM (HARD):
- "the model learns to attend to..." → "after training, the attention
  weights concentrate on..."
- "the model decides..." → "the output distribution puts most mass on..."
- "the model understands..." → "behavior is consistent with..."
- "the model wants to..." / "tries to..." → just describe the optimization
  target or the observed behavior.
- These rewrites are not optional — anthropomorphic phrasing is the most
  common credibility-killer in ML explainers. Keep agency where it
  actually lives: the loss function, the data, the optimizer.

CONCRETE-FIRST OPENING:
The first ~90 seconds of the episode (roughly the first paragraph or
opening segment) MUST contain a concrete instance, scenario, or number.
Not "this paper proposes a method for...". Lead with the example, the
failure mode, the surprising observation. The abstract framing comes after.

RESULT-WITH-BASELINE:
Every quoted number must arrive with the baseline AND the claim it is
evidence for. "87.3 on GLUE" alone is bare. "87.3 on GLUE, up from
84.1, which is evidence that the gating actually transfers" is the form.

STYLE:
- Talk, don't write. "So", "right?", "here's the thing", "the reason this matters is".
- Excitement is allowed but must come from the IDEAS, never from adjectives.
- Honest about difficulty: "this part is genuinely subtle, let me slow down" beats
  pretending it's obvious.
- Layer complexity: simple → nuance → full picture.
- Every key equation gets at least one self-questioning beat (single_host) or
  clarifying question (two_host) so the listener doesn't drift.

OUTPUT:
Output ONLY the words that will be spoken aloud — plain prose. No preamble, no markdown
fence, no stage directions or scene markers (no `[SCENE START]`, `**INT. ...**`, music
cues), no markdown formatting (no `**bold**`, `*italic*`, headers, bullets, code ticks),
no podcast framing (no "Welcome to ...", "Today we're talking about ...", "That's all
the time we have", "Join us next time", "thanks for listening"), no invented show name,
no self-introductions. Person1 and Person2 are TTS routing tags, NOT named characters —
neither speaker says "I'm [name]" or "with me is".

For two_host: wrap each turn in <Person1>...</Person1> or <Person2>...</Person2>,
alternating, with turns of 1-3 sentences for natural cadence. Get straight into the content.
"""


# Structure block when no plan was generated — the original prescriptive arc.
# Used as a fallback so existing `extract → teach` flows keep working unchanged.
_STRUCTURE_DEFAULT = """SHAPE — pick the arc that serves THIS paper, or invent your own:

▸ MYSTERY ARC (surprising results) — open with the surprising fact, walk what the field
  expected, reveal what they did, walk the math, land the kicker.
▸ BUILD ARC (theory papers with one key construction) — open with the simpler-thing-
  that-doesn't-work, walk why, introduce the construction, walk it carefully, reflect
  on what's actually new vs. notation.
▸ DETECTIVE ARC (empirical papers) — open with the result that doesn't match priors,
  walk the ablations as evidence, arrive at the explanation, caveat what it doesn't show.
▸ TAXONOMY ARC (surveys / position papers) — open with the question the field is asking,
  walk the families of answers, land on what's actually settled vs. still contested.

Open with a concrete instance — a number, a failure mode, a phenomenon. Not the title,
not the authors. THE CONTRACT above tells you what every arc must deliver; the arc
tells you the order in which to deliver it for THIS paper."""


# Structure block when a plan IS provided — the plan IS the structure, so the
# realizer follows the arc instead of imposing the default 7-act shape. Each
# segment's `purpose` is the contract; `takes` are the persona's stance to draw
# from in critique segments and asides.
_STRUCTURE_FROM_PLAN = """STRUCTURE — FOLLOW THE PLAN, NOT A TEMPLATE:
The plan above is the spine. Walk it segment by segment, in order. For each segment:
- Deliver the segment's `purpose` — that is the contract.
- Cover the outline items listed in `covers` (with `critical` items getting full decomposition,
  per the coverage requirements above).
- When `callbacks` are present, briefly tie back to that earlier segment so the listener feels
  the through-line. Don't force it if there's nothing real to call back to.
- DO NOT pre-bake transitions from the plan — invent the bridge from the prior segment's
  payoff to this one's setup, in your own voice. This is what makes each episode sound different
  rather than templated.

When the segment's role is `critique` (or any role where the persona's stance belongs), pull
from the plan's `takes` — those are committed opinions the professor holds throughout, not
improvisations. Each take's `appears_in` lists the segments where it should land; respect
that — a take that shows up in only one segment reads as an aside, not a stance. Weave the
`sits_alongside` references and `why_now` framing in wherever they land most naturally; don't
force them into a dedicated segment unless the plan asked for one.

If the plan includes a `naive_attempt` segment, deliver the naive thing as
if you were teaching it — let the listener feel why it works at first, then
let it break. The contribution arriving as a response to that breakage is
the whole point of the segment.

PACING:
- Total target: ~{target_words} words (~{target_minutes} min). Distribute across segments
  based on weight, not evenly — a `core` segment carrying a critical equation deserves 2-3×
  the words of an `opening` segment.
- The `closer` segment (or whatever the plan calls the final segment) should still end on
  one sentence to remember plus one concrete 10-minute follow-up — these earn their place
  in every episode regardless of arc.

Trust the plan. If it says 6 segments, write 6 segments. If it invents a `historical_aside`
role, treat that as a real beat. Don't smuggle the default 7-act shape in over the top."""


# --------------------------------------------------------------------------------------
# STAGE 3 — Coverage audit
# --------------------------------------------------------------------------------------

AUDIT_COVERAGE = """Audit a podcast script for technical coverage AND faithfulness AND voice. Be blunt — catch glossing.

OUTLINE (the contract):
---
{outline_yaml}
---

SCRIPT:
---
{script}
---

Six independent checks. Any high-severity failure → `regenerate_with_gaps`; multiple failures or coverage collapse → `regenerate_from_scratch`.

CHECK 1 — COVERAGE:
For each `critical`/`important` outline item, is it SUBSTANTIVELY covered?
- equations: problem-it-solves + role of each major component + key trick + a concrete picture (geometric or numerical). Just naming = fail. "And they define a loss" = fail.
- concepts: what it IS + why it matters + leads with `first_concrete_instance`. Passing mention = fail.
- limitations: by name, not "of course there are open questions".

CHECK 2 — GLOSSING:
Single hand-wavy clauses without operational meaning ("and then we apply standard techniques", "the model learns to attend to the right thing", "it's just MSE").

CHECK 3 — FAITHFULNESS / RESULT-WITH-BASELINE:
Every reported NUMBER must appear with (a) baseline or prior-best AND (b) the claim it's evidence for. Bare "87.3 on GLUE" = fail.

CHECK 4 — ANTHROPOMORPHISM:
"the model learns/decides/understands/wants/tries/figures out/realizes". Each = a violation. Exception: clearly-marked author-paraphrase.

CHECK 5 — NAME-DROPPING WITHOUT OPERATIONALIZATION:
Every named technique (DPO, GRPO, FlashAttention, …) needs an operational definition within ~one sentence of first mention.

CHECK 6 — VOICE-FIRST:
Symbols-aloud, LaTeX leakage, raw Greek, missed acronym expansion, broken numerical-rewrite rules.

OUTPUT — pure YAML:

coverage_status: complete | partial | poor
items_missing:
  - id:
    name:
    what_was_said: <verbatim quote, or "not mentioned">
    what_is_missing: <specific>
    severity: critical | important
items_glossed:
  - id:
    quote:
    why_its_a_gloss:
voice_first_violations:
  - quote:
    why:
faithfulness_violations:
  - quote:
    why:
anthropomorphism_violations:
  - quote:
    why:
name_drop_violations:
  - technique:
    quote:
    why:
overall_assessment: <2-3 sentences. Blunt. End on regenerate-or-ship.>
recommendation: ship | regenerate_with_gaps | regenerate_from_scratch
"""


# --------------------------------------------------------------------------------------
# Render helpers — thin wrappers over the shared scaffolding in _prompts.
# --------------------------------------------------------------------------------------

from .. import _prompts


def render_extract(*, arxiv_id: str, title: str, taste_profile: str, paper_text: str) -> str:
    return _prompts.render_extract_template(
        EXTRACT_OUTLINE,
        arxiv_id=arxiv_id,
        title=title,
        taste_profile=taste_profile,
        paper_text=paper_text,
    )


def render_plan(
    *,
    arxiv_id: str,
    title: str,
    taste_profile: str,
    outline_yaml: str,
) -> str:
    return _prompts.render_plan_template(
        PLAN_EPISODE,
        arxiv_id=arxiv_id,
        title=title,
        taste_profile=taste_profile,
        outline_yaml=outline_yaml,
    )


def render_teach(
    *,
    arxiv_id: str,
    title: str,
    taste_profile: str,
    paper_text: str,
    outline_yaml: str,
    mode: str = "single_host",
    plan_yaml: str | None = None,
    target_words: int | None = None,
    target_minutes: int | None = None,
) -> str:
    return _prompts.render_teach_template(
        TEACH_FROM_OUTLINE,
        structure_default=_STRUCTURE_DEFAULT,
        structure_from_plan=_STRUCTURE_FROM_PLAN,
        arxiv_id=arxiv_id,
        title=title,
        taste_profile=taste_profile,
        paper_text=paper_text,
        outline_yaml=outline_yaml,
        mode=mode,
        plan_yaml=plan_yaml,
        target_words=target_words,
        target_minutes=target_minutes,
        voice_guide=_VOICE_GUIDE,
    )


def render_audit(*, outline_yaml: str, script: str) -> str:
    return _prompts.render_audit_template(
        AUDIT_COVERAGE, outline_yaml=outline_yaml, script=script
    )
