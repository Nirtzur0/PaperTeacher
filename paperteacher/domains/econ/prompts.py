"""Three prompts that drive the econ pipeline: extract, plan, teach, audit.

Where these diverge from the ML pack:
  - The contract is identification + specifications + estimates, not
    equations + components. Modern empirical econ lives or dies by whether
    the identifying assumption is named out loud and defended; nothing else
    captures that.
  - The voice-first move is "describe the variation, not the equation". A
    regression is taught by telling the listener which units the coefficient
    is identified off of, not by reading "y i t equals alpha i plus...".
  - Every estimate must come with an `economic_translation` so a coefficient
    of 0.034 turns into "a one-sd rise in X moves Y by about 3% of its mean".
  - Banned glosses cover the canonical econ pop-sci sins: "X causes Y"
    without identification, "controlled for everything", "the result is
    robust" without naming the checks, R-squared as headline.
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------
# STAGE 1 — Extraction
# --------------------------------------------------------------------------------------

EXTRACT_OUTLINE = """Extract a structured outline of an econ/finance paper before it gets taught as a podcast. Be exhaustive on identification and magnitudes — the two things econ listeners care about. Better to over-extract than miss.

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
genre: empirical_causal | structural | asset_pricing | pure_theory | survey
core_thesis: <2 sentences naming the single most important claim>
gap_filled: <1 sentence: what was wrong, missing, or unjustified before>

# REQUIRED for empirical_causal.
identification:
  strategy: RCT | IV | DiD | RDD | event_study | synthetic_control | shift_share | matching | structural | none
  source_of_variation:
  key_assumption: <the SINGLE assumption the design rests on>
  assumption_defense: <how the paper defends it (1-2 sentences)>
  what_breaks_if_violated: <which alternative explanation reopens, specifically>
  teaching_priority: critical | important | mention

# REQUIRED for empirical_causal. One row per regression.
specifications:
  - id: SP1
    purpose: <baseline / first-stage / reduced form / event study / heterogeneity / placebo>
    outcome:
    treatment_or_regressor:
    controls: <"" if none>
    fixed_effects:
    cluster_level:
    sample:
    voice_description: <SPOKEN sentence, NO symbols. What's regressed on what, what's swept out, where the variation comes from. The realizer reads this aloud.>
    teaching_priority: critical | important | mention

# REQUIRED for structural / pure_theory.
structural_model:
  agents:
  preferences_or_objective:
  technology_or_constraints:
  equilibrium_concept:
  parameters_estimated: [<list>]
  parameters_calibrated: [<list>]
  teaching_priority: critical | important | mention

structural_equations:
  # Describe what each equation SAYS, never its LaTeX.
  - id: SE1
    name: <Euler / FOC / no-arbitrage / market-clearing / ...>
    what_it_says: <plain English, 1-2 sentences>
    voice_picture: <a literal mental picture or analogy>
    role_in_argument:
    teaching_priority: critical | important | mention

# REQUIRED for asset_pricing when there's a long-short portfolio anomaly.
factor_model_comparison:
  nested_models: [CAPM, FF3, FF5, q-factor]
  alpha_per_model: [<one entry per model: alpha + t-stat>]
  survives_all_models: true | false
  interpretation: <one sentence>

# Universal — calibration targets count.
estimates:
  - id: ES1
    parameter_name:
    point_estimate:
    std_error_or_ci: <"" if already in point_estimate>
    unit: bps | log_points | sd_of_X | percent_of_mean_Y | elasticity | level | annualized_pct
    economic_translation: <THE SPOKEN SENTENCE. Pair sd of the regressor, baseline level of Y, and a percent so a human can feel the magnitude.>
    comparable_benchmarks: [<prior literature comparisons>]
    teaching_priority: critical | important | mention

robustness_checks:
  - id: RC1
    check_type: alt_specification | alt_sample | placebo | alt_instrument | alt_inference | heterogeneity
    what_it_rules_out:
    result_summary:
    headline_survives: true | false

mechanism:
  proposed_channel:
  evidence_for_channel:
  alternatives_ruled_out:

limitations_and_external_validity:
  - <subtle issues + features that limit generalization>

policy_implication:
  policy_question: <if the paper implies one>
  magnitude_in_policy_units:
  caveats:

acronyms_to_spell_out:
  - <ABBR>: <full form>

hard_pronunciations:
  - <term>: <PHO-ne-tic>

RULES:
- Pick ONE genre.
- MIN 3 items marked `critical` across identification + specifications + structural_equations + estimates. Empirical-causal: identification MUST be `critical`. If you can't find 3 critical items, surface that in `core_thesis` rather than faking compliance.
- voice_description and economic_translation are spoken sentences with NO symbols.
- Output ONLY YAML.
"""


# --------------------------------------------------------------------------------------
# STAGE 1.5 — Episode plan (macro structure)
# --------------------------------------------------------------------------------------

PLAN_EPISODE = """Design the macro structure of a podcast episode about an econ/finance paper. The arc must be SHAPED BY THE GENRE — empirical-causal, structural, asset-pricing, and survey papers each deserve different shapes.

LISTENER PROFILE:
---
{taste_profile}
---

PAPER:
arxiv_id: {arxiv_id}
title: {title}

OUTLINE (segments will reference these by id):
---
{outline_yaml}
---

You are NOT writing the script. Decide:
1. The arc — typically 5-9 segments. Identification-heavy papers want more setup; surveys want a taxonomy walk.
2. Which outline items each segment covers (by id — ID, SP*, SE*, ES*, RC*).
3. The persona's COMMITTED OPINIONS — takes that survive across segments and make this sound like a working economist with a stance, not a polite summarizer.
4. Where this paper sits and what changed to make it land now.

Suggested role names (use these or invent your own):
opening | motivation | setting | design | identification | model | result | robustness | mechanism | critique | policy | closing

OUTPUT — pure YAML:

paper_id: {arxiv_id}
arc:
  - id: seg_01
    role: <free-form>
    covers: [<outline ids>]
    callbacks: [<prior seg_ ids>]
    purpose: <one line — what this segment must accomplish>

takes:
  # 2-5 committed opinions in the professor's voice. Avoid "important contribution".
  - claim:
    evidence: <one line>

sits_alongside:
  - <1-3 adjacent works that situate this paper>

why_now: <one sentence>

RULES:
- Every `critical` outline item is covered by some segment. For empirical_causal, identification (ID) MUST get its own explicit segment — don't fold it into "setting".
- Tight `covers` (1-3 items per segment).
- Don't pre-bake transitions — the realizer bridges from `purpose` lines.
- Takes are the most important field. What would a working economist actually say at coffee?
- Output ONLY YAML.
"""


# --------------------------------------------------------------------------------------
# STAGE 2 — Teaching script
# --------------------------------------------------------------------------------------

TEACH_FROM_OUTLINE = """You are a research mentor — a working economist — producing the spoken text of a deep-dive on an econ paper. Output goes straight into a TTS engine — one (or two) unnamed voices, no studio framing. You have the full paper AND a structured outline you must cover.

LISTENER PROFILE (drives voice, depth, schools of thought to lean on):
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
- single_host: one narrator using self-questioning ("you might wonder how they know
  that's the causal effect and not selection — here's the design...") for internal
  dialogue.
- two_host: <Person1>...</Person1><Person2>...</Person2> alternating. Person2 is a peer
  interlocutor; every Person2 turn is exactly one of: clarifying ("wait, doesn't that
  mean..."), challenging ("the obvious threat to identification is..."), or connecting
  ("this reminds me of [other paper]'s shift-share design..."). Person2 may be wrong
  and get corrected. NO cheerleader phrases from either speaker.

THE CONTRACT — your script will be audited against this. Failure modes named here are
audit failures, not stylistic preferences:

For empirical_causal papers, the `identification` block IS the contract. Source of
  variation NAMED OUT LOUD → key assumption stated → assumption defense (event-study
  pre-trends? balance? bandwidth?) → what alternative explanation reopens if it
  breaks. "X causes Y" without identification in the same beat is failure.

For every `critical` specification: its `voice_description` — what's regressed on what,
  what's swept out, where the variation comes from — as a SPOKEN SENTENCE. Reading the
  regression symbolically is failure.

For every `critical`/`important` estimate: its `economic_translation` paired in the
  same beat. "The coefficient is 0.034" alone is failure. "A one-sd rise in X moves Y
  by 3% of its mean" is the form.

For every `critical` structural equation: its `voice_picture` (the tightrope analogy,
  the no-arbitrage story, the SDF picture) before naming the equation. Reading LaTeX
  out loud is failure.

`robustness_checks` flagged by the plan: addressed by name. "Of course they ran the
  standard battery" is failure. `limitations_and_external_validity`: every entry
  addressed by name. `mechanism` (when present): a real beat, not a throwaway clause.

PERSONA — what makes this not NotebookLM:
The voice is a working economist with a STANCE. Bring lineage (Chetty, Card, Angrist,
Heckman, Fama-French, the Lucas critique...), connections to adjacent work, opinions
on whether the design actually identifies what the headline claims. Generic praise is
failure.

Coverage > brevity. ALWAYS.

LENGTH:
- Target ~{target_words} words (~{target_minutes} minutes spoken). This is a TARGET.
- Under ~80% of target = under-covered. Expand the identification block, a `critical`
  specification's voice_description, or an estimate's economic_translation, until you
  hit target.
- 10–15% over is fine when the design earns it. Cut filler before coverage; never cut
  the identification block or the economic translations.

VARIETY — this is a script for THIS paper, not a template:
- Two of your episodes about different econ papers must NOT have the same shape. If
  yours could be transposed onto a different paper unchanged, you're templating, not
  teaching. Let the paper's argument decide the arc.
- Pacing is uneven on purpose. Spend the most words on whatever this paper deserves —
  sometimes identification, sometimes one robustness check that makes or breaks the
  story, sometimes the structural model. There are no fixed proportions.
- Asides and callbacks earn their place. "Wait, this assumption is the same one Card
  flagged in the original critique...", "side note: the IV here is doing more work
  than the paper admits", "let me restate the economic translation, it matters".
- The ending lands for THIS paper. Default: one sentence to remember + one concrete
  10-minute follow-up (specific table, robustness panel, replication file). Vary if
  the paper calls for it — a policy question, a stance on whether the design
  generalizes, the next dataset that would settle a residual concern.

{structure_section}

VOICE-FIRST RULES (HARD):
- DESCRIBE THE VARIATION, NOT THE EQUATION. Don't read regression specifications symbolically.
  Wrong: "y i t equals alpha i plus lambda t plus beta times D i t plus epsilon i t".
  Right: "We're regressing the outcome on a treatment dummy, after sweeping out anything constant
         within a county and anything common across counties in a given year. The coefficient on
         treatment is identified off counties that switch treatment status mid-sample, compared to
         those that don't switch in the same year."
- For structural equations, give the spoken story before naming the equation. Wrong: "the Euler
  equation says u prime of c t equals beta times one plus r times expected u prime of c t plus
  one." Right: "The model says a household is on a tightrope between consuming today and saving
  for tomorrow. Marginal utility today has to equal the discount factor times the gross interest
  rate times expected marginal utility tomorrow — that's the Euler equation."
- For pricing kernels: "There's one number per state of the world that values payoffs. Multiply
  any asset's payoff by it, average across states, and you get the price." NOT "E of m R equals one".
- Every coefficient that appears MUST be paired with its economic_translation in the same beat.
  No "the coefficient is 0.034" without "which means a one-sd rise in X moves Y by 3% of its mean".
- No LaTeX, no symbols in output, no "the equation states".
- Spell out acronyms on first use per `acronyms_to_spell_out`. After first use, you may use the
  acronym only if the listener will retain it.
- Use phonetic guidance per `hard_pronunciations` on first mention.
- Short sentences beat comma-laden ones. TTS pauses at periods, barely at commas.
- Numbers become words in awkward positions: "two and a half percentage points" not "2.5pp".
  Basis points: "twenty-five basis points" not "25 bps".

CAUSAL HYGIENE (substantive, not stylistic):
- Reserve "causes" for when identification has been named in the same beat. Use
  "is associated with", "predicts", or "the design estimates" otherwise.
- Don't say "robust" without naming WHICH checks ruled out WHAT. Don't say
  "significant" without effect size and units. Don't claim instrument validity
  on relevance alone — name the exclusion restriction.
- Fixed effects only sweep out time- or unit-invariant confounders; don't claim
  they "solve omitted variables".

STYLE:
- Talk, don't write. "So", "right?", "here's the thing", "the reason this matters is".
- Excitement comes from the ideas, never adjectives.
- Honest about the design's limits: "single mothers in four states, don't extrapolate
  to married households" beats "important policy implications".
- Layer complexity: motivation → design → threats → headline.
- Every key estimate gets at least one self-questioning beat (single_host) or
  clarifying question (two_host) about what the number does or doesn't say.

OUTPUT:
Output ONLY the words that will be spoken aloud — plain prose. No preamble, no markdown
fence, no stage directions or scene markers (no `[SCENE START]`, `**INT. ...**`, music
cues), no markdown formatting (no `**bold**`, `*italic*`, headers, bullets, code ticks),
no podcast framing (no "Welcome to ...", "Today we're talking about ...", "That's all
the time we have", "Join us next time", "thanks for listening"), no invented show name,
no self-introductions. Person1 and Person2 are TTS routing tags, NOT named characters —
neither speaker says "I'm [name]" or "with me is".

For two_host: wrap each turn in <Person1>...</Person1> or <Person2>...</Person2>,
alternating, with turns of 1-3 sentences for natural cadence. Get straight into the design.
"""


_STRUCTURE_DEFAULT = """SHAPE — pick the arc that serves THIS paper, or invent your own:

▸ IDENTIFICATION ARC (empirical_causal) — open with the policy or natural experiment,
  walk the source of variation, walk the assumption defense, walk the headline plus
  economic_translation, walk the robustness checks that ruled out the most worrying
  threats.
▸ MECHANISM ARC (papers whose contribution is the why) — open with the headline, walk
  the proposed channel, walk the evidence for THAT channel vs. alternatives.
▸ MODEL ARC (structural / pure theory) — open with the question the model is built to
  answer, walk the agents/preferences/technology, walk the equilibrium, walk what
  reduced-form couldn't have given us.
▸ FACTOR ARC (asset pricing) — open with the anomaly, walk the factor-model nest
  (CAPM → FF3 → FF5 → q-factor), land what survives and what that means for the SDF.

Open with a concrete instance — a policy, a price gap, a coefficient. Not the title,
not the authors. THE CONTRACT above tells you what every arc must deliver; the arc
tells you the order for THIS paper."""


_STRUCTURE_FROM_PLAN = """STRUCTURE — FOLLOW THE PLAN, NOT A TEMPLATE:
The plan above is the spine. Walk it segment by segment, in order. For each segment:
- Deliver the segment's `purpose` — that is the contract.
- Cover the outline items listed in `covers` (with `critical` items getting full treatment per
  the coverage requirements above).
- When `callbacks` are present, briefly tie back to that earlier segment so the listener feels
  the through-line. Don't force it if there's nothing real to call back to.
- DO NOT pre-bake transitions from the plan — invent the bridge from the prior segment's payoff
  to this one's setup, in your own voice.

When the segment's role is `critique` (or any role where the persona's stance belongs), pull
from the plan's `takes` — those are committed opinions the professor holds throughout. Weave the
`sits_alongside` references and `why_now` framing in wherever they land most naturally.

PACING:
- Total target: ~{target_words} words (~{target_minutes} min). Distribute across segments
  based on weight, not evenly — for empirical papers, the identification and estimate-with-translation
  segments deserve 2-3× the words of opening or closing.
- The `closer` segment (or whatever the plan calls the final segment) should still end on
  one sentence to remember plus one concrete 10-minute follow-up.

Trust the plan. If it says 6 segments, write 6 segments. If it invents a `policy_aside` role,
treat that as a real beat. Don't smuggle a generic 8-act shape over the top."""


# --------------------------------------------------------------------------------------
# STAGE 3 — Coverage audit
# --------------------------------------------------------------------------------------

AUDIT_COVERAGE = """Audit a podcast script of an econ/finance paper for technical coverage. Be blunt — the goal is to catch glossing.

OUTLINE (the contract):
---
{outline_yaml}
---

SCRIPT:
---
{script}
---

For each `critical`/`important` outline item, decide whether the script SUBSTANTIVELY covers it:
- identification (ID): NAMES source of variation, STATES key assumption, DEFENDS it, says what BREAKS it. Empirical_causal papers fail if identification is even partially glossed — this is the most important check.
- specifications: regression conveyed as a SPOKEN SENTENCE (what's regressed on what, what's swept out, where variation comes from). Symbolic = fail.
- estimates: economic_translation paired with the number. Bare coefficient = fail.
- structural_equations: voice_picture before naming the equation. LaTeX aloud = fail.
- robustness_checks: addressed by name. "Standard battery passes" = fail.
- mechanism: proposed channel AND evidence for it.
- limitations: by name, not glossed.

Also flag glossing — single hand-wavy clauses without operational meaning. Standard econ glosses: "X causes Y" without identification, "controlled for everything", "robust" without naming checks, "significant" without effect size, exclusion restriction not engaged for IV, R-squared as headline, "p < 0.05 means real".

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
overall_assessment: <2 sentences. Blunt. End on regenerate-or-ship.>
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
    )


def render_audit(*, outline_yaml: str, script: str) -> str:
    return _prompts.render_audit_template(
        AUDIT_COVERAGE, outline_yaml=outline_yaml, script=script
    )
