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

EXTRACT_OUTLINE = """You are extracting the structured outline of an economics or finance research paper BEFORE it gets taught as a podcast. Your job is to be exhaustive about identification and magnitudes — the two things careful econ listeners care about most. Better to over-extract than to miss.

LISTENER PROFILE (so you know what to flag for extra explanation):
---
{taste_profile}
---

PAPER:
arxiv_id: {arxiv_id}
title: {title}
---
{paper_text}
---

Modern empirical econ papers live or die by the identifying assumption. Theory papers live or die by the model's economic content. Asset-pricing papers live or die by whether the alpha survives the standard factor models. The outline must surface whichever applies.

OUTPUT — pure YAML, no preamble, no markdown fence, no commentary:

paper_id: {arxiv_id}
genre: empirical_causal | structural | asset_pricing | pure_theory | survey
core_thesis: <exactly 2 sentences naming the single most important claim>
gap_filled: <1 sentence: what was wrong, missing, or unjustified before this paper>

# REQUIRED for genre=empirical_causal. Omit otherwise.
identification:
  strategy: RCT | IV | DiD | RDD | event_study | synthetic_control | shift_share | matching | structural | none
  source_of_variation: <where the variation comes from. e.g. "the staggered rollout of Medicaid expansion across states 2014–2020">
  key_assumption: <the SINGLE assumption the design rests on. e.g. "parallel trends between expansion and non-expansion states absent treatment">
  assumption_defense: <how the paper defends it. event-study pre-trends? balance tests? bandwidth sensitivity? 1-2 sentences>
  what_breaks_if_violated: <what alternative explanation re-opens. be specific — "selection on time-varying state characteristics correlated with expansion timing">
  teaching_priority: critical | important | mention

# REQUIRED for genre=empirical_causal. Each row is one regression actually run.
specifications:
  - id: SP1
    purpose: <baseline / first-stage IV / reduced form / event study / heterogeneity / placebo>
    outcome: <plain English. e.g. "log of household quarterly consumption (CEX)">
    treatment_or_regressor: <plain English. e.g. "indicator for living in an expansion state after expansion takes effect">
    controls: <names of controls; "" if none>
    fixed_effects: <e.g. "state and year-by-quarter">
    cluster_level: <e.g. "state">
    sample: <e.g. "households with income below 138% FPL, 2010–2022">
    voice_description: <THIS is what the realizer will read. Write it as a spoken sentence with NO symbols. "We're regressing log consumption on a treatment dummy, sweeping out anything constant within a state and anything common across states in a given quarter, with errors clustered at the state level. The coefficient on treatment is identified off the staggered timing of expansion across states.">
    teaching_priority: critical | important | mention

# REQUIRED for genre=structural OR genre=pure_theory.
structural_model:
  agents: <who's in the model. "households, firms, a government">
  preferences_or_objective: <"CRRA utility over consumption with discount factor beta">
  technology_or_constraints: <"Cobb-Douglas production with capital adjustment costs">
  equilibrium_concept: <"recursive competitive equilibrium" / "Markov perfect" / "rational expectations">
  parameters_estimated: [<list>]
  parameters_calibrated: [<list>]
  teaching_priority: critical | important | mention

structural_equations:
  # The economic equations of the model — Euler, FOC, no-arbitrage, market-clearing.
  # NEVER write the LaTeX form; describe what the equation SAYS.
  - id: SE1
    name: <"Euler equation", "no-arbitrage condition", "intratemporal FOC">
    what_it_says: <plain English, 1-2 sentences. "Marginal utility today equals the gross return times the discount factor times expected marginal utility tomorrow.">
    voice_picture: <a literal mental picture or analogy. "Think of a person on a tightrope between consuming today and saving for tomorrow — the equation says the rope tightens at the rate of interest.">
    role_in_argument: <what this equation does for the paper. "This is the moment condition we estimate against in the GMM step.">
    teaching_priority: critical | important | mention

# REQUIRED for genre=asset_pricing when the paper documents/explains a portfolio anomaly.
factor_model_comparison:
  nested_models: [CAPM, FF3, FF5, q-factor]
  alpha_per_model: ["0.42% (t=2.8)", "0.36% (t=2.3)", "0.29% (t=1.9)", "0.18% (t=1.1)"]
  survives_all_models: false
  interpretation: <one sentence. "the alpha fades by half against q-factor, suggesting investment + profitability explain most of it">

# Universal. Every paper has at least one estimate to report (a calibration target counts).
estimates:
  - id: ES1
    parameter_name: <"ATT", "minimum-wage employment elasticity", "Sharpe ratio of the long-short portfolio", "discount factor beta">
    point_estimate: <"0.034 (0.012)" — keep parens for SE if reported that way; or "-0.07">
    std_error_or_ci: <"" if already in point_estimate; else the SE or 95% CI>
    unit: bps | log_points | sd_of_X | percent_of_mean_Y | elasticity | level | annualized_pct
    economic_translation: <THE SPOKEN SENTENCE. Always pair (a) sd of the regressor, (b) baseline level of Y, (c) headline percent. "A one-sd ($350) rise in EITC generosity raises labor-force participation by 0.6 percentage points — about 4% of the sample baseline.">
    comparable_benchmarks: [<prior literature comparisons. "Eissa & Liebman 1996 found ~2.4pp; this is roughly a quarter of that magnitude.">]
    teaching_priority: critical | important | mention

# Modern empirical papers run 4-10 of these. List the ones the headline depends on.
robustness_checks:
  - id: RC1
    check_type: alt_specification | alt_sample | placebo | alt_instrument | alt_inference | heterogeneity
    what_it_rules_out: <which alternative explanation. "selection on time-varying state characteristics">
    result_summary: <"coefficient drops by 8% but remains significant at the 5% level">
    headline_survives: true | false

mechanism:
  proposed_channel: <"liquidity rather than wealth — the effect is concentrated among households with no precautionary savings">
  evidence_for_channel: <"heterogeneity is monotone in pre-period liquid assets; effect is null in the top quartile">
  alternatives_ruled_out: <"a wealth-effect story would predict the opposite gradient">

limitations_and_external_validity:
  - <subtle issues the authors acknowledge. quote or paraphrase>
  - <population / time / institutional features that limit generalization>

policy_implication:
  policy_question: <if the paper implies one. e.g. "should the EITC be expanded?">
  magnitude_in_policy_units: <"a $1k EITC top-up raises participation by ~0.4pp on average; equivalent to ~120k workers nationally">
  caveats: <"effects are estimated for single mothers; extrapolation to married households is unsupported">

banned_glosses:
  # specific phrases that, if used in the script for THIS paper, would constitute hand-waving.
  # MUST INCLUDE: any "X causes Y" without naming identification, any "robust" without naming the checks
  - <e.g., "the paper shows that minimum wage raises unemployment">
  - <e.g., "they controlled for everything">

acronyms_to_spell_out:
  # spell out as letters: OLS, GLS, GMM, 2SLS, IV, RDD, ATE, ATT, LATE, IRF, VAR, SVAR, VECM,
  # SDF, HJM, FF3, FF5, FOMC, BLS, BEA, NIPA, GDP, CPI, PPI, MPC, MPS, EIS, CRRA, CES.
  # Pronounce as words: GARCH, Tobit, Probit, Logit, CAPM (cap-em).
  # DiD: prefer "diff-in-diff" not "D-i-D".
  - <ABBR>: <full form>

hard_pronunciations:
  # author or method names TTS will mangle. Examples:
  #   Acemoğlu: AH-jeh-MOH-loo
  #   Piketty:  pee-keh-TEE
  #   Bénassy-Quéré: beh-nah-SEE keh-RAY
  #   Modigliani: moh-deel-YAH-nee
  #   Mirrlees: MIR-leez
  #   Mankiw: MAN-kew
  #   Schumpeter: SHOOM-pay-ter
  #   Bartik: BAR-tik
  #   Engle (ARCH/GARCH): ENG-gul
  - <term>: <PHO-ne-tic>

RULES:
- Pick ONE genre. If a paper bridges (e.g. structural + reduced form), pick the one where the contribution lives — usually the one in the abstract's lead sentence — and put the secondary content into the matching field anyway.
- For empirical_causal papers: identification is the most important field. If the paper doesn't make it explicit, NAME WHAT THE PAPER IS IMPLICITLY ASSUMING and flag it as critical.
- For each specification, voice_description must be a spoken sentence with NO symbols. Describe what's regressed on what, what's swept out, and where the variation comes from.
- For each estimate, economic_translation must convert the number into something a human can feel — pair the sd of the regressor with the baseline level of the outcome and a percent.
- For asset-pricing papers, fill factor_model_comparison whenever there's a long-short portfolio with alphas across factor models.
- Output ONLY YAML. No prose, no markdown fence, no "Here is the outline:".
"""


# --------------------------------------------------------------------------------------
# STAGE 1.5 — Episode plan (macro structure)
# --------------------------------------------------------------------------------------

PLAN_EPISODE = """You are designing the macro structure of a podcast episode about an economics or finance paper. The structure MUST BE SHAPED BY THE PAPER's GENRE — an empirical causal-inference paper does not get the same arc as an asset-pricing paper, and a theory paper definitely doesn't get the same arc as either.

The listener's taste profile is available at `profile://taste` — your host already loaded it; do not expect it inlined here.

PAPER:
arxiv_id: {arxiv_id}
title: {title}

The paper's full text is intentionally NOT inlined — the outline below carries every claim, equation, prior attempt, limitation, and result the planner needs. Plan the arc from the outline.

OUTLINE (already extracted — segments will reference these by id):
---
{outline_yaml}
---

You are NOT writing the script. You are deciding:
1. The arc of segments — typically 5-9. Identification-heavy papers want more setup; survey papers want a taxonomy walk.
2. Which outline items each segment covers (by id — SP*, SE*, ES*, RC*, ID).
3. The persona's COMMITTED OPINIONS — the takes that survive across segments and make the professor sound like an actual economist with a stance, not a polite summarizer.
4. Where this paper sits in the field — adjacent works, why this lands now.

ARC SHAPING BY GENRE — DON'T TEMPLATE:
- empirical_causal: spend a real segment on identification (often two — the design + the threats). Then specifications + headline estimate. Then robustness as one segment, not a montage. Then mechanism. Critique should engage with what the design CAN'T identify.
- structural / pure_theory: walk the model's economic content first (agents → preferences → technology → equilibrium), then the equations the model lives in, then what the model gets you that reduced-form can't.
- asset_pricing: motivate the anomaly, walk through the factor-model nest (CAPM → FF3 → FF5 → q-factor), reflect on what the surviving alpha implies. Often a "what's the SDF picture" detour is worth one segment.
- survey: taxonomy walk; each segment = a sub-literature. The persona's stance comes from "what's actually settled" vs "what's still contested".

Suggested role names (use these or invent your own — `setting`, `design`, `identification`, `first_stage`, `headline`, `robustness`, `mechanism`, `model`, `counterfactual`, `policy`, `critique`, whatever fits):
opening | motivation | setting | design | identification | model | result | robustness | mechanism | critique | policy | closing

OUTPUT — pure YAML, no preamble, no markdown fence:

paper_id: {arxiv_id}
arc:
  - id: seg_01
    role: <free-form. opening / setting / design / identification / model / result / robustness / mechanism / critique / closing — or invent>
    covers: [<outline ids — ID, SP*, SE*, ES*, RC* — that this segment delivers>]
    callbacks: [<prior seg_ ids this segment leans on, if any>]
    purpose: <one line. what this segment must accomplish for the listener>
  # ... continue with seg_02, seg_03, ...

takes:
  # 2-5 committed opinions. The realizer pulls from these in critique and asides.
  # Avoid: "this is an important contribution to the literature".
  # Prefer:  "the design is clever but the population is narrow — single mothers in
  #           four states. The headline number won't generalize, and they undersell that."
  - claim: <the opinion in the professor's voice>
    evidence: <what in the paper or field supports it (one line)>

sits_alongside:
  # 1-3 adjacent works or lines of work that situate this paper.
  - <e.g., "Chetty et al. on tax salience — same identification flavor (exposure-based DiD), different domain">

why_now: <one sentence. what changed in tooling / data / theory that made this paper possible or urgent right now>

RULES:
- Every `critical` outline item should be covered by some segment. `important` items mostly should. `mention` items can be folded in or skipped.
- A segment's `covers` should be tight — usually 1-3 outline items. A segment that covers everything covers nothing.
- For empirical_causal papers, the `identification` outline item (id: ID) MUST be covered explicitly by some segment. Don't fold it into "setting".
- Don't pre-bake transitions. The realizer figures out how to bridge from one segment to the next based on neighbouring `purpose` lines.
- The `takes` are the most important field. A plan with weak takes produces a polite, forgettable episode. What would a working economist actually say at coffee about this paper?
- Output ONLY YAML.
"""


# --------------------------------------------------------------------------------------
# STAGE 2 — Teaching script
# --------------------------------------------------------------------------------------

TEACH_FROM_OUTLINE = """You are a research mentor — a working economist — recording a podcast episode about a research paper. You have the full paper AND a structured outline of everything that must be covered. Produce the spoken script.

The listener's taste profile is available at `profile://taste` — your host already loaded it; do not expect it inlined here.

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
- single_host: one narrator. Use self-questioning to create internal dialogue:
    "Now you might be wondering — how do they know that's the causal effect and not selection? Here's the design..."
- two_host:    output as <Person1>...</Person1><Person2>...</Person2> tags, alternating.
                Person2 is a peer-level interlocutor — NEVER a cheerleader. Person2's questions
                must be exactly one of these types:
                  - clarifying:  "wait, doesn't that mean..."
                  - challenging: "the obvious threat to identification is..."
                  - connecting:  "this reminds me of the [other paper]'s shift-share design..."
                Person2 may occasionally be wrong and corrected by Person1 — pedagogically valuable.

COVERAGE REQUIREMENTS (NON-NEGOTIABLE):
1. The `identification` block (when the genre is empirical_causal) MUST appear with: the source of variation NAMED OUT LOUD, the key assumption stated, the assumption defense, and what would break if the assumption fails. Skipping any of these constitutes a critical gloss.
2. Every `critical` specification MUST appear via its `voice_description`, NOT its symbols. Describe the regression as a sentence the listener can hold without pen and paper.
3. Every `critical` and `important` estimate MUST appear with its `economic_translation` — a coefficient is never the headline; the translated sentence ("a one-sd rise raises Y by ~3% of the mean") is.
4. Every `critical` structural equation MUST appear with its `voice_picture` — the geometric or story analog, not the symbols.
5. Every entry in `robustness_checks` flagged by the plan must be addressed by name. "And of course they ran the standard battery" does NOT count.
6. Every entry in `limitations_and_external_validity` must be addressed by name.
7. The `mechanism` (when present) gets a real beat, not a throwaway clause.

LENGTH:
- Target ~{target_words} words (~{target_minutes} minutes spoken).
- That word count is a CEILING on padding, not a floor on rambling. Cut anything that does not earn its place.
- If you cannot cover all `critical` items in the target, cut adjective density and meta-commentary.
  NEVER cut the identification block or the economic translations of estimates. Coverage > brevity.

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
- Use phonetic guidance per `hard_pronunciations`. e.g., "Acemoğlu (AH-jeh-MOH-loo)" the first time.
- Short sentences beat comma-laden ones. TTS pauses at periods, barely at commas.
- Numbers become words in awkward positions: "two and a half percentage points" not "2.5pp".
  Basis points: "twenty-five basis points" not "25 bps".

BANNED PHRASES (do not use any of these — they read as filler or as the canonical econ glosses):
- "delve into", "dive into", "let's explore", "navigating the landscape",
  "in the realm of", "at the intersection of"
- "in conclusion", "to summarize", "in this episode", "today we're going to talk about"
- "fascinating", "intriguing", "wow", "amazing", "cool", "awesome", "remarkable", "incredible"
- "as is well known", "trivially", "obviously", "clearly"
- "this paper shows X causes Y" — ONLY say "causes" when identification has been named in the
  same beat. Use "is associated with", "predicts", or "the design estimates" otherwise.
- "they controlled for everything" — controls don't fix endogeneity; this conflates OLS-with-controls
  and identified causal effects.
- "the result is robust" — without naming WHICH checks were run and WHAT they ruled out.
- "the coefficient is significant" — without effect size and units.
- "the instrument is valid because it's correlated with X" — relevance is the easy condition;
  the exclusion restriction is the hard one. Never gloss the exclusion restriction.
- "fixed effects solve omitted variables" — they only sweep out time-invariant or unit-invariant
  confounders.
- "p less than 0.05 means the effect is real" — misstates frequentist inference.
- "the R-squared was X, so the model fits well" — R-squared is almost never the headline in
  causal work.
- "this is the first paper to..." — almost always false; reviewers hate it.
- bullet-list disguised as prose ("First... Second... Third...")
- section headers read aloud ("Section three. Results.")
- Plus every paper-specific phrase listed in the outline's `banned_glosses`.

STYLE:
- Talk, don't write. "So", "right?", "here's the thing", "the reason this matters is".
- Earn excitement through the ideas, never adjectives.
- Honest about the design's limits. "Single mothers in four states — don't extrapolate to
  married households" beats "important policy implications".
- Layer complexity: motivation first, then the design, then the threats, then the headline.
- For every key estimate, include a self-questioning beat (single_host) or one clarifying
  question (two_host) about what the number does or doesn't say.

OUTPUT:
Output ONLY the script. No preamble, no headers, no stage directions, no markdown fence.
Just the words a TTS would speak.
For two_host: wrap each turn in <Person1>...</Person1> or <Person2>...</Person2>, alternating,
with turns of 1-3 sentences for natural cadence.
"""


_STRUCTURE_DEFAULT = """STRUCTURE (write as flowing speech, not as labelled sections).
The proportions below are a guide, not a contract — total budget is ~{target_words} words.
Allocate the actual word count yourself; the identification + estimates segments get the most.

For empirical_causal papers — recommended shape:
1. Cold open (~5%). One surprising sentence. Lead with the FACT or RESULT, not the title or authors.
   Example: "When two states sit on either side of the Mason-Dixon line and one of them expands
   Medicaid in 2014, the other doesn't, what happens to consumption smoothing? These authors looked,
   and the answer reorganizes how we think about the income effect."
2. Setting + question (~10%). What's the policy or natural experiment, what's the outcome we care about.
3. Identification (~20% — THIS IS THE CONTRACT). Name the source of variation. State the key assumption.
   Defend it the way the paper defends it. Say what would break it.
4. Specifications + headline estimate (~25%). Walk the baseline regression as a SPOKEN SENTENCE
   (voice_description, not symbols). Give the headline coefficient AND the economic_translation
   in the same beat.
5. Robustness (~12%). Pick the 2-3 checks that ruled out the most worrying threats. Don't list all of them.
6. Mechanism (~10%). The "why". The proposed channel + evidence + alternatives ruled out.
7. Bigger picture + limits (~10%). External validity, where this sits, what's still open.
8. Closer (~5%). EXACTLY two things:
     - One sentence to remember.
     - One concrete 10-minute follow-up: "if you have ten minutes, read Table 5 — the placebo
       on pre-2010 outcomes is the cleanest part of the paper."

For structural / pure_theory papers — collapse identification + robustness, expand structural
equations + counterfactuals.

For asset_pricing papers — replace identification with the factor-model nest walk; replace
robustness with sample-period and turnover stress-tests.

For survey papers — taxonomy walk: each segment = one sub-literature."""


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

AUDIT_COVERAGE = """You are auditing a podcast script of an economics or finance paper for technical coverage. Be blunt — the goal is to catch glossing, not to be encouraging.

OUTLINE (the contract the script was supposed to meet):
---
{outline_yaml}
---

SCRIPT (the actual output):
---
{script}
---

For each `critical` and `important` item in the outline, decide whether the script SUBSTANTIVELY covers it. "Substantively" means:

- For `identification` (id: ID): the script must NAME the source of variation, STATE the key assumption, DEFEND it, and SAY what would break it. Skipping any of those is a critical gap. Note: empirical_causal papers fail the audit if identification is even partially glossed; this is the single most important check in the econ pack.
- For `specifications`: the script must convey the regression as a SPOKEN SENTENCE — what's regressed on what, what's swept out, where the variation comes from. Naming the equation symbolically does NOT count.
- For `estimates`: the script must convey the economic_translation of the coefficient, not just the raw number. "The coefficient is 0.034" alone is a critical gap.
- For `structural_equations`: the script must convey the voice_picture — the geometric or story analog. Reading the LaTeX out loud does NOT count.
- For `robustness_checks`: the script must address the named checks specifically. "And the standard robustness checks pass" does NOT count.
- For `mechanism`: the script must lay out the proposed channel AND the evidence for it.
- For `limitations_and_external_validity`: must be addressed by name, not glossed.

Also check for glossing — places where the script "covers" something but does so in a way the outline explicitly forbade. The standard econ glosses to flag:
- "X causes Y" without identification named in the same beat
- "controlled for everything"
- "the result is robust" without naming the checks
- "the coefficient is significant" without effect size
- exclusion restriction not engaged with for any IV claim
- R-squared treated as the headline
- "p less than 0.05 means the effect is real"

OUTPUT — pure YAML, no preamble, no markdown fence:

coverage_status: complete | partial | poor
items_missing:
  # Empty list ([]) if coverage is complete. Otherwise one entry per gap.
  - id: <e.g., ID, SP1, ES2, SE1, RC3>
    name: <name from outline (parameter_name / strategy / etc.)>
    what_was_said: <verbatim quote from script if anything was said, else "not mentioned">
    what_is_missing: <specific. e.g., "key_assumption named but assumption_defense skipped" or "economic_translation missing — only the raw coefficient was read">
    severity: critical | important
items_glossed:
  # Things technically "covered" but in a way that constitutes hand-waving.
  - id: <ID>
    quote: <the glossing phrase from the script>
    why_its_a_gloss: <e.g., "uses 'X causes Y' without naming the identification strategy in the same beat">
banned_phrases_used: [<list any banned phrases that appeared verbatim>]
voice_first_violations:
  # Places where the script reads symbols aloud, includes LaTeX, or otherwise breaks voice-first rules.
  - quote: <the offending phrase>
    why: <e.g., "reads 'y i t equals alpha plus beta D' aloud — should describe variation, not symbols">
overall_assessment: <2 sentences. Be blunt. End with one sentence on whether to regenerate or ship.>
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
    outline_yaml: str,
    taste_profile: str | None = None,  # back-compat; not inlined
    paper_text: str | None = None,     # back-compat; not inlined
) -> str:
    return _prompts.render_plan_template(
        PLAN_EPISODE,
        arxiv_id=arxiv_id,
        title=title,
        taste_profile=taste_profile,
        paper_text=paper_text,
        outline_yaml=outline_yaml,
    )


def render_teach(
    *,
    arxiv_id: str,
    title: str,
    paper_text: str,
    taste_profile: str | None = None,  # back-compat; not inlined
    inline_voice_guide: bool = True,
    domain_name: str | None = None,
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
