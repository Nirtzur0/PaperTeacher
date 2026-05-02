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

EXTRACT_OUTLINE = """You are extracting the structured outline of a research paper BEFORE it gets taught as a podcast. Your job is to be exhaustive about what needs to be covered. Better to over-extract than to miss.

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

For every significant equation in the paper you MUST produce a structured decomposition. Your output is the constraint on the teaching pass — anything you mark `critical` will be covered with its full decomposition. Anything you mark `mention` gets one substantive sentence. There is no fourth option.

OUTPUT — pure YAML, no preamble, no markdown fence, no commentary:

paper_id: {arxiv_id}
type: theoretical | empirical | position | survey
core_thesis: <exactly 2 sentences naming the single most important claim>
gap_filled: <1 sentence: what was wrong, missing, or unjustified before this paper>
stake_claim: <ONE sentence committing to a stake. "If this paper is right, X
              changes." Force a position the script can defend or push against.
              Avoid generic "this advances the field" phrasing.>

key_concepts:
  - id: C1
    name: <short descriptive name>
    plain_english: <one sentence the listener could repeat at dinner>
    why_it_matters: <one sentence connecting to the listener's profile or to the field>
    first_concrete_instance: <the concrete example the script should LEAD with
                              before any abstract definition. "Concrete-then-abstract"
                              is the move shared by every world-class explainer.
                              e.g., "imagine a 3D point [1,0,0] and ask where the
                              score field points" — NOT "the score is the gradient
                              of log-density">
    teaching_priority: critical | important | mention

key_equations:
  # INCLUDE EVERY SIGNIFICANT EQUATION. If the paper has 12 numbered equations, your YAML
  # should reference all 12. You may mark trivial ones `mention`, but they must appear.
  - id: E1
    english_name: <name in plain English, not the symbolic form>
    what_it_solves: <the problem this equation solves, in plain English>
    structure_in_words: <describe the equation's shape WITHOUT symbols. e.g., "expectation of the squared difference between two gradient fields">
    components:
      # one entry per significant term. Describe ROLE, not SYMBOL.
      - role: <what this term IS — physically, geometrically, semantically. NEVER the symbol.>
        intuition: <a picture, analogy, or everyday meaning>
        what_if_removed: <what concretely changes if this term is set to zero or removed>
    key_trick: <the non-obvious insight that makes this equation work. Often "X cancels Y" or "we replaced expensive A with cheap B">
    geometric_picture: <a literal mental picture — vector fields, surfaces, flows, attention matrices, graphs. Something the listener can SEE.>
    numerical_walkthrough: <a small concrete example. e.g., "Take a 3-dimensional point [1, 0, 0]. The score at this point would be the gradient of log-density, which here equals...">
    bridge_to_next: <how this equation connects to what comes next>
    teaching_priority: critical | important | mention

results_to_highlight:
  - id: R1
    claim: <what the result says>
    what_it_demonstrates: <what this proves about the theory or method>
    why_surprising: <if applicable; otherwise omit>
    # Provenance — every reported number must come WITH context, not just a SOTA score.
    benchmark: <the dataset/eval/setting; "" if no number is involved>
    baseline: <"prior best 84.1; obvious baseline 70.0" — never a bare SOTA number.
               If the paper hides the baseline, say so.>
    variance_note: <"3 seeds, std 0.4" or "single run, no error bars reported".
                    Reproducibility research finds variance up to 90% across seeds —
                    capture what the paper shows about robustness.>
    compute: <"8x A100 for 12 hours" / "8x H100 for 3 weeks". "" if not applicable.>

# What was tried before this contribution and SPECIFICALLY why it failed.
# This drives the "naive-attempt-that-fails" pedagogical move (Karpathy's
# bigram-then-transformer arc): the listener feels the limitation before
# the contribution arrives. Without this section the contribution lands as
# arbitrary cleverness. List one per critical contribution.
prior_attempts:
  - name: <e.g., "vanilla softmax attention", "PPO with KL penalty", "...">
    what_failed: <the specific failure mode this paper addresses, in one sentence>

# Structured ablation evidence. Most papers' real story is here, not in the
# headline number. Prefer this over vague "various ablations confirm".
# Include only ablations the paper actually reports; don't fabricate.
ablations:
  - component_removed: <the variant tested; e.g., "the gating">
    metric_delta: <"drops 4.2 points on GLUE" — concrete number from the paper>
    implies: <what this reveals about which part is doing the work>

# The silent contracts under which the theory holds. Markov, i.i.d.,
# Lipschitz, full-rank, bounded reward, smoothness, etc. Naming the
# assumption AND where it would break in practice is what separates an
# honest explanation from a marketing one. Skip if the paper is purely empirical.
assumption_boundaries:
  - assumption: <e.g., "tokens are i.i.d. given the prefix">
    where_it_breaks: <e.g., "heavy-tailed code corpora or long boilerplate runs">

# Per-benchmark caveats. What the dataset/eval doesn't measure, contamination
# risks, known biases. Construct-validity matters: benchmarks rarely measure
# what their headline claims to measure.
benchmark_caveats:
  # name: caveat (one line each)
  # GLUE: heavily contaminated post-2022; tells you about memorization more than reasoning
  # MMLU: 4-choice format; captures recognition not generation

# Compute and scale envelope, in audibly grounding form. "this is a 3-week,
# 64-H100 result" is concrete; "they trained a model" is not. "" if the
# paper doesn't report it.
compute_envelope: <e.g., "64 H100s, 3 weeks, ~$120k of compute" or "">

# Common misreadings to pre-empt — canonical confusions for THIS genre of
# paper. RLHF papers, scaling-law papers, mech-interp papers each have
# their own characteristic failure modes of interpretation. Include the
# 1-3 misreadings most likely to occur if the script isn't careful.
common_misreadings:
  - <e.g., "readers conflate 'linear attention' with 'as good as softmax' — the paper is explicit it's a different operating point">
  - <e.g., "people will think the gain is from the architecture; the ablation shows it's the data filtering">

limitations_and_open_questions:
  - <subtle issues the authors acknowledge — quote or paraphrase>
  - <natural extensions worth thinking about>

banned_glosses:
  # specific phrases that, if used in the script for THIS paper, would constitute hand-waving
  - <e.g., "the loss is just MSE">
  - <e.g., "and then we apply standard techniques">
  - <e.g., "the model learns to attend to the relevant tokens">

acronyms_to_spell_out:
  # acronyms in the paper that should be expanded on first reading aloud
  - <ABBR>: <full form>

hard_pronunciations:
  # author names, foreign terms, or jargon that TTS will mangle
  - <term>: <PHO-ne-tic>

# Pre-computed substitution table for every symbol that appears in the paper.
# The TEACH stage uses this as a lookup so the script NEVER reads symbols
# aloud. Values are spoken-language descriptions of what the symbol IS in
# this specific paper (a symbol can play different roles in different papers).
# Cover greek letters, subscripted parameters (pi_theta, q_phi), notations
# like log p(x|y), composite expressions like Q K^T / sqrt(d), etc.
symbol_glossary:
  # symbol_or_expression: spoken description
  # pi_theta: "the policy"
  # log p(x|y): "the log-likelihood of x given y"
  # Q K^T / sqrt(d): "queries dotted with keys, scaled down by the square root of the head dimension"
  # nabla_x log p(x): "the score — the gradient of log-density at x"

RULES:
- Be exhaustive on equations. Do not skip the appendix if it has the cleanest proof.
- **MINIMUM 3 ITEMS MARKED `critical`** across `key_equations` + `key_concepts`. The
  `critical` tier exists to FORCE the teach stage to do full decomposition. If you
  mark everything `mention` to dodge the harder schema fields, you produce a hollow
  outline and the script collapses into an abstract paraphrase. Pick the 3-5 things
  the paper genuinely lives or dies by, and mark them `critical`. If you honestly
  cannot find 3 — the paper is too thin to teach; surface that fact in
  `core_thesis` rather than faking compliance with everything-is-mention.
- For `critical` equations every component-and-trick field is mandatory.
- For each component, describe ROLE not SYMBOL. "the model's predicted gradient field at each input point" — NOT "nabla theta of f sub theta".
- Every CRITICAL concept must have a `first_concrete_instance`. The script will lead with it.
- Every reported NUMBER in `results_to_highlight` must carry baseline + variance + compute. A SOTA number without context is a teaching failure.
- For `prior_attempts` and `assumption_boundaries`, include only what the paper actually says or what the genre obviously requires. Don't fabricate.
- `symbol_glossary` should be exhaustive enough that the teach-stage prompt can substitute every symbol it would otherwise be tempted to read aloud.
- If you don't have a clean intuition for an equation, mark it `mention` and add `note: "I don't have a clean intuition — the script should acknowledge that"`. Flag, don't fake.
- Output ONLY YAML. No prose, no markdown fence, no "Here is the outline:".
"""


# --------------------------------------------------------------------------------------
# STAGE 1.5 — Episode plan (macro structure)
# --------------------------------------------------------------------------------------

PLAN_EPISODE = """You are designing the macro structure of a podcast episode about a research paper. The structure MUST BE SHAPED BY THE PAPER — a survey paper does not get the same arc as a theory paper, and a position paper definitely doesn't get the same arc as an empirical one. Your job is to think about what THIS specific paper deserves, then commit the persona's stance about it.

LISTENER PROFILE (anchors voice, depth level, and what to lean into):
---
{taste_profile}
---

The paper's full text is intentionally NOT inlined here — the outline below carries every structurally relevant claim, equation, prior attempt, limitation, and result. Plan the arc from the outline + the listener profile.

PAPER:
arxiv_id: {arxiv_id}
title: {title}

OUTLINE (already extracted — segments will reference these by id):
---
{outline_yaml}
---

You are NOT writing the script. You are deciding:
1. The arc of segments — typically 5-9, but use your judgment. A dense theory paper might want 10; a sharp position paper might want 4.
2. Which outline items each segment covers (by id).
3. The persona's COMMITTED OPINIONS about this paper — the takes that survive across segments and make the professor sound like a real expert with a stance, not a polite summarizer.
4. Where this paper sits in the field — adjacent works, why this lands now.

ARC SHAPING — DON'T TEMPLATE:
Different papers want different arcs. A few examples (don't follow these literally — let the paper lead):
- A theory paper might open with the historical problem, spend most of the runtime building the math from prereqs, and end reflecting on what's actually new vs. notation.
- An empirical paper might spend two segments on what was wrong with prior evals before introducing the method, then dwell on results with skeptical eyes.
- A position paper has no equations to walk through — the arc IS the argument, and the critique segment is bigger than usual.
- A survey can use a "taxonomy walk" rather than a setup→payoff arc.
- A method paper with one clever trick should pace toward the trick, then reflect on what it actually bought.

Suggested role names (use these or invent your own — `worked_example`,
`historical_aside`, `comparison`, `vibes_check`, `taxonomy`,
`live_derivation`, `naive_attempt`, whatever fits):
opening | motivation | naive_attempt | prereq | setup | core | result | critique | closing

PEDAGOGY MOVE — STRONGLY PREFER:
The "naive_attempt that fails" move is the load-bearing pedagogical pattern
from Karpathy / Sanderson / distill.pub: motivate the contribution by
letting the listener feel the simpler approach break first. If the
outline's `prior_attempts` is non-empty, give it its own segment — don't
collapse "what was tried before" into a half-sentence in the setup.

OUTPUT — pure YAML, no preamble, no markdown fence:

paper_id: {arxiv_id}
arc:
  - id: seg_01
    role: <free-form. opening / motivation / naive_attempt / prereq / setup / core / result / critique / closing — or invent>
    covers: [<outline ids — eq_*, con_*, res_* — that this segment delivers>]
    callbacks: [<prior seg_ ids this segment leans on, if any>]
    purpose: <one line. what this segment must accomplish for the listener>
  # ... continue with seg_02, seg_03, ...

takes:
  # 2-5 committed opinions. The realizer will pull from these in critique segments
  # and asides — they are the persona's STANCE, not generic praise.
  #
  # AT LEAST ONE take MUST be METHODOLOGICAL — a stance about what the
  # evidence actually shows vs. what the abstract claims. Examples:
  #  - "the gain is from data filtering, not the architectural change —
  #     section 4.2 ablation makes this obvious"
  #  - "the headline number on GLUE is dominated by memorization — the
  #     OOD eval in appendix B is the honest result"
  # Without a methodological take, the critique segment will be polite
  # and the take will not survive contact with the script.
  #
  # Avoid: "this is an important contribution to the field".
  # Prefer:  "the loss term is the right primitive, but they're underselling that it
  #           only works in the high-data regime — section 4.2 ablation makes this obvious".
  - claim: <the opinion itself, in the professor's voice>
    evidence: <what in the paper or field supports it (one line)>
    appears_in: [<seg_ ids where this take should land — pick 2-3 segments,
                  not just one. A take that appears in only one segment
                  reads as an aside, not a stance.>]

sits_alongside:
  # 1-3 adjacent works or lines of work that situate this paper.
  - <e.g., "MiniLLM (Gu et al., 2023) — same KL-on-outputs intuition, no reasoning-trace alignment">

why_now: <one sentence. what changed in tooling / data / theory that made this paper possible or urgent right now>

RULES:
- Every `critical` outline item should be covered by some segment. `important` items mostly should. `mention` items can be folded in or skipped.
- A segment's `covers` should be tight — usually 1-3 outline items. A segment that covers everything covers nothing.
- `callbacks` are real narrative dependencies, not links for the sake of links. If seg_05 doesn't actually need to reference seg_02, leave it empty.
- Don't pre-bake transitions. The realizer figures out how to bridge from one segment to the next based on neighbouring `purpose` lines — that's why each episode sounds different rather than templated.
- The `takes` are the most important field. A plan with weak takes produces a polite, forgettable episode. Spend real thought on them — what would a domain expert *actually* say at coffee about this paper?
- AT LEAST one take must be methodological (not thematic), and every take should `appears_in` at least 2 segments.
- Output ONLY YAML. No prose, no markdown fence.
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

Every entry in `limitations_and_open_questions` and every paper-specific phrase in
  `banned_glosses` is addressed by name. If the outline carries a `note` for something
  you didn't fully understand, acknowledge that subtlety honestly — don't fake confidence.

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

{structure_section}

VOICE-FIRST RULES (HARD):
- NEVER read equations symbolically. Use the outline's `symbol_glossary` as
  your substitution table. If a symbol appears that isn't in the glossary,
  describe its role in one short clause the first time, then alias.
- No LaTeX, no symbols in output, no "the equation states".
- Spell out acronyms on first use per `acronyms_to_spell_out`. After first use, you may use
  the acronym only if it's a common spoken word (RNN → say "recurrent net" the first time,
  then either is fine; ELBO → say "evidence lower bound" first, then say "ELBO" — but pronounce
  it as a word "EL-bow", not letters).
- Use phonetic guidance per `hard_pronunciations`. e.g., "Schrödinger" → write "Schrödinger
  (SHRO-ding-er)" the first time it appears.
- Short sentences beat comma-laden ones. TTS pauses at periods, barely at commas.
- Numbers become words: see the NUMERICAL REWRITES section below.
- ARCHITECTURE acronyms (GQA, SwiGLU, RoPE, KV cache, MoE, FlashAttention,
  LoRA): expand on first mention, then alias. Don't introduce them mid-flow
  without naming them.

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
evidence for. "87.3 on GLUE" alone is banned. "87.3 on GLUE, up from
84.1, which is evidence that the gating actually transfers" is the form.

BANNED PHRASES (do not use any of these — they read as filler or hype):
- "delve into", "dive into", "delve", "let's explore", "navigating the landscape",
  "in the realm of", "at the intersection of", "leverage" (as a verb),
  "robust" (used as filler), "essentially", "under the hood", "at the end of the day",
  "sort of", "kind of", "let me unpack that"
- "in conclusion", "to summarize", "in this episode", "today we're going to talk about",
  "without further ado"
- "fascinating", "intriguing", "wow", "amazing", "cool", "awesome", "remarkable", "incredible"
- "great point", "great question", "Exactly!", "absolutely", "100%"
- "as is well known", "the equation simply states", "trivially", "obviously", "clearly"
- "outperforms" without the delta, "state-of-the-art" without the benchmark
- "the model learns to", "the model decides", "the model understands",
  "the model wants", "the model figures out"
- bullet-list disguised as prose: "First... Second... Third..." or "There are three reasons:
  one, ...; two, ...; three, ..."
- section headers read aloud ("Section three. Results.")
- show/podcast framing: "Welcome to", "Welcome back to", "Today we're diving into",
  "Today we're talking about", "On today's episode", "That's all the time we have",
  "Join us next time", "see you next time", "thanks for listening"
- self-introductions: "I'm Alex", "I'm Ben", "with me is", "joined by", any invented
  host name. Person1 and Person2 are TTS routing tags, NOT named characters — they
  do not introduce themselves and they have no proper names.
- invented show / podcast / column / column-name: never name the production. There
  is no show.
- Plus every paper-specific phrase listed in the outline's `banned_glosses`.

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
_STRUCTURE_DEFAULT = """STRUCTURE — flowing speech, not labelled sections. Total budget
~{target_words} words; the math section gets the most. Proportions are a guide.

1. Cold open (~5%). One surprising sentence ANCHORED IN A CONCRETE INSTANCE — a number,
   scenario, or failure mode. Not the title, not the authors. Two more sentences: the
   gap, why we should care. ("Most people think attention scales like n-squared. These
   folks made it linear without losing anything — and the reason isn't engineering,
   it's a geometric observation about what attention actually computes.")

2. Context (~10%). Just enough background that the key idea lands. If `prior_attempts`
   is non-empty, this is where the naive-thing-that-fails pattern earns its space.

3. Key idea (~12%). Plain language first, anchored in the `first_concrete_instance` from
   the outline. Could-retell-at-dinner test.

4. The math, equation by equation (~55-60% — THE MEAT). Every `critical` equation per
   its full chain (defined above in THE CONTRACT). Use `symbol_glossary` verbatim.

5. Results (~10%). Pick 2-3 from `results_to_highlight`. Numbers arrive with baseline
   AND the claim-it-is-evidence-for. The most informative `ablation` lands here, not
   buried.

6. Bigger picture (~5-7%). Where this sits, what it enables. Address every entry in
   `limitations_and_open_questions` plus the most consequential `assumption_boundaries`.

7. Closer (~5%). EXACTLY two things: one sentence to remember (sharper than the cold
   open), and one concrete 10-minute follow-up — name a specific section, figure, or
   repo. ("If you have ten minutes today, skim section 3.2 — the cancellation-lemma
   proof is the cleanest thing in the paper.")"""


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

AUDIT_COVERAGE = """You are auditing a podcast script for technical coverage AND faithfulness AND voice. Be blunt — the goal is to catch glossing, not to be encouraging.

OUTLINE (the contract the script was supposed to meet):
---
{outline_yaml}
---

SCRIPT (the actual output):
---
{script}
---

Run six independent checks. A failure on any high-severity check below
should bias toward `regenerate_with_gaps`; multiple failures or a coverage
collapse should push to `regenerate_from_scratch`.

CHECK 1 — COVERAGE:
For each `critical` and `important` item in the outline, decide whether the script SUBSTANTIVELY covers it. "Substantively" means:
- For equations: the script must convey the problem it solves, the role of each major component, the key trick, AND a concrete picture (geometric or numerical). Just naming the equation does NOT count. Saying "and they define a loss function" does NOT count.
- For concepts: the script must convey what the concept IS, why it matters, AND lead with the
  outline's `first_concrete_instance` before any abstract definition. A passing mention does NOT count.
- For limitations: the script must acknowledge them by name, not gloss with "of course there are some open questions".

CHECK 2 — GLOSSING:
Places where the script "covers" something but does so in a way the outline explicitly forbade (anything matching `banned_glosses`). Treat single-clause hand-waves as gloss, not coverage.

CHECK 3 — FAITHFULNESS / RESULT-WITH-BASELINE:
For every reported NUMBER in the script, verify it appears with (a) the
baseline or prior-best AND (b) the claim it is evidence for. A bare
"achieves 87.3 on GLUE" with no baseline and no inferential connection is
a faithfulness failure — it pretends a SOTA score is a result rather than
evidence. Flag each occurrence.

CHECK 4 — ANTHROPOMORPHISM:
Pattern-match for: "the model learns to", "the model decides", "the model
understands", "the model wants", "the model tries to", "figures out",
"realizes that". Each occurrence is a voice violation. The exception:
direct paraphrase of authors' phrasing CLEARLY marked as such ("the
authors describe this as the model 'learning to'...").

CHECK 5 — NAME-DROPPING WITHOUT OPERATIONALIZATION:
For every named technique introduced in the script (DPO, GRPO, FlashAttention,
SwiGLU, etc.), check that an operational definition appears within ONE
sentence after first mention. "They use Direct Preference Optimization"
followed by no description of what DPO does is a failure.

CHECK 6 — VOICE-FIRST:
Places where the script reads symbols aloud, includes LaTeX, retains raw
Unicode greek letters, fails to expand acronyms on first use, or breaks
the numerical-rewrite rules.

OUTPUT — pure YAML, no preamble, no markdown fence:

coverage_status: complete | partial | poor
items_missing:
  # Empty list ([]) if coverage is complete. Otherwise one entry per gap.
  - id: <e.g., E3>
    name: <english_name from outline>
    what_was_said: <verbatim quote from script if anything was said, else "not mentioned">
    what_is_missing: <specific. e.g., "the geometric picture is missing" or "the numerical walkthrough was skipped" or "the key trick was named but not explained" or "concept was defined abstractly without leading from first_concrete_instance">
    severity: critical | important
items_glossed:
  # Things technically "covered" but in a way that constitutes hand-waving.
  - id: <ID>
    quote: <the glossing phrase from the script>
    why_its_a_gloss: <e.g., "calls it 'just MSE' which the outline explicitly bans">
banned_phrases_used: [<list any banned phrases that appeared verbatim>]
voice_first_violations:
  # Symbols-aloud, LaTeX leakage, raw Greek, missed acronym-expansion, etc.
  - quote: <the offending phrase>
    why: <e.g., "reads 'sigma sub i' aloud — should describe role from symbol_glossary">
faithfulness_violations:
  # Each one is a number reported without baseline + evidential connection,
  # OR a claim that doesn't trace to an outline field. Empty list if none.
  - quote: <the offending phrase from script>
    why: <e.g., "quotes '87.3 on GLUE' with no baseline and no claim-of-evidence">
anthropomorphism_violations:
  # Pattern-match results. Empty list if none.
  - quote: <verbatim phrase>
    why: <e.g., "'the model learns to attend to' — agency belongs to the optimizer/data, not the model">
name_drop_violations:
  # Named technique without operational definition within one sentence of first mention.
  - technique: <e.g., "DPO">
    quote: <the introduction phrase>
    why: <e.g., "first mentioned at minute 4:30 with no description of the loss form">
overall_assessment: <2-3 sentences. Be blunt. End with one sentence on whether to regenerate or ship. Multiple high-severity violations or a coverage collapse → regenerate_from_scratch; localized gaps → regenerate_with_gaps; clean → ship.>
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
