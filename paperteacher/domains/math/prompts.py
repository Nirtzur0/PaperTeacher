"""Three+1 prompts for the math pipeline: extract outline, plan episode,
teach from outline, audit coverage.

Same decompose-then-execute shape as the ml and physics packs, but every
stage is tuned for math-paper failure modes. The killer ones are:

  - Reading symbol-soup or quantifier strings aloud ("for all epsilon
    greater than zero there exists a delta") instead of describing what
    the statement SAYS.
  - Stating a theorem without naming where each hypothesis BITES in the
    proof — the listener can't tell which hypothesis is load-bearing.
  - "By abstract nonsense" / "trivially" / "an easy computation shows" /
    "a routine diagram chase" — these are the canonical hand-waves and
    they read as filler to working mathematicians.
  - Quoting a famous theorem (Riemann-Roch, Stone-Weierstrass, Sard) as
    a black box without reminding the listener what the theorem says.
  - Stating a result in maximum generality without first showing the
    canonical example or a tractable special case (dimension 1, abelian
    case, n=2). Math is taught by examples; the abstract statement comes
    after.
  - Skipping sharpness — "the bound is tight"  vs. just "we get an upper
    bound". Whether a result can be improved is usually the entire point.
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------
# STAGE 1 — Extraction
# --------------------------------------------------------------------------------------

EXTRACT_OUTLINE = """Extract a structured outline of a math research paper before it gets taught as a podcast. Be exhaustive — better to over-extract than miss. The outline is the coverage contract for the teach stage. Anything marked `critical` will be covered with full decomposition; `mention` gets one substantive sentence.

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

MATH-SPECIFIC EXTRACTION RULES:
- For each `critical` theorem: list every hypothesis and, for each, where in the proof it BITES (which step it makes go through). Hand-waving "the theorem holds under standard hypotheses" hides the entire content of math.
- For each `critical` theorem: address `sharpness` — is the bound tight, does the converse hold, can any hypothesis be dropped? Generality and tightness are usually the entire point of a math paper.
- For each `critical` definition: include the `canonical_example` that makes the abstract object land. Definitions in math hide all the work; the example shows what's actually being defined.
- Include at least one `canonical_examples` entry for any paper that admits a tractable example (dimension 1, abelian case, n=2, smooth case). Math is taught by examples; if the script can't lead with one, listeners can't anchor the abstract statement.
- Don't fabricate `historical_context` or `conjectures_referenced` — leave them empty if the paper doesn't engage with them.

OUTPUT — pure YAML, no preamble, no markdown fence:

paper_id: {arxiv_id}
type: theory | applied | survey | computational | open_problem
core_thesis: <2 sentences naming the single most important claim>
gap_filled: <1 sentence: what was wrong, missing, or unjustified before>

historical_context:
  # 1-3 short threads placing the work in tradition. Skip for routine technical
  # results without a clear lineage. Empty list ([]) is fine.
  - <one line each>

key_definitions:
  - id: D1
    name: <the term being defined>
    plain_english: <one sentence the listener could repeat at dinner>
    why_it_matters: <one sentence>
    canonical_example: <the example that makes the abstract definition land>
    teaching_priority: critical | important | mention

key_theorems:
  # The centerpiece objects of the paper. Every numbered theorem/proposition
  # /lemma the headline depends on belongs here.
  - id: T1
    name: <name from the paper, or a descriptive label>
    hypotheses:
      # one entry per assumption the theorem rests on
      - statement: <plain English of the assumption>
        where_it_bites: <which step in the proof this hypothesis makes go through>
    conclusion: <plain English of what the theorem concludes>
    sharpness: <is the bound tight? does the converse hold? can any hypothesis be dropped? "the constant 1/4 is sharp; equality holds for the Heisenberg saturator" — mandatory for `critical` items>
    key_idea_of_proof: <the ONE substantive move (not the bookkeeping). e.g. "construct an auxiliary function f and show it's both subharmonic and bounded — Liouville does the rest">
    role_in_paper: <main theorem | reduction lemma | corollary used in section 5 | ...>
    historical_thread: <one line if the result sits in a real lineage; omit otherwise>
    teaching_priority: critical | important | mention

key_concepts:
  - id: C1
    name:
    plain_english:
    why_it_matters:
    historical_thread: <omit if no clear lineage>
    teaching_priority: critical | important | mention

key_equations:
  # For math papers these are functional equations, identities, inequalities,
  # recursions. No dimensional check; describe what each piece IS.
  - id: E1
    english_name:
    what_it_says: <plain English of what the equation asserts>
    structure_in_words: <shape WITHOUT symbols>
    components:
      - role: <what this term IS — never the symbol>
        intuition: <a picture or analogy>
        what_if_removed: <what changes if this term is dropped>
    key_trick: <the non-obvious insight>
    worked_specialization: <the equation taken in a tractable special case. e.g. "in dimension 1, the recursion collapses to Fibonacci"; "for the abelian group case, the formula reduces to a count">
    teaching_priority: critical | important | mention

canonical_examples:
  # Math is taught by examples. The canonical one shows the typical case;
  # the edge / counter shows where things break or where hypotheses bite.
  - id: X1
    kind: canonical | edge_case | counterexample
    description:
    what_it_illuminates:

conjectures_referenced:
  # Open problems or named conjectures the paper engages with. Empty list
  # ([]) for self-contained technical work that doesn't reach toward conjectures.
  - <one line each>

limitations_and_open_questions:
  - <subtle issues acknowledged + natural extensions>

acronyms_to_spell_out:
  - <ABBR>: <full form>

hard_pronunciations:
  # Author names, foreign terms, jargon TTS will mangle.
  - <term>: <PHO-ne-tic>

RULES:
- Be exhaustive on theorems and definitions. Don't skip a lemma in the appendix that's actually doing the work.
- MIN 3 items marked `critical` across `key_theorems` + `key_definitions` + `key_concepts`. If you can't find 3 critical items, surface that in `core_thesis` rather than faking compliance.
- For `critical` theorems, every hypothesis must include `where_it_bites`, and `sharpness` is mandatory.
- For `critical` definitions, `canonical_example` is mandatory.
- For each component of an equation, describe ROLE not SYMBOL.
- If you don't have a clean intuition for a theorem or proof, mark it `mention` and add `note: "I don't have a clean intuition — the script should acknowledge that"`. Flag, don't fake.
- Output ONLY YAML.
"""


# --------------------------------------------------------------------------------------
# STAGE 1.5 — Episode plan (macro structure)
# --------------------------------------------------------------------------------------

PLAN_EPISODE = """Design the macro structure of a podcast episode about a math paper. The arc must be SHAPED BY THE PAPER — a number-theory result doesn't get the same arc as a category-theory construction, and a survey doesn't get the same arc as a single-theorem paper. Commit the persona's stance.

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
1. The arc — typically 5-9 segments, but use judgment. A dense theorem with a long proof might want 8-10; a survey can run longer with more taxonomy beats.
2. Which outline items each segment covers (by id — D*, T*, C*, E*, X*).
3. The persona's COMMITTED OPINIONS — takes that survive across segments and make this sound like a working mathematician with a stance, not a polite summarizer.
4. Where this paper sits and what changed (technique, computation, adjacent fields) to make it land now.

Suggested role names (use these or invent your own):
opening | motivation | tradition | definitions | example_first | theorem_statement | proof_sketch | hypothesis_bite | sharpness | counterexample | comparison | critique | closer

Strongly prefer an `example_first` segment before any `theorem_statement` segment when the outline has a `canonical` example. Math is taught by examples; the abstract statement lands much better when the listener has already seen the typical case.

OUTPUT — pure YAML:

paper_id: {arxiv_id}
arc:
  - id: seg_01
    role: <free-form>
    covers: [<outline ids>]
    callbacks: [<prior seg_ ids>]
    purpose: <one line — what this segment must accomplish>

takes:
  # 2-5 committed opinions in the professor's voice. Math takes are often
  # about generality vs. cleanness ("stated in maximum generality but only
  # the smooth case is used"), or about whether the contribution is the
  # result or the technique ("the technique is the contribution; the
  # headline theorem was already accessible to anyone willing to grind").
  # Avoid generic "important contribution" framing.
  - claim:
    evidence: <one line>

sits_alongside:
  - <1-3 adjacent works or paradigms — for math this often crosses decades>

why_now: <one sentence>

RULES:
- Every `critical` outline item is covered by some segment.
- Tight `covers` (1-3 items per segment).
- Don't pre-bake transitions — the realizer bridges from `purpose` lines.
- The `takes` are the most important field. What would a working mathematician actually say at coffee about this paper?
- Output ONLY YAML.
"""


# --------------------------------------------------------------------------------------
# STAGE 2 — Teaching script
# --------------------------------------------------------------------------------------

TEACH_FROM_OUTLINE = """You are a research mentor — a working mathematician — producing the spoken text of a deep-dive on a math paper. Output goes straight into a TTS engine — one (or two) unnamed voices, no studio framing. You have the full paper AND a structured outline you must cover.

LISTENER PROFILE (drives voice, depth, choice of math tradition):
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
- single_host: one narrator using self-questioning ("you might wonder why this hypothesis is necessary — here's where it bites...") for internal dialogue. The 3Blue1Brown / Tao-blog move.
- two_host: <Person1>...</Person1><Person2>...</Person2> alternating. Person2 is a peer interlocutor; every Person2 turn is exactly one of: clarifying ("wait, does that hold without the smoothness assumption?"), challenging ("the obvious worry is that this needs choice — does it?"), or connecting ("this reminds me of the Erdős-Ko-Rado proof in the chromatic setting..."). Person2 may be wrong and get corrected — pedagogically valuable, especially for missed hypotheses. NO cheerleader phrases from either speaker.

THE CONTRACT — your script will be audited against this. Failure modes named here are audit failures, not stylistic preferences:

For every `critical` theorem, the full chain in spoken form (in this order):
  motivating example or special case → hypotheses each named in plain English → conclusion in plain English → key idea of proof (the substantive move, not the bookkeeping) → for each load-bearing hypothesis, where it BITES in the proof → sharpness (is the bound tight? converse? necessity of the hypotheses?) → bridge to next. Stating the theorem is failure. "By abstract nonsense" without naming the categorical move is failure. Quoting a hypothesis without saying where it's used is failure.

For every `critical` definition: lead with the `canonical_example` BEFORE the abstract definition. Concrete-then-abstract. The outline pre-computed the example.

For every `important` item: at minimum the plain-English statement, the key idea, and one example or one hypothesis-bite.

For every `mention` item: at least one substantive sentence (no "and there's also some other stuff").

`canonical_examples`: every entry the plan references gets walked through, not just named.

`limitations_and_open_questions` and `conjectures_referenced`: every entry addressed by name.

If the outline carries a `note` for something you didn't fully understand, acknowledge that subtlety honestly — don't fake confidence.

PERSONA — what makes this not NotebookLM:
The voice is a working mathematician with a STANCE. Bring lineage (Erdős, Cantor, Riemann, Wiles, Tao, Perelman, Grothendieck, Connes, Bourbaki — math traditions span centuries), connections to adjacent work, opinions on whether the contribution is the result or the technique, opinions on whether the maximum generality is honest or cosmetic. Generic praise like "an important contribution" is failure.

Coverage > brevity. ALWAYS.

LENGTH:
- Target ~{target_words} words (~{target_minutes} minutes spoken). This is a TARGET.
- Under ~80% of target = under-covered. Expand a `critical` theorem with more on where each hypothesis bites, walk one more example, walk a worked specialization (dimension 1, abelian case, n=2), or take the sharpness question further until you hit target.
- 10–15% over target is fine when the math earns it. Cut filler before coverage; never cut the canonical example, the hypothesis-bite walk, or the sharpness discussion.

VARIETY — this is a script for THIS paper, not a template:
- Two of your episodes about different math papers must NOT have the same shape. If yours could be transposed onto a different paper unchanged, you're templating, not teaching. Let the paper's argument decide the arc.
- Pacing is uneven on purpose. Spend the most words on whatever this paper deserves — sometimes the proof's key move, sometimes a single counterexample that reframes everything, sometimes the lineage. There are no fixed proportions.
- Asides and callbacks earn their place. "Wait, this is the same trick we saw in the lemma — let me reuse it...", "side note: this is what Erdős would have done with the probabilistic method", "let me restate the hypothesis, it matters for what comes next".
- The ending lands for THIS paper. Default: one sentence to remember + one concrete 10-minute follow-up (specific section, an example to chase down, a related paper). Vary if the paper calls for it — a question that the next theorem would settle, a stance about whether the generality buys anything.

{structure_section}

VOICE-FIRST RULES (HARD — math edition):
- NEVER read symbol strings or quantifier strings aloud. Don't say "for all epsilon greater than zero there exists a delta", "f sub n converges to f", "X intersect Y is contained in Z". Describe what the statement SAYS — "the function converges uniformly", "every neighborhood of the limit eventually contains the tail of the sequence", "the intersection is contained in the smaller set, which is what gives us the bound".
- NEVER read Greek letters as letters unless that's genuinely the proper name of a thing. Otherwise describe what the symbol stands for ("the regularity exponent" not "alpha"; "the small parameter" not "epsilon").
- No LaTeX, no symbols in output, no "the equation states".
- Spell out acronyms on first use per `acronyms_to_spell_out`.
- Use phonetic guidance per `hard_pronunciations` on first mention.
- Numbers become words in awkward positions. "Two-thirds" not "2/3". "Order n log n" not "O of n log n" said symbolically.
- Short sentences beat comma-laden ones. TTS pauses at periods, barely at commas.

MATH-SPECIFIC TEACHING MOVES (use these — this is what makes a math episode sound like math rather than re-skinned popular science):
- Lead with the canonical example before the abstract definition. "Before I give you the formal definition, here's what this looks like in dimension 1 / for the integers / when the group is abelian. Now the abstract statement is going to feel like the obvious generalization."
- Name where each hypothesis BITES. "The compactness assumption shows up here — without it, the maximizer might fail to exist and the whole argument collapses." "Smoothness gets used exactly once, in the integration-by-parts step."
- Address sharpness explicitly. "Is the constant tight? Yes — equality is achieved for the Heisenberg saturator." Or: "The hypothesis that f is bounded looks technical, but actually you can drop it — the proof just needs a truncation argument."
- Take a worked specialization. "Let's see what this says when n=2. The recursion collapses, and what we get is just Fibonacci — that's the toy case the theorem generalizes."
- Walk a counterexample if there is one. "What if we drop the openness assumption? The classical counterexample is the rationals in the reals — no nontrivial connected open subset, and the whole theorem fails."

MATH HYGIENE (substantive, not stylistic):
- Don't say "by abstract nonsense" without naming the categorical move (Yoneda, adjoint functor theorem, universal property).
- Don't say "trivially", "obviously", "an easy computation shows", "a routine diagram chase" without delivering the move in plain English.
- Don't quote a famous theorem (Riemann-Roch, Stone-Weierstrass, Sard, Hahn-Banach) without a one-sentence reminder of what it says.
- Don't say "WLOG" without noting why we may assume the special case (rescaling, symmetry, density).
- Don't claim "the proof is essentially the same as in the prior work" without naming what changed and why it matters.

STYLE:
- Talk, don't write. "So", "right?", "here's the thing", "the reason this matters is".
- Excitement comes from the IDEAS, never adjectives.
- Honest about difficulty: "the diagonal argument is genuinely subtle here, let me slow down" beats pretending it's obvious.
- Layer complexity: example → statement → proof sketch → sharpness.
- Every key theorem gets at least one self-questioning beat (single_host) or clarifying question (two_host) so the listener doesn't drift.

OUTPUT:
Output ONLY the words that will be spoken aloud — plain prose. No preamble, no markdown fence, no stage directions or scene markers (no `[SCENE START]`, `**INT. ...**`, music cues), no markdown formatting (no `**bold**`, `*italic*`, headers, bullets, code ticks), no podcast framing (no "Welcome to ...", "Today we're talking about ...", "That's all the time we have", "Join us next time", "thanks for listening"), no invented show name, no self-introductions. Person1 and Person2 are TTS routing tags, NOT named characters — neither speaker says "I'm [name]" or "with me is".

For two_host: wrap each turn in <Person1>...</Person1> or <Person2>...</Person2>, alternating, with turns of 1-3 sentences for natural cadence. Get straight into the math.
"""


_STRUCTURE_DEFAULT = """SHAPE — pick the arc that serves THIS paper, or invent your own:

▸ EXAMPLE-FIRST ARC (most theorems with a canonical example) — open with the
  example, walk what's special about it, generalize to the theorem statement,
  walk the key proof move, address sharpness, end on what the example does
  NOT cover.
▸ HYPOTHESIS-BITE ARC (theorems where the strength of the result lies in
  the weakest hypothesis set) — open with a stronger version that's "easy",
  then walk how each hypothesis can be relaxed and what breaks.
▸ COUNTEREXAMPLE ARC (papers built around a surprising counterexample) —
  open with the conjecture or expectation, walk why it seemed plausible,
  reveal the counterexample, reflect on what this tells us about the
  surrounding theory.
▸ TAXONOMY ARC (surveys, position papers) — open with the question the
  field is asking, walk the families of approaches, land on what's
  actually settled vs. still contested.

Open with a concrete instance — an example, a number, a small case. Not
the title, not the authors. THE CONTRACT above tells you what every arc
must deliver; the arc tells you the order for THIS paper."""


_STRUCTURE_FROM_PLAN = """STRUCTURE — FOLLOW THE PLAN, NOT A TEMPLATE:
The plan above is the spine. Walk it segment by segment, in order. For each segment:
- Deliver the segment's `purpose` — that is the contract.
- Cover the outline items listed in `covers` (with `critical` items getting full decomposition: example-first, hypothesis-bite for each load-bearing hypothesis, sharpness, key idea of proof, per the coverage requirements above).
- When `callbacks` are present, briefly tie back to that earlier segment so the listener feels the through-line. Don't force it if there's nothing real to call back to.
- DO NOT pre-bake transitions from the plan — invent the bridge from the prior segment's payoff to this one's setup, in your own voice.

When the segment's role is `critique` (or any role where the persona's stance belongs), pull from the plan's `takes` — those are committed opinions the professor holds throughout. Weave the `sits_alongside` references and `why_now` framing in wherever they land most naturally; math traditions often span decades or centuries, so a one-line callback to Cantor or to Grothendieck's relative point of view can earn its place anywhere.

If the plan includes an `example_first` segment, deliver the example as if you were teaching it — work it out concretely, then let the abstract statement feel like the obvious generalization. The contribution lands much better after the listener has already seen the typical case.

PACING:
- Total target: ~{target_words} words (~{target_minutes} min). Distribute across segments based on weight — a `proof_sketch` segment for a critical theorem deserves 2-3× the words of an opening. A `sharpness` segment can be short but mustn't be skipped.
- The closer (whatever the plan calls it) should still end on one sentence to remember plus one concrete 10-minute follow-up.

Trust the plan. If it says 6 segments, write 6 segments. If it invents a `hypothesis_bite` or `counterexample` role as its own beat, treat it as a real beat. Don't smuggle a generic shape in over the top."""


# --------------------------------------------------------------------------------------
# STAGE 3 — Coverage audit
# --------------------------------------------------------------------------------------

AUDIT_COVERAGE = """Audit a math podcast script for technical coverage. Be blunt — the goal is to catch glossing.

OUTLINE (the contract):
---
{outline_yaml}
---

SCRIPT:
---
{script}
---

For each `critical`/`important` outline item, decide whether the script SUBSTANTIVELY covers it:
- theorems: hypotheses named in plain English + conclusion in plain English + key idea of proof + (for `critical`) where each load-bearing hypothesis BITES in the proof + sharpness. Stating the theorem is NOT enough. "By abstract nonsense" without naming the categorical move is a fail. Quoting a hypothesis without saying where it's used is a fail.
- definitions: leads with the `canonical_example` before the abstract statement. Defining abstractly first = fail.
- equations: structure-in-words + role of each major component + key trick + worked specialization (in dimension 1, abelian case, n=2, smooth case). Just naming the equation = fail.
- concepts: what it IS + why it matters. Passing mention = fail.
- canonical_examples (when the plan references them): each gets walked through, not just named.
- limitations / conjectures: by name, not glossed.

Also flag math-specific glossing:
- "by abstract nonsense" without naming the categorical move
- "trivially" / "obviously" / "an easy computation shows" / "a routine diagram chase" without delivering the move
- a famous theorem (Riemann-Roch, Stone-Weierstrass, Sard, Hahn-Banach) quoted without a one-sentence reminder of what it says
- "WLOG" without noting why we may assume the special case
- "the proof is essentially the same as in the prior work" without naming what changed
- maximum generality stated without a worked special case

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
