"""Three+1 prompts for the physics pipeline: extract outline, plan episode,
teach from outline, audit coverage.

Same decompose-then-execute shape as the ml pack, but every stage is tuned
for physics-paper failure modes. The killer ones are:

  - Reading tensor / index notation aloud ("g sub mu nu times g upper mu
    nu") instead of describing what each piece IS.
  - Glossing over dimensional analysis. A physicist's first move on any
    new equation is "do the units balance?", and a script that skips it
    sounds like cargo-cult math.
  - Skipping the limiting case. Almost every interesting physics equation
    reduces to something already understood (Newton, Galileo, classical
    EM, BCS) in some limit; if the script doesn't take the limit, the
    listener can't anchor the new thing to what they already know.
  - Hand-waving with "natural", "of order one", "by symmetry", "in the
    appropriate limit" without naming the symmetry, the scale, or the
    limit. These read as filler to working physicists.
  - Confusing prediction with measurement. A theory paper proposes
    observables; an experimental paper measures them. The script should
    be honest about which side of that line it's on.

The audit stage hunts for exactly these failure modes.
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------
# STAGE 1 — Extraction
# --------------------------------------------------------------------------------------

EXTRACT_OUTLINE = """You are extracting the structured outline of a physics research paper BEFORE it gets taught as a podcast. Be exhaustive about what needs to be covered. Better to over-extract than to miss — the teaching pass uses your output as a coverage contract.

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

For every significant equation in the paper you MUST produce a structured decomposition. Anything you mark `critical` will be covered with its full decomposition in the script. Anything you mark `mention` gets one substantive sentence. There is no fourth option.

PHYSICS-SPECIFIC EXTRACTION RULES:
- Every `critical` equation MUST have a non-empty `dimensional_check`, `limiting_case`, and either a `symmetries` list or a `conservation_law` (or both). Skipping these is the standard failure mode of physics popularization.
- For each component of an equation, describe ROLE not SYMBOL. "the four-momentum of the outgoing particle" — NEVER "p sub mu". "the Higgs field's vacuum expectation value" — NEVER "phi sub zero".
- Predictions live in `observables_and_predictions`, not in `results_to_highlight`. An observable is a quantity an experiment can measure, with units and (where the paper quotes one) an uncertainty.
- For an experimental or observational paper, fill `experimental_setup` — the apparatus or campaign, what it actually measured, the dominant systematic. Theoretical / phenomenological papers leave this list empty.
- For each new concept, set `historical_thread` if the concept genuinely sits inside a tradition (gauge theory, BCS, the Standard Model, ΛCDM). Skip it for genuinely new ones — don't manufacture lineage.

OUTPUT — pure YAML, no preamble, no markdown fence, no commentary:

paper_id: {arxiv_id}
type: theoretical | phenomenological | experimental | observational | computational | review
core_thesis: <exactly 2 sentences naming the single most important claim>
gap_filled: <1 sentence: what was wrong, missing, or unjustified before this paper>

historical_context:
  # 1-3 short threads placing the work in tradition. Skip for routine cond-mat
  # measurements where the lineage isn't load-bearing. Empty list ([]) is fine.
  - <e.g., "extends the running of the Standard Model couplings — the same Wilsonian flow Wilson formalized in the 1970s">

regime_and_assumptions:
  # The validity envelope of the whole paper. Naming these up front makes the
  # script say "valid in the X limit" instead of hand-waving with "appropriate".
  - <e.g., "non-relativistic v ≪ c">
  - <e.g., "weak coupling, perturbative in α_s">
  - <e.g., "Markov approximation on the bath">

key_concepts:
  - id: C1
    name: <short descriptive name>
    plain_english: <one sentence the listener could repeat at dinner>
    why_it_matters: <one sentence connecting to the listener's profile or to the field>
    teaching_priority: critical | important | mention
    historical_thread: <one line if the concept sits in a real lineage; omit otherwise>

key_equations:
  # INCLUDE EVERY SIGNIFICANT EQUATION. If the paper has 12 numbered equations,
  # your YAML should reference all 12. You may mark trivial ones `mention`, but
  # they must appear.
  - id: E1
    english_name: <name in plain English, not the symbolic form>
    what_it_solves: <the problem this equation solves, in plain English>
    structure_in_words: <describe the equation's shape WITHOUT symbols. e.g., "balance between a kinetic term that wants to spread the wave function out and a potential that wants to localize it">
    components:
      # one entry per significant term. Describe ROLE, not SYMBOL.
      - role: <what this term IS — physically, geometrically. NEVER the symbol.>
        intuition: <a physical picture, analogy, or everyday meaning>
        what_if_removed: <what physics breaks if this term is set to zero>

    # PHYSICS SANITY GATES — mandatory for `critical`, recommended for `important`:
    dimensional_check: <units balance, in plain words. e.g., "left side is energy density (energy per unit volume); right side is field-tensor squared, which carries the same dimensions in natural units once ℏc is reinstated">
    symmetries:
      - <e.g., "Lorentz invariant under boosts and rotations">
      - <e.g., "gauge invariant under U(1)">
      - <e.g., "parity-odd — flips sign under spatial inversion">
    conservation_law: <Noether: which symmetry → which conserved quantity. e.g., "time-translation symmetry of the action gives energy conservation". Empty if not applicable.>
    limiting_case: <a regime where this reduces to something already understood. e.g., "in the v/c → 0 limit, the kinetic term recovers (1/2)mv² — Newton drops out cleanly">

    key_trick: <the non-obvious insight. Often "X cancels Y" or "we replaced expensive A with cheap B" or "the symmetry forces this term to vanish so only Z survives">
    geometric_picture: <a literal mental picture — fields, worldlines, phase-space orbits, level sets, a sphere on a saddle, gauge bundle. Something the listener can SEE.>
    fermi_estimate: <a first-principles order-of-magnitude estimate. e.g., "for a star-mass black hole, M ≈ 10³⁰ kg gives a Schwarzschild radius around 3 km — within a factor of two of textbook values">
    bridge_to_next: <how this equation connects to what comes next>
    teaching_priority: critical | important | mention

observables_and_predictions:
  # What the paper says you could (or did) measure. Empty list ([]) for purely
  # formal / mathematical papers.
  - id: O1
    name: <e.g., "branching ratio of B → K μ⁺μ⁻", "shift in the CMB TT spectrum near ℓ ≈ 200">
    predicted_value: <numerical value with uncertainty if quoted; free-form so units and asymmetric error bars all fit>
    how_measured: <which apparatus or dataset would reach this. e.g., "LHCb full Run 3 luminosity">
    falsifiability: <one sentence: what value or non-detection would rule it out?>

experimental_setup:
  # For experimental / observational papers only. Theoretical papers leave [].
  - id: X1
    apparatus: <e.g., "ATLAS detector, Run 3 (139 fb⁻¹)", "JWST NIRSpec G395M">
    what_is_measured: <one line>
    key_systematic: <dominant uncertainty, in physics terms — not "various sources">

results_to_highlight:
  # Claims that aren't observables — derived bounds, no-go theorems, scaling
  # laws, structural results.
  - id: R1
    claim: <what the result says>
    what_it_demonstrates: <what this proves about the theory>
    why_surprising: <if applicable; otherwise omit>

limitations_and_open_questions:
  - <subtle issues the authors acknowledge — quote or paraphrase>
  - <natural extensions worth thinking about>

banned_glosses:
  # Specific phrases that, if used in the script for THIS paper, would be hand-waving.
  - <e.g., "by symmetry, this term vanishes" — without naming WHICH symmetry>
  - <e.g., "the standard machinery">
  - <e.g., "in the appropriate limit" — without naming the limit>
  - <e.g., "of order one" — without saying in units of what>

acronyms_to_spell_out:
  # First-use expansions for TTS. Common physics ones: QCD, QED, EFT, RG, EW,
  # CMB, ΛCDM, NMSSM, MSSM, LIGO, JWST, etc.
  - <ABBR>: <full form>

hard_pronunciations:
  # Author names, foreign terms, jargon TTS will mangle. Physics is full of
  # these: Schrödinger, Wess-Zumino-Witten, Ehrenfest, Bogoliubov, Calabi-Yau.
  - <term>: <PHO-ne-tic>

RULES:
- Be exhaustive on equations. Don't skip the appendix if it has the cleanest derivation.
- For `critical` equations every field above is mandatory, including dimensional_check, limiting_case, and at least one of (symmetries, conservation_law).
- For each component, describe ROLE not SYMBOL. "the gauge field that mediates the strong force" — NOT "A sub mu sup a".
- If you don't have a clean intuition for an equation, mark it `mention` and add `note: "I don't have a clean intuition — the script should acknowledge that"`. Flag, don't fake.
- Output ONLY YAML. No prose, no markdown fence, no "Here is the outline:".
"""


# --------------------------------------------------------------------------------------
# STAGE 1.5 — Episode plan (macro structure)
# --------------------------------------------------------------------------------------

PLAN_EPISODE = """You are designing the macro structure of a podcast episode about a physics research paper. The structure MUST BE SHAPED BY THE PAPER — a string-theory derivation does not get the same arc as an ATLAS measurement, and an LIGO event paper definitely doesn't get the same arc as a hep-th formalism paper. Decide what THIS specific paper deserves, then commit the persona's stance about it.

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
1. The arc of segments — typically 5-9, but use your judgment. A dense formalism paper might want 10; a single-result observational paper might want 4.
2. Which outline items each segment covers (by id).
3. The persona's COMMITTED OPINIONS about this paper — the takes that survive across segments.
4. Where this paper sits in the field — adjacent works/paradigms, why this lands now.

ARC SHAPING — DON'T TEMPLATE:
Different physics papers want different arcs. Examples (don't follow literally — let the paper lead):

- A theory / formalism paper (hep-th, math-ph) might open with the puzzle that motivates the new construction, walk derivation step by step with a dimensional-analysis beat between hard moves, take a limiting case explicitly to show it recovers known physics, and end on what new computations the formalism enables.
- A phenomenology paper (hep-ph, astro-ph.CO) might spend two segments on the regime/assumptions before any equation, then dwell on observables and what would falsify the prediction.
- An experimental / observational paper (hep-ex, astro-ph.HE, LIGO papers) earns time on the apparatus and the dominant systematic — listeners often don't know how the measurement is even possible. Then the result, then comparison to theory.
- A condensed-matter measurement paper (cond-mat) often has one striking signature (a kink in resistivity, an unexpected peak in ARPES) — pace toward that, then discuss the candidate theories that could explain it.
- A review / position paper has no single equation to walk through — the arc IS the taxonomy or the argument, and the critique segment is bigger than usual.

Suggested role names (use these or invent your own):
opening | motivation | tradition | regime_setup | derivation | dimensional_check | limiting_case | symmetry_argument | prediction | experimental_status | comparison | critique | closer

USE THE SANITY-GATE ROLES SPARINGLY: a `dimensional_check` or `limiting_case` segment is appropriate when the move is genuinely the heart of the paper — otherwise these checks are a beat inside a `derivation` segment, not their own segment.

OUTPUT — pure YAML, no preamble, no markdown fence:

paper_id: {arxiv_id}
arc:
  - id: seg_01
    role: <free-form. opening / motivation / tradition / regime_setup / derivation / dimensional_check / limiting_case / symmetry_argument / prediction / experimental_status / comparison / critique / closer — or invent>
    covers: [<outline ids — eq_*, con_*, obs_*, exp_*, res_* — that this segment delivers>]
    callbacks: [<prior seg_ ids this segment leans on, if any>]
    purpose: <one line. what this segment must accomplish for the listener>
  # ... continue with seg_02, seg_03, ...

takes:
  # 2-5 committed opinions. The realizer pulls from these in critique segments
  # and asides — they are the persona's STANCE, not generic praise.
  # Avoid: "this is an important contribution to the field".
  # Prefer: "the bound they derive is sharp but the assumption set is doing a lot
  #          of work — table 2 of arXiv:24XX.YYYYY shows the same scaling under a
  #          much weaker assumption, which they don't cite".
  # Or:     "the discrepancy with the lattice result is two sigma — interesting
  #          but not a tension yet, and the systematic on the lattice side is
  #          probably underestimated for these matrix elements".
  - claim: <the opinion itself, in the professor's voice>
    evidence: <what in the paper or field supports it (one line)>

sits_alongside:
  # 1-3 adjacent works or paradigms. For physics this often crosses decades.
  # Examples: "Bekenstein-Hawking black-hole entropy", "the Higgs mechanism as
  # formulated in 1964", "BCS theory of superconductivity". Pair with a recent
  # arXiv adjacent work where useful.
  - <e.g., "Hawking radiation calculation (1975) — same semiclassical limit, this paper sharpens the back-reaction term">

why_now: <one sentence. what changed in apparatus, computing, or theory that made this paper possible or urgent right now. e.g., "JWST early-release data is finally precise enough at z>10 to test models people had on the shelf for a decade">

RULES:
- Every `critical` outline item should be covered by some segment. `important` items mostly should. `mention` items can be folded in or skipped.
- A segment's `covers` should be tight — usually 1-3 outline items. A segment that covers everything covers nothing.
- For `critical` equations, the segment that covers them is responsible for the full decomposition (dimensional_check, limiting_case, symmetries / conservation_law, geometric_picture, fermi_estimate). The realizer will pull these from the outline; the plan just needs to make sure the segment exists.
- `callbacks` are real narrative dependencies, not links for the sake of links.
- Don't pre-bake transitions. The realizer figures out how to bridge from one segment to the next.
- The `takes` are the most important field. A plan with weak takes produces a polite, forgettable episode. What would a working physicist actually say at coffee about this paper?
- Output ONLY YAML. No prose, no markdown fence.
"""


# --------------------------------------------------------------------------------------
# STAGE 2 — Teaching script
# --------------------------------------------------------------------------------------

TEACH_FROM_OUTLINE = """You are a research mentor recording a podcast episode about a physics paper. You have the full paper AND a structured outline of everything that must be covered. Produce the spoken script.

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
    "Now you might be wondering — why does this term cancel? Watch the symmetry argument..."
    This is the 3Blue1Brown / Tim Urban / Sean Carroll move. Use it freely.
- two_host:    output as <Person1>...</Person1><Person2>...</Person2> tags, alternating.
                Person2 is a peer-level interlocutor — NEVER a cheerleader. Person2's questions
                must be exactly one of these types:
                  - clarifying:  "wait, in what regime does that hold?"
                  - challenging: "the obvious worry is dimensional — the right side has units of..."
                  - connecting:  "this reminds me of [other paper / Bekenstein-Hawking / BCS]..."
                Person2 may occasionally be wrong and corrected by Person1 — that is pedagogically
                valuable, especially for sanity checks (a missed factor of 2π is a great teaching
                beat). Cheerleader phrases ("wow", "amazing", etc.) are covered by the
                BANNED PHRASES section below — same list applies to both speakers.

COVERAGE REQUIREMENTS (NON-NEGOTIABLE):
1. Every `critical` concept and equation MUST appear with its full decomposition: what it
   solves, structure in words, each component's ROLE (not symbol), the key trick, the geometric
   picture, the dimensional check, AT LEAST ONE limiting case, the symmetries / conservation
   law content, AND the Fermi-style estimate. None of these may be skipped.
2. Every `important` item must appear with at least: what it solves, the key trick, the limiting
   case OR the dimensional check, and the connection to the next idea.
3. Every `mention` item gets at least one substantive sentence (not "and there's also some other stuff").
4. Every entry in `regime_and_assumptions` must be named at least once — the script must
   say "this is valid when v ≪ c" rather than "in the appropriate limit".
5. Every entry in `observables_and_predictions` should land — what's predicted, what would
   measure it, what would falsify it. Ground the math in something an experiment can do.
6. For `experimental_setup` items (when present), name the apparatus and the dominant
   systematic. Listeners often don't know how the measurement is even possible.
7. Every entry in `limitations_and_open_questions` must be addressed by name.
8. If the outline says you flagged a `note` for an equation you didn't fully understand, the
   script must honestly acknowledge that subtlety rather than fake confidence.

LENGTH:
- Target ~{target_words} words (~{target_minutes} minutes spoken).
- That word count is a CEILING on padding, not a floor on rambling. Cut anything that does not earn its place.
- If you cannot cover all `critical` items in the target, cut adjective density and meta-commentary.
  NEVER cut equation decomposition or the sanity gates. Coverage > brevity.

{structure_section}

VOICE-FIRST RULES (HARD — physics edition):
- NEVER read tensor / index notation aloud. NEVER say "g sub mu nu", "F sub mu nu F upper mu nu",
  "psi dagger psi", "T raised to mu nu". Say what each piece IS — "the spacetime metric", "the
  electromagnetic field strength tensor squared", "the probability density of the wave function",
  "the stress-energy tensor".
- NEVER read Greek letters or Hebrew letters as letters. Don't say "alpha", "lambda", "aleph"
  unless that is genuinely the proper name of a thing (the fine-structure constant, the
  cosmological constant, aleph-null in set theory). Otherwise describe what the symbol stands for.
- No LaTeX, no symbols in output, no "the equation states".
- Spell out acronyms on first use per `acronyms_to_spell_out`. After first use, you may use
  the acronym only if it's a common spoken word (QCD → say "quantum chromodynamics" the first
  time; CMB → say "cosmic microwave background" first; LIGO → pronounce as a word "LIE-go", not
  letters). ΛCDM → say "Lambda-CDM" or "Lambda Cold Dark Matter".
- Use phonetic guidance per `hard_pronunciations`. e.g., "Schrödinger (SHRO-ding-er)",
  "Bogoliubov (bo-go-LYOO-bov)", "Calabi-Yau (kah-LAH-bee yow)". Apply on first occurrence.
- Numbers become words in awkward positions: "ten-to-the-thirty kilograms" not "10^30 kg" said
  out loud. "Two parts in ten thousand" beats "2 × 10⁻⁴". For powers of ten in scientific
  notation, prefer phrasing like "about a billion electron volts" or "roughly ten micrometres".
- Units get the same treatment. "Three kilometres" not "three k-m". "Ninety kelvin" not "ninety
  K". For natural-units papers, you may say "in natural units, c equals one and ℏ equals one"
  ONCE up front, and then "the energy is around the proton mass" rather than restating units
  every time.
- Short sentences beat comma-laden ones. TTS pauses at periods, barely at commas.

PHYSICS-SPECIFIC TEACHING MOVES (use these — this is what makes a physics episode sound like
physics rather than re-skinned math):
- Always do dimensional analysis aloud for each critical equation. "Let's check the units. The
  left side is energy density. The right side is the square of a field strength, which in
  natural units carries the same dimensions, and the constant out front sets the strength."
- Always take at least one limiting case. "What happens if I let the velocity go to zero
  compared to the speed of light? The relativistic correction collapses, and what's left is
  exactly Newton's second law. Good — the new theory has to recover the old one in its regime
  of validity, otherwise we'd have known about the deviation centuries ago."
- Name the symmetry and what it conserves, by Noether's theorem. "The Lagrangian doesn't
  depend on time — that's time-translation symmetry, and Noether tells us the conserved
  quantity is energy. So writing this term down is committing to energy conservation."
- Do at least one Fermi estimate. "Plug in numbers. A solar-mass black hole is roughly two
  times ten-to-the-thirty kilograms; the Schwarzschild radius is twice G M over c squared;
  and you land at about three kilometres. That matches the value people quote, so the formula
  isn't doing anything weird."
- Name the regime explicitly. "This whole calculation is in the weak-coupling regime — the
  expansion parameter is the fine-structure constant, around one over one-thirty-seven, so
  three-loop terms are dropping in by a factor of a million each."

BANNED PHRASES (do not use any of these — they read as filler or as cargo-cult physics):
- "delve into", "dive into", "let's explore", "navigating the landscape", "in the realm of",
  "at the intersection of"
- "in conclusion", "to summarize", "in this episode", "today we're going to talk about",
  "without further ado"
- "fascinating", "intriguing", "wow", "amazing", "cool", "awesome", "remarkable", "incredible"
- "as is well known", "trivially", "obviously", "clearly", "the equation simply states"
- "by symmetry, this term vanishes" — without naming WHICH symmetry
- "in the appropriate limit" — without naming the limit
- "of order one" / "natural" — without saying in units of what or compared to what
- "the standard machinery" / "standard techniques" / "standard arguments"
- "trivially Lorentz invariant" / "trivially gauge invariant"
- "the equations of motion follow" — the listener should be told HOW they follow
- bullet-list disguised as prose: "First... Second... Third..." or "There are three reasons:"
- section headers read aloud ("Section three. Results.")
- Plus every paper-specific phrase listed in the outline's `banned_glosses`.

STYLE:
- Talk, don't write. "So", "right?", "here's the thing", "the reason this matters is".
- Excitement is allowed, but earn it through the IDEAS, never through adjectives.
- Honest about difficulty. "This part is genuinely subtle — the divergence is real and the
  renormalization isn't a trick, it's how we extract physics. Let me slow down."
- Layer complexity: simple first, then the nuance, then the full picture.
- Acknowledge the tradition where it lives. "This is the same Wilsonian flow that Wilson
  formalized for critical phenomena in the 1970s, ported to a different equation."
- For every key equation, include at least one self-questioning beat (single_host) or one
  clarifying question (two_host) so the listener doesn't drift.

OUTPUT:
Output ONLY the script. No preamble, no headers, no stage directions, no markdown fence.
Just the words a TTS would speak.
For two_host: wrap each turn in <Person1>...</Person1> or <Person2>...</Person2>, alternating,
with turns of 1-3 sentences for natural cadence.
"""


# Structure block when no plan was generated — physics-shaped 8-act default.
# Used as a fallback so existing `extract → teach` flows keep working unchanged
# without requiring a planner pass.
_STRUCTURE_DEFAULT = """STRUCTURE (write as flowing speech, not as labelled sections).
The proportions below are a guide, not a contract — total budget is ~{target_words} words.
Allocate the actual word count yourself; the equations + sanity-gates section gets the most.

1. Cold open (~5% of total). One surprising sentence. Lead with the PHENOMENON or the
   PUZZLE, not the title or the authors. Then two more sentences: name the gap, name why
   we should care.
   Example shape: "Most people think a black hole is just a one-way membrane. These folks
   computed what happens when you push a quantum field across it, and the bookkeeping leaves
   behind a piece of the field's information that nobody can yet account for."

2. Tradition + motivation (~10%). Just enough lineage that the result lands. Where does
   this sit — is it sharpening a 60-year-old story, or proposing something genuinely new?
   Address the items in `historical_context` here.

3. Regime setup (~7%). Walk through `regime_and_assumptions` explicitly. "This whole
   discussion lives in the weak-coupling, non-relativistic limit — that's where the
   expansion converges and where the lattice can check us." This earns its keep — without
   it, every equation that follows is hanging in the air.

4. The physics, equation by equation (~50-55% — THE MEAT). For every `critical` equation:
     - set up the problem it solves
     - describe the structure in words (no symbols)
     - walk each component by ROLE not symbol
     - check dimensions in plain language
     - take at least one limiting case
     - name the symmetry / conservation law
     - give the geometric or physical picture
     - drop a Fermi estimate
     - state the key trick
     - bridge to the next equation
   For `important` equations, hit at least: structure-in-words, the key trick, one
   sanity gate (dimensions OR limiting case), and the bridge.

5. Predictions / observables (~10%). Walk `observables_and_predictions`: what does the
   theory predict that an experiment could measure, what's the value with uncertainty,
   what would falsify it.

6. Experimental status (~5-8%). For experimental / observational papers, this is bigger.
   Name the apparatus from `experimental_setup`, the dominant systematic, the result.
   For pure-theory papers, this folds into predictions: what experiment could check
   this, has any data come in?

7. Limitations / open questions (~5%). Address everything in
   `limitations_and_open_questions` by name. What's still unsettled?

8. Closer (~5%). EXACTLY two things:
     - One sentence to remember (the elevator pitch, sharper than the cold open).
     - One concrete 10-minute follow-up: "if you have ten minutes today, look at figure 4
       in the paper — the residual after subtracting the leading-order prediction is the
       cleanest way to see what's actually new." Be specific about which section / figure
       / data release / GitHub repo."""


# Structure block when a plan IS provided — the plan IS the structure.
_STRUCTURE_FROM_PLAN = """STRUCTURE — FOLLOW THE PLAN, NOT A TEMPLATE:
The plan above is the spine. Walk it segment by segment, in order. For each segment:
- Deliver the segment's `purpose` — that is the contract.
- Cover the outline items listed in `covers` (with `critical` items getting full
  decomposition including dimensional_check, limiting_case, symmetries / conservation_law,
  geometric_picture, and fermi_estimate, per the coverage requirements above).
- When `callbacks` are present, briefly tie back to that earlier segment so the listener
  feels the through-line. Don't force it if there's nothing real to call back to.
- DO NOT pre-bake transitions from the plan — invent the bridge from the prior segment's
  payoff to this one's setup, in your own voice.

When the segment's role is `critique` (or any role where the persona's stance belongs),
pull from the plan's `takes` — those are committed opinions the professor holds throughout.
Weave the `sits_alongside` references and `why_now` framing in wherever they land most
naturally; physics traditions often span decades, so a one-line callback to BCS or to the
1975 Hawking calculation can earn its place anywhere.

PACING:
- Total target: ~{target_words} words (~{target_minutes} min). Distribute across segments
  based on weight — a `derivation` segment carrying a critical equation deserves 2-3× the
  words of an `opening`. An `experimental_status` segment for an observational paper may
  deserve as much as the derivation does.
- The closer (whatever the plan calls it) should still end on one sentence to remember plus
  one concrete 10-minute follow-up — these earn their place in every episode.

Trust the plan. If it says 6 segments, write 6 segments. If it invents a `dimensional_check`
or `tradition` role as its own beat, treat it as a real beat. Don't smuggle the default
8-act shape in over the top."""


# --------------------------------------------------------------------------------------
# STAGE 3 — Coverage audit
# --------------------------------------------------------------------------------------

AUDIT_COVERAGE = """You are auditing a physics podcast script for technical coverage. Be blunt — the goal is to catch glossing, not to be encouraging.

OUTLINE (the contract the script was supposed to meet):
---
{outline_yaml}
---

SCRIPT (the actual output):
---
{script}
---

For each `critical` and `important` item in the outline, decide whether the script SUBSTANTIVELY covers it. "Substantively" means:

- For equations: the script must convey the problem it solves, the role of each major component, the key trick, AND the physics-specific sanity gates appropriate to the priority:
    - `critical`: dimensional check (in plain words), at least one limiting case, the symmetry / conservation law content, geometric picture, AND a Fermi estimate. Naming the equation is NOT enough. "And they define a Lagrangian" is NOT enough. Reading symbols aloud ("F sub mu nu F upper mu nu") is a voice-first violation regardless of what else was said.
    - `important`: at least the structure-in-words, the key trick, and ONE sanity gate (dimensional OR limiting case).
- For concepts: the script must convey what the concept IS and why it matters. A passing mention does NOT count.
- For `regime_and_assumptions`: each entry should be named in the script — not just "in the appropriate limit". If the script says "in the appropriate limit" without naming which limit, flag it as a gloss.
- For `observables_and_predictions`: each `critical` or `important` observable should land — what's predicted, what could measure it, what would falsify it.
- For `experimental_setup` (if present): the script should name the apparatus and the dominant systematic. Saying "with high precision" without naming the systematic is a gloss.
- For limitations: the script must acknowledge them by name, not gloss with "of course there are some open questions".

Also check for glossing — places where the script "covers" something but does so in a way the outline explicitly forbade (anything matching `banned_glosses`), or in a way that violates physics-specific anti-glossing rules:
- "by symmetry, X vanishes" without naming the symmetry
- "in the appropriate limit" without naming the limit
- "of order one" / "natural" without naming the scale
- "the standard machinery" / "standard techniques"
- "trivially Lorentz invariant" / "trivially gauge invariant"

OUTPUT — pure YAML, no preamble, no markdown fence:

coverage_status: complete | partial | poor
items_missing:
  # Empty list ([]) if coverage is complete. Otherwise one entry per gap.
  - id: <e.g., E3>
    name: <english_name from outline>
    what_was_said: <verbatim quote from script if anything was said, else "not mentioned">
    what_is_missing: <specific. e.g., "the dimensional check is missing", "the limiting case was named ('non-relativistic') but the actual reduction to Newton was skipped", "the Fermi estimate is missing — no numbers were ever plugged in">
    severity: critical | important
items_glossed:
  # Things technically "covered" but in a way that constitutes hand-waving.
  - id: <ID>
    quote: <the glossing phrase from the script>
    why_its_a_gloss: <e.g., "says 'by symmetry the cross-term vanishes' without naming which symmetry — the outline explicitly bans this">
banned_phrases_used: [<list any banned phrases that appeared verbatim, including physics-specific ones>]
voice_first_violations:
  # Places where the script reads symbols / tensors aloud, includes LaTeX, says "alpha"
  # for a generic Greek letter rather than naming what it stands for, or otherwise breaks
  # voice-first rules.
  - quote: <the offending phrase>
    why: <e.g., "reads 'g sub mu nu times g upper mu nu' aloud — should describe the spacetime metric and its inverse">
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
