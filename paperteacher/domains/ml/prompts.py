"""Three prompts that drive the pipeline: extract outline, teach from outline, audit coverage.

The pipeline is decompose-then-execute. Stage 1 forces the model to enumerate
every equation and concept BEFORE narrative pressure makes it want to skip the
hard math. Stage 2 writes the script with explicit coverage requirements drawn
from stage 1. Stage 3 verifies and triggers regeneration if anything was glossed.
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

key_concepts:
  - id: C1
    name: <short descriptive name>
    plain_english: <one sentence the listener could repeat at dinner>
    why_it_matters: <one sentence connecting to the listener's profile or to the field>
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

limitations_and_open_questions:
  - <subtle issues the authors acknowledge — quote or paraphrase>
  - <natural extensions worth thinking about>

banned_glosses:
  # specific phrases that, if used in the script for THIS paper, would constitute hand-waving
  - <e.g., "the loss is just MSE">
  - <e.g., "and then we apply standard techniques">

acronyms_to_spell_out:
  # acronyms in the paper that should be expanded on first reading aloud
  - <ABBR>: <full form>

hard_pronunciations:
  # author names, foreign terms, or jargon that TTS will mangle
  - <term>: <PHO-ne-tic>

RULES:
- Be exhaustive on equations. Do not skip the appendix if it has the cleanest proof.
- For `critical` equations every field above is mandatory.
- For each component, describe ROLE not SYMBOL. "the model's predicted gradient field at each input point" — NOT "nabla theta of f sub theta".
- If you don't have a clean intuition for an equation, mark it `mention` and add `note: "I don't have a clean intuition — the script should acknowledge that"`. Flag, don't fake.
- Output ONLY YAML. No prose, no markdown fence, no "Here is the outline:".
"""


# --------------------------------------------------------------------------------------
# STAGE 2 — Teaching script
# --------------------------------------------------------------------------------------

TEACH_FROM_OUTLINE = """You are a research mentor recording a podcast episode. You have the full paper AND a structured outline of everything that must be covered. Produce the spoken script.

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

OUTLINE (MANDATORY COVERAGE — your script will be audited against this):
---
{outline_yaml}
---

DELIVERY MODE: {mode}
- single_host: one narrator. Use self-questioning to create internal dialogue:
    "Now you might be wondering — why does this term cancel? Here's the trick..."
    This is the 3Blue1Brown / Tim Urban move. Use it freely.
- two_host:    output as <Person1>...</Person1><Person2>...</Person2> tags, alternating.
                Person2 is a peer-level interlocutor — NEVER a cheerleader. Person2's questions
                must be exactly one of these types:
                  - clarifying:  "wait, doesn't that mean..."
                  - challenging: "the obvious worry is..."
                  - connecting:  "this reminds me of [other paper / other field]..."
                Person2 may occasionally be wrong and corrected by Person1 — that is pedagogically
                valuable. Banned in two_host: "wow", "fascinating", "amazing", "cool", "awesome",
                "that's so interesting".

COVERAGE REQUIREMENTS (NON-NEGOTIABLE):
1. Every `critical` concept and equation MUST appear with its full decomposition: what it solves,
   structure in words, each component's ROLE (not symbol), the key trick, the geometric picture,
   AND the numerical walkthrough. None of these may be skipped.
2. Every `important` item must appear with at least: what it solves, the key trick, and the
   connection to the next idea.
3. Every `mention` item gets at least one substantive sentence (not "and there's also some other stuff").
4. Every entry in `limitations_and_open_questions` must be addressed by name.
5. If the outline says you flagged a `note` for an equation you didn't fully understand, the
   script must honestly acknowledge that subtlety rather than fake confidence.

LENGTH:
- Target ~2500 words (~15 minutes spoken).
- 2500 is a CEILING on padding, not a floor on rambling. Cut anything that does not earn its place.
- If you cannot cover all `critical` items in 2500 words, cut adjective density and meta-commentary.
  NEVER cut equation decomposition. Coverage > brevity.

STRUCTURE (write as flowing speech, not as labelled sections):
1. Cold open (~30 sec / 75 words). One surprising sentence. Lead with the IDEA. Not the title,
   not the authors. Then two more sentences: name the gap, name why we should care.
   Example shape: "Most people think attention scales like n-squared. These folks made it linear
   without losing anything — and the reason it works isn't engineering, it's a geometric
   observation about what attention actually computes."
2. Context (~90 sec / 225 words). Just enough background that the key idea lands.
3. Key idea (~2 min / 300 words). Plain language first. Could-retell-at-dinner test.
4. The math, equation by equation (~8-10 min / 1500 words — THE MEAT). Walk through every
   `critical` equation using its outline decomposition:
     - set up the problem it solves
     - describe the structure in words
     - walk each component by ROLE not symbol
     - give the geometric picture
     - give the numerical walkthrough
     - state the key trick
     - bridge to the next equation
5. Results (~2 min / 300 words). Pick 2-3 from `results_to_highlight`. Explain what each
   demonstrates about the theory, not just what numbers were achieved.
6. Bigger picture (~1 min / 150 words). Where this sits, what it enables, what's still open.
   Address the items in `limitations_and_open_questions` here.
7. Closer (~30 sec / 75 words). EXACTLY two things:
     - One sentence to remember (the elevator pitch, sharper than the cold open).
     - One concrete 10-minute follow-up: "if you have ten minutes today, skim section 3.2 —
       the proof of the cancellation lemma is the cleanest thing in the paper." Be specific
       about which section / figure / GitHub repo.

VOICE-FIRST RULES (HARD):
- NEVER read equations symbolically. NEVER say "sigma sub i", "nabla theta", "log p of x",
  "L equals expected value of...". Say what each piece IS.
- No LaTeX, no symbols in output, no "the equation states".
- Spell out acronyms on first use per `acronyms_to_spell_out`. After first use, you may use
  the acronym only if it's a common spoken word (RNN → say "recurrent net" the first time,
  then either is fine; ELBO → say "evidence lower bound" first, then say "ELBO" — but pronounce
  it as a word "EL-bow", not letters).
- Use phonetic guidance per `hard_pronunciations`. e.g., "Schrödinger" → write "Schrödinger
  (SHRO-ding-er)" the first time it appears.
- Short sentences beat comma-laden ones. TTS pauses at periods, barely at commas.
- Numbers become words in awkward positions: "two and a half billion parameters" not
  "2.5B parameters". "Eighty-two million" not "82M".

BANNED PHRASES (do not use any of these — they read as filler):
- "delve into", "dive into", "delve", "let's explore", "navigating the landscape",
  "in the realm of", "at the intersection of"
- "in conclusion", "to summarize", "in this episode", "today we're going to talk about",
  "without further ado"
- "fascinating", "intriguing", "wow", "amazing", "cool", "awesome", "remarkable", "incredible"
- "as is well known", "the equation simply states", "trivially", "obviously", "clearly"
- bullet-list disguised as prose: "First... Second... Third..." or "There are three reasons:
  one, ...; two, ...; three, ..."
- section headers read aloud ("Section three. Results.")
- Plus every paper-specific phrase listed in the outline's `banned_glosses`.

STYLE:
- Talk, don't write. "So", "right?", "here's the thing", "the reason this matters is".
- Excitement is allowed, but earn it through the IDEAS, never through adjectives.
- Honest about difficulty. "This part is genuinely subtle, so let me slow down" beats
  pretending it's obvious.
- Layer complexity: simple first, then the nuance, then the full picture.
- For every key equation, include at least one self-questioning beat (single_host) or one
  clarifying question (two_host) so the listener doesn't drift.

OUTPUT:
Output ONLY the script. No preamble, no headers, no stage directions, no markdown fence.
Just the words a TTS would speak.
For two_host: wrap each turn in <Person1>...</Person1> or <Person2>...</Person2>, alternating,
with turns of 1-3 sentences for natural cadence.
"""


# --------------------------------------------------------------------------------------
# STAGE 3 — Coverage audit
# --------------------------------------------------------------------------------------

AUDIT_COVERAGE = """You are auditing a podcast script for technical coverage. Be blunt — the goal is to catch glossing, not to be encouraging.

OUTLINE (the contract the script was supposed to meet):
---
{outline_yaml}
---

SCRIPT (the actual output):
---
{script}
---

For each `critical` and `important` item in the outline, decide whether the script SUBSTANTIVELY covers it. "Substantively" means:
- For equations: the script must convey the problem it solves, the role of each major component, the key trick, AND a concrete picture (geometric or numerical). Just naming the equation does NOT count. Saying "and they define a loss function" does NOT count.
- For concepts: the script must convey what the concept IS and why it matters. A passing mention does NOT count.
- For limitations: the script must acknowledge them by name, not gloss with "of course there are some open questions".

Also check for glossing — places where the script "covers" something but does so in a way the outline explicitly forbade (anything matching `banned_glosses`).

OUTPUT — pure YAML, no preamble, no markdown fence:

coverage_status: complete | partial | poor
items_missing:
  # Empty list ([]) if coverage is complete. Otherwise one entry per gap.
  - id: <e.g., E3>
    name: <english_name from outline>
    what_was_said: <verbatim quote from script if anything was said, else "not mentioned">
    what_is_missing: <specific. e.g., "the geometric picture is missing" or "the numerical walkthrough was skipped" or "the key trick was named but not explained">
    severity: critical | important
items_glossed:
  # Things technically "covered" but in a way that constitutes hand-waving.
  - id: <ID>
    quote: <the glossing phrase from the script>
    why_its_a_gloss: <e.g., "calls it 'just MSE' which the outline explicitly bans">
banned_phrases_used: [<list any banned phrases that appeared verbatim>]
voice_first_violations:
  # Places where the script reads symbols aloud, includes LaTeX, or otherwise breaks voice-first rules.
  - quote: <the offending phrase>
    why: <e.g., "reads 'sigma sub i' aloud — should describe role">
overall_assessment: <2 sentences. Be blunt. End with one sentence on whether to regenerate or ship.>
recommendation: ship | regenerate_with_gaps | regenerate_from_scratch
"""


# --------------------------------------------------------------------------------------
# Render helpers (used by server.py to fill in placeholders)
# --------------------------------------------------------------------------------------


def render_extract(*, arxiv_id: str, title: str, taste_profile: str, paper_text: str) -> str:
    return EXTRACT_OUTLINE.format(
        arxiv_id=arxiv_id,
        title=title,
        taste_profile=taste_profile,
        paper_text=paper_text,
    )


def render_teach(
    *,
    arxiv_id: str,
    title: str,
    taste_profile: str,
    paper_text: str,
    outline_yaml: str,
    mode: str = "single_host",
) -> str:
    return TEACH_FROM_OUTLINE.format(
        arxiv_id=arxiv_id,
        title=title,
        taste_profile=taste_profile,
        paper_text=paper_text,
        outline_yaml=outline_yaml,
        mode=mode,
    )


def render_audit(*, outline_yaml: str, script: str) -> str:
    return AUDIT_COVERAGE.format(outline_yaml=outline_yaml, script=script)
