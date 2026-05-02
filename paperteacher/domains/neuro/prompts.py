"""Three-stage prompts for the neuroscience pack.

Same decompose-then-execute architecture as the ML pack, swapping the central
unit. Where ML asks the extractor to enumerate every equation by ROLE not
SYMBOL, neuro asks it to enumerate every key finding alongside the **method**
that produced it and the **control** that rules out the alternative — because
the gap between method and claim is where neuroscience papers actually live.

Voice-first rules are different too: TTS mangles brain-region acronyms and
biological terms much worse than it mangles math notation, and reading
p-values aloud is the equivalent of reading "sigma sub i" — describe the
effect's shape, not its statistical packaging.
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------
# STAGE 1 — Extraction
# --------------------------------------------------------------------------------------

EXTRACT_OUTLINE = """You are extracting the structured outline of a neuroscience paper BEFORE it gets taught as a podcast. Your job is to be exhaustive about what needs to be covered. Better to over-extract than to miss.

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

For every significant finding you MUST produce a structured decomposition: not just what was found, but the METHOD that produced it (different techniques license different conclusions) and the CONTROL that rules out the obvious alternative explanation. The gap between data and claim is where neuroscience papers live; the audit at stage 3 will fail you if the script glosses methods or skips controls.

OUTPUT — pure YAML, no preamble, no markdown fence, no commentary:

paper_id: {arxiv_id}
type: experimental | modeling | clinical | review | position
core_thesis: <exactly 2 sentences naming the single most important claim>
gap_filled: <1 sentence: what was wrong, missing, or unjustified before this paper>

subjects:
  organism: <e.g., "C57BL/6J mice", "two adult rhesus macaques (M. mulatta)", "n=24 humans, age 18-35", or "" if N/A>
  sample_size: <e.g., "n=8 mice, 412 cells", "n=24 humans">
  brain_regions: [<spelled out: "primary visual cortex (V1)", "dorsolateral prefrontal cortex (dlPFC)">]

key_concepts:
  - id: C1
    name: <short descriptive name>
    plain_english: <one sentence the listener could repeat at dinner>
    why_it_matters: <one sentence connecting to the listener's profile or to the field>
    teaching_priority: critical | important | mention

key_methods:
  # Every distinct technique that load-bears on a critical finding. If the paper does
  # in-vivo calcium imaging AND optogenetic perturbation AND a behavioral task, that's
  # 2 methods (the task lives below) — both must appear.
  - id: M1
    name: <plain-English name, NOT just an acronym. "two-photon calcium imaging", not "2P">
    what_it_measures: <the actual physical signal — fluorescence as a proxy for spiking averaged over hundreds of milliseconds, BOLD as a proxy for neurovascular coupling, etc.>
    spatial_temporal_resolution: <e.g., "single cells, ~100ms" / "voxels ~2mm, ~1s" / "single units, ~1ms" / "millions of cells averaged, hours">
    typical_confounds: [<the well-known limits — "calcium signals lag spiking", "GCaMP saturates at high firing rates", "BOLD reflects vascular not neural per se">]
    teaching_priority: critical | important | mention

behavioral_tasks:
  # Empty list ([]) if the paper has no behavioral component (cell biology, anatomy,
  # connectomics, in-vitro slice, modeling). Otherwise one entry per distinct task.
  - id: T1
    name: <e.g., "delayed match-to-sample", "evidence-accumulation 2AFC">
    what_subjects_did: <plain English, one trial described concretely>
    what_is_varied: <the parametric or categorical variable the analysis hangs on — coherence, delay length, reward size>
    why_this_design: <what the design rules in or out — "isolating short-term memory from sensory processing because the cue is gone before the response">
    teaching_priority: critical | important | mention

key_findings:
  # INCLUDE EVERY SIGNIFICANT FINDING. The audit will check coverage against this list.
  # Mark trivial ones `mention`, but they must appear.
  - id: F1
    name: <plain-English name. "Place cells in CA1 remap when the reward location moves", NOT "Figure 3">
    what_it_shows: <one sentence — the effect>
    effect_in_words: <the magnitude/direction described as a picture, not numbers. "Roughly half the cells changed which location they fired at; the rest kept their old preferred location even when it stopped being rewarded.">
    concrete_picture: <a literal mental image — a tuning curve flattening, a population vector rotating, a manifold compressing along one axis. Something the listener can SEE.>
    key_control: <the control or alternative explanation ruled out, AND HOW. "They could have argued the cells just track reward, but the same cells maintained their old fields on probe trials with no reward — see Fig 4B.">
    numerical_anchor: <ONE concrete number: "n=8 mice, 412 cells", "the response was three times larger", "in 70% of sessions". Not p-values.>
    bridge_to_next: <how this finding connects to what comes next>
    teaching_priority: critical | important | mention

results_to_highlight:
  - id: R1
    claim: <statistical or replication-level finding — effect size, n, replicated in N labs>
    what_it_demonstrates: <what this proves about the biology or method>
    why_surprising: <if applicable; otherwise omit>

limitations_and_open_questions:
  - <subtle issues the authors acknowledge — quote or paraphrase>
  - <natural extensions worth thinking about>

banned_glosses:
  # specific phrases that, if used in the script for THIS paper, would constitute hand-waving
  - <e.g., "they recorded from neurons">
  - <e.g., "the cells were active during the task">
  - <e.g., "they showed a significant effect">

acronyms_to_spell_out:
  # spell out brain-region and technique acronyms on first reading
  - <ABBR>: <full form — e.g., dlPFC: dorsolateral prefrontal cortex>

hard_pronunciations:
  # author names, brain regions, latin terms TTS will mangle
  - <term>: <PHO-ne-tic>

RULES:
- Be exhaustive on findings. Do not skip the supplementary if it has the cleanest control.
- **MINIMUM 3 ITEMS MARKED `critical`** across `key_findings` + `key_methods` +
  `key_concepts`. The `critical` tier forces the teach stage to do full decomposition
  including the named control (alternative explanation + how it was ruled out). If
  you mark everything `mention`, you produce a hollow outline and the script
  collapses into an abstract paraphrase. Pick the 3-5 things the paper genuinely
  lives or dies by, and mark them `critical`. If you honestly cannot find 3 — the
  paper is too thin; surface that in `core_thesis` rather than faking compliance.
- For `critical` findings every field above is mandatory. The `key_control` must NAME the alternative explanation and the data that ruled it out.
- For each method, describe what it MEASURES, not the brand. "Fluorescence as a proxy for spiking averaged over hundreds of milliseconds" — NOT "GCaMP6f imaging".
- NEVER write p-values, F-statistics, or "significant at p<0.05" into the outline. Use effect direction and magnitude instead. The listener will tune out at "p equals zero point zero zero one".
- If you don't have a clean intuition for a finding, mark it `mention` and add `note: "I don't have a clean intuition — the script should acknowledge that"`. Flag, don't fake.
- Output ONLY YAML. No prose, no markdown fence, no "Here is the outline:".
"""


# --------------------------------------------------------------------------------------
# STAGE 1.5 — Episode plan (macro structure)
# --------------------------------------------------------------------------------------

PLAN_EPISODE = """You are designing the macro structure of a podcast episode about a neuroscience paper. The structure MUST BE SHAPED BY THE PAPER — a connectomics paper does not get the same arc as a behavior paper, and a clinical translational paper definitely doesn't get the same arc as an in-vitro slice paper. Your job is to think about what THIS specific paper deserves, then commit the persona's stance about it.

LISTENER PROFILE (anchors voice, depth level, and which neuroscience traditions to lean on):
---
{taste_profile}
---

The paper's full text is intentionally NOT inlined — the outline below carries every method, finding, control, and limitation the planner needs. Plan the arc from the outline + the listener profile.

PAPER:
arxiv_id: {arxiv_id}
title: {title}

OUTLINE (already extracted — segments will reference these by id):
---
{outline_yaml}
---

You are NOT writing the script. You are deciding:
1. The arc of segments — typically 5-9, but use your judgment. A subtle controls-heavy paper might want 10; a single-finding clinical paper might want 4.
2. Which outline items each segment covers (by id).
3. The persona's COMMITTED OPINIONS about this paper — the takes that survive across segments and make the professor sound like a real expert with a stance, not a polite summarizer.
4. Where this paper sits in the field — adjacent works, why this lands now.

ARC SHAPING — DON'T TEMPLATE:
Different neuroscience papers want different arcs. A few examples (don't follow these literally — let the paper lead):
- A circuit-mapping paper might open with what the field thought the wiring was, walk the new connectome, then dwell on what the data does and does NOT license about function.
- A behavioral paper might spend two segments on what was wrong with prior tasks before introducing the design, then dwell on the surprising effect with skeptical eyes on the controls.
- A modeling paper has no recordings to walk through — the arc IS the model, and the "what would falsify this" segment is bigger than usual.
- A clinical translational paper should pace the gap between the animal mechanism and the human population, not paper over it.
- A connectomics or methods paper with one new technique should pace toward the technique, then reflect on what it actually buys you.

Suggested role names (use these or invent your own — `controls_walk`, `historical_aside`, `replication_check`, `vibes_check`, `live_dissection`, whatever fits):
opening | motivation | prereq | setup | method | task | finding | control | critique | closing

OUTPUT — pure YAML, no preamble, no markdown fence:

paper_id: {arxiv_id}
arc:
  - id: seg_01
    role: <free-form. opening / motivation / prereq / setup / method / task / finding / control / critique / closing — or invent>
    covers: [<outline ids — M*, T*, F*, C* — that this segment delivers>]
    callbacks: [<prior seg_ ids this segment leans on, if any>]
    purpose: <one line. what this segment must accomplish for the listener>
  # ... continue with seg_02, seg_03, ...

takes:
  # 2-5 committed opinions. The realizer will pull from these in critique segments
  # and asides — they are the persona's STANCE, not generic praise.
  # Avoid: "this is an important contribution to the field".
  # Prefer:  "the imaging is beautiful, but they're underselling that the effect only
  #           shows up in trained animals — the naive cohort in supplementary Fig 7
  #           barely shifts. The story is about training, not innate coding."
  - claim: <the opinion itself, in the professor's voice>
    evidence: <what in the paper or field supports it (one line)>

sits_alongside:
  # 1-3 adjacent works or lines of work that situate this paper.
  - <e.g., "Hardcastle et al. 2017 — same remapping framing in MEC, different task structure">

why_now: <one sentence. what changed in tooling / data / theory that made this paper possible or urgent right now>

RULES:
- Every `critical` outline item should be covered by some segment. `important` items mostly should. `mention` items can be folded in or skipped.
- A segment's `covers` should be tight — usually 1-3 outline items. A segment that covers everything covers nothing.
- For each `critical` finding, there should be a segment (or sub-beat) that delivers its `key_control`. Skipping the control is the cardinal sin of teaching neuroscience.
- `callbacks` are real narrative dependencies, not links for the sake of links. If seg_05 doesn't actually need to reference seg_02, leave it empty.
- Don't pre-bake transitions. The realizer figures out how to bridge from one segment to the next based on neighbouring `purpose` lines.
- The `takes` are the most important field. A plan with weak takes produces a polite, forgettable episode. Spend real thought on them — what would a domain expert *actually* say at journal club about this paper?
- Output ONLY YAML. No prose, no markdown fence.
"""


# --------------------------------------------------------------------------------------
# STAGE 2 — Teaching script
# --------------------------------------------------------------------------------------

TEACH_FROM_OUTLINE = """You are a research mentor producing the spoken text of a deep-dive on a neuroscience paper. Output goes straight into a TTS engine — one (or two) unnamed voices, no studio framing. You have the full paper AND a structured outline you must cover.

LISTENER PROFILE (drives voice, depth, framings the listener already has):
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
- single_host: one narrator using self-questioning ("you might wonder if the cells just
  track reward — here's how they ruled that out...") for internal dialogue.
- two_host: <Person1>...</Person1><Person2>...</Person2> alternating. Person2 is a peer
  interlocutor; every Person2 turn is exactly one of: clarifying ("wait, when you say
  calcium imaging, you mean..."), challenging ("the worry is the imaging signal lags
  by ~100ms..."), or connecting ("this reminds me of the Hardcastle remapping paper...").
  Person2 may be wrong and get corrected. NO cheerleader phrases; same banned list as
  below applies to both.

THE CONTRACT — your script will be audited against this. Failure modes named here are
audit failures, not stylistic preferences:

For every `critical` finding, the full chain in spoken form:
  what the method actually MEASURES (the physical signal — "fluorescence as a proxy for
  spiking", "BOLD as a proxy for neurovascular coupling" — NEVER "they recorded neurons")
  → effect in plain language → concrete mental picture (tuning curve flattening,
  population vector rotating, manifold compressing) → the KEY CONTROL by name: state
  the alternative explanation FIRST, then the experimental logic that ruled it out →
  one numerical anchor that is NOT a p-value. Naming the finding is failure. "They showed
  a significant effect" is failure. Skipping the control is the cardinal sin.

For every `critical` method: what it physically measures, resolution, typical confounds.
For every `critical` behavioral task: what subjects did (one trial concretely described)
  + what's varied + what the design rules in or out.

For every `important` item: at minimum what-it-is, why-it-matters, the connection to next.
For every `mention` item: at least one substantive sentence.

The `subjects` line is stated explicitly — "this is two macaques, not a human study"
  beats "subjects were used". Every entry in `limitations_and_open_questions` is
  addressed by name. If the outline carries a `note` for something you didn't fully
  understand, acknowledge that subtlety honestly.

PERSONA — what makes this not NotebookLM:
The voice is a working neuroscientist with a STANCE. Bring lineage (place cells,
remapping, predictive coding, ALM, drift...), connections to adjacent labs, opinions
on whether the controls actually nail down what the headline claims. Generic praise is
failure.

Coverage > brevity. ALWAYS.

LENGTH:
- Target ~{target_words} words (~{target_minutes} minutes spoken). This is a TARGET.
- Under ~80% of target = under-covered. Expand a `critical` finding with more on the
  method, the effect-as-picture, the concrete mental image, or the key control by name
  (alternative explanation + how it was ruled out), until you hit target.
- 10–15% over is fine when the controls earn it. Cut filler before coverage; never cut
  controls.

VARIETY — this is a script for THIS paper, not a template:
- Two of your episodes about different neuro papers must NOT have the same shape. If
  yours could be transposed onto a different paper unchanged, you're templating, not
  teaching. Let the paper's argument decide the arc.
- Pacing is uneven on purpose. Spend the most words on whatever this paper deserves —
  sometimes the controls, sometimes the method's limits, sometimes one population's
  pre-trial response that flips the interpretation. There are no fixed proportions.
- Asides and callbacks earn their place. "Wait, this is the same population we said
  was non-task-modulated five minutes ago...", "side note: this is reminiscent of
  Hardcastle 2017", "let me re-state the alternative explanation, it matters".
- The ending lands for THIS paper. Default: one sentence to remember + one concrete
  10-minute follow-up (specific figure, supplementary panel, dataset). Vary if the
  paper calls for it — a question that the next experiment would settle, a stance
  on whether the species/population gap actually carries.

{structure_section}

VOICE-FIRST RULES (HARD):
- NEVER read brain-region or technique acronyms cold. Spell out per `acronyms_to_spell_out`
  on first use. After first use you may use the acronym only if it's pronounceable as a word
  (fMRI → say "functional MRI" first, then either is fine; dlPFC → say "dorsolateral
  prefrontal cortex" first, then "the prefrontal cortex" or "dlPFC" only if pronounced
  "dee-ell-PFC" smoothly).
- NEVER read p-values, F-statistics, or test names aloud. "Significant at p less than zero
  point zero zero one" is the neuroscience equivalent of "sigma sub i" — listeners tune out.
  Translate: "the effect held up across animals" / "in seven of eight mice" / "roughly three
  times larger than baseline".
- NEVER describe a method as "we recorded neurons" or "we imaged the brain". Say what it
  actually measures: "We watched fluorescence — basically a chemical proxy for how often
  cells were spiking, averaged over a few hundred milliseconds — in roughly four hundred
  cells at once."
- NEVER skip the control. The pattern is: state the alternative explanation, then the
  experimental logic that ruled it out. Always in that order.
- Use phonetic guidance per `hard_pronunciations`. e.g., "Cajal (kah-HAHL)", "locus
  coeruleus (low-kus seh-ROO-lee-us)", "entorhinal (en-tor-EYE-nuhl)".
- Numbers as words in awkward positions: "eight mice and four hundred cells" not "n=8, 400
  cells". "Two and a half millivolts" not "2.5 mV".
- Short sentences beat comma-laden ones. TTS pauses at periods, barely at commas.

BANNED PHRASES (do not use any of these — they read as filler):
- "delve into", "dive into", "delve", "let's explore", "navigating the landscape",
  "in the realm of", "at the intersection of"
- "in conclusion", "to summarize", "in this episode", "today we're going to talk about",
  "without further ado"
- "fascinating", "intriguing", "wow", "amazing", "cool", "awesome", "remarkable", "incredible"
- "as is well known", "trivially", "obviously", "clearly", "they showed a significant effect",
  "the cells were active during the task", "they recorded from neurons"
- bullet-list disguised as prose: "First... Second... Third..." or "There are three reasons:
  one, ...; two, ...; three, ..."
- section headers read aloud ("Section three. Results.")
- show/podcast framing: "Welcome to", "Welcome back to", "Today we're diving into",
  "Today we're talking about", "On today's episode", "That's all the time we have",
  "Join us next time", "see you next time", "thanks for listening"
- self-introductions: "I'm Alex", "I'm Ben", "with me is", "joined by", any invented
  host name. Person1 and Person2 are TTS routing tags, NOT named characters — they
  do not introduce themselves and they have no proper names.
- invented show / podcast / column name: never name the production. There is no show.
- Plus every paper-specific phrase listed in the outline's `banned_glosses`.

STYLE:
- Talk, don't write. "So", "right?", "here's the thing", "the reason this matters is".
- Excitement comes from the IDEAS and the controls, never adjectives.
- Honest about difficulty: "this control is genuinely subtle, let me slow down".
- Layer complexity: organism + region → task → method → finding → control →
  the alternative it shut down.
- Every key finding gets at least one self-questioning beat (single_host) or
  challenging question about the controls (two_host).

OUTPUT:
Output ONLY the words that will be spoken aloud — plain prose. No preamble, no markdown
fence, no stage directions or scene markers (no `[SCENE START]`, `**INT. ...**`, music
cues), no markdown formatting (no `**bold**`, `*italic*`, headers, bullets, code ticks),
no podcast framing (no "Welcome to ...", "Today we're talking about ...", "That's all
the time we have", "Join us next time", "thanks for listening"), no invented show name,
no self-introductions. Person1 and Person2 are TTS routing tags, NOT named characters —
neither speaker says "I'm [name]" or "with me is".

For two_host: wrap each turn in <Person1>...</Person1> or <Person2>...</Person2>,
alternating, with turns of 1-3 sentences for natural cadence. Get straight into the science.
"""


# Structure block when no plan was generated — the prescriptive arc.
# Used as a fallback so existing `extract → teach` flows keep working unchanged.
_STRUCTURE_DEFAULT = """SHAPE — pick the arc that serves THIS paper, or invent your own:

▸ FINDING ARC (single-finding papers) — open with the effect, walk the method, land
  the control, reflect on what the data does and does NOT show.
▸ CONTROL ARC (papers whose contribution is a better control) — open with what people
  had been claiming, walk why the previous control wasn't enough, walk the new
  control, land the corrected picture.
▸ METHOD ARC (new technique papers) — open with what was unmeasurable before, walk
  the technique, walk what it reveals, caveat the failure modes.
▸ TRANSLATIONAL ARC (clinical / translational) — open with the human population,
  walk the animal mechanism, walk the species/population gap honestly, land what
  would make this clinically real.

Open with a concrete observation — an effect, a tuning curve shift, a behavior. Not
the title, not the authors. State the `subjects` early — "this is two macaques, not
a human study". THE CONTRACT above tells you what every arc must deliver; the arc
tells you the order for THIS paper."""


# Structure block when a plan IS provided — the plan IS the structure.
_STRUCTURE_FROM_PLAN = """STRUCTURE — FOLLOW THE PLAN, NOT A TEMPLATE:
The plan above is the spine. Walk it segment by segment, in order. For each segment:
- Deliver the segment's `purpose` — that is the contract.
- Cover the outline items listed in `covers` (with `critical` items getting full decomposition,
  per the coverage requirements above — controls included).
- When `callbacks` are present, briefly tie back to that earlier segment so the listener feels
  the through-line. Don't force it if there's nothing real to call back to.
- DO NOT pre-bake transitions from the plan — invent the bridge from the prior segment's
  payoff to this one's setup, in your own voice. This is what makes each episode sound different
  rather than templated.

When the segment's role is `critique` or `controls_walk` (or any role where the persona's
stance belongs), pull from the plan's `takes` — those are committed opinions the professor
holds throughout, not improvisations. Weave the `sits_alongside` references and `why_now`
framing in wherever they land most naturally.

PACING:
- Total target: ~{target_words} words (~{target_minutes} min). Distribute across segments
  based on weight, not evenly — a `finding` segment carrying a critical effect with a subtle
  control deserves 2-3× the words of an `opening` segment.
- The `closer` segment (or whatever the plan calls the final segment) should still end on
  one sentence to remember plus one concrete 10-minute follow-up — these earn their place
  in every episode regardless of arc.

Trust the plan. If it says 6 segments, write 6 segments. If it invents a `replication_check`
role, treat that as a real beat. Don't smuggle the default 7-act shape in over the top."""


# --------------------------------------------------------------------------------------
# STAGE 3 — Coverage audit
# --------------------------------------------------------------------------------------

AUDIT_COVERAGE = """You are auditing a podcast script for technical coverage of a neuroscience paper. Be blunt — the goal is to catch glossing, especially of methods and controls, not to be encouraging.

OUTLINE (the contract the script was supposed to meet):
---
{outline_yaml}
---

SCRIPT (the actual output):
---
{script}
---

For each `critical` and `important` item in the outline, decide whether the script SUBSTANTIVELY covers it. "Substantively" means:
- For findings: the script must convey what was measured, the method that produced it, the effect in plain language, a concrete picture, the KEY CONTROL by name (alternative explanation + how it was ruled out), and at least one numerical anchor. Naming the finding does NOT count. "They showed a significant effect" does NOT count.
- For methods: the script must convey what the method actually MEASURES (the physical signal), at what resolution, and its main limitations. "They imaged the brain" does NOT count.
- For tasks: the script must convey what subjects did, the variable that was varied, and what the design rules in or out.
- For concepts: the script must convey what the concept IS and why it matters.
- For limitations: the script must acknowledge them by name, not gloss with "of course there are some open questions".

Also check for glossing — places where the script "covers" something but does so in a way the outline explicitly forbade (anything matching `banned_glosses`).

OUTPUT — pure YAML, no preamble, no markdown fence:

coverage_status: complete | partial | poor
items_missing:
  # Empty list ([]) if coverage is complete. Otherwise one entry per gap.
  - id: <e.g., F2>
    name: <name from outline>
    what_was_said: <verbatim quote from script if anything was said, else "not mentioned">
    what_is_missing: <specific. e.g., "the key control was named but the experimental logic that ruled it out was skipped" or "method described as 'they imaged the brain' — what it actually measures was not stated">
    severity: critical | important
items_glossed:
  # Things technically "covered" but in a way that constitutes hand-waving.
  - id: <ID>
    quote: <the glossing phrase from the script>
    why_its_a_gloss: <e.g., "calls calcium imaging 'just neural recording' which the outline explicitly bans">
banned_phrases_used: [<list any banned phrases that appeared verbatim>]
voice_first_violations:
  # Places where the script reads p-values aloud, reads a region acronym un-spelled-out on first use, or otherwise breaks voice-first rules.
  - quote: <the offending phrase>
    why: <e.g., "reads 'p less than 0.001' aloud — should describe effect direction and magnitude" or "uses 'dlPFC' on first appearance without spelling out 'dorsolateral prefrontal cortex'">
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
