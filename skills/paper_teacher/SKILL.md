---
name: paper-teacher
description: Daily research-paper podcast. Picks one trending arXiv/HF paper, runs a 3-stage decompose-then-execute pipeline (extract structured outline → write script with mandatory coverage of every critical equation → audit and regenerate-if-glossed), renders to audio (Kokoro local or Vertex AI Chirp 3), and delivers a text hook + voice note over WhatsApp. Decompose-then-execute beats single-shot summarization for technical depth.
homepage: https://github.com/Nirtzur0/PaperTeacher
metadata: {"openclaw":{"emoji":"📄","requires":{"bins":["uv","ffmpeg"]}}}
schedule:
  cron: "0 8 * * *"
  timezone: "local"
mcp_servers:
  - paperteacher
delivery:
  channel: whatsapp
  recipient: "@me"
---

# Daily paper teacher (3-stage pipeline)

Once a day, run this end-to-end. Every step is a small, named action; do not
combine them in your head.

## Stage 0 — pick the paper

1. Call `paperteacher.fetch_trending_papers` with
   `arxiv_categories=["cs.LG", "cs.CL", "cs.AI", "stat.ML", "math.ST", "math.OC"]`.
   Already-seen and already-skipped papers are filtered server-side.
2. Read the listener's profile from the `profile://taste` resource — load
   it once now; the plan and teach prompts no longer inline it (saves
   ~1.5K redundant tokens per pipeline run).
3. Read the active pack's voice guide from the `voice-guide://<domain>`
   resource (e.g. `voice-guide://ml`). The teach prompt references it
   instead of re-shipping the ~1K-token pronunciation/banned-phrases table
   on every (re)generation. Apply the rules to every line you emit.
4. Call `paperteacher.topic_distribution(window=30)` to see which topic
   tags have been over- or under-represented in recent deliveries. Favor
   underrepresented topics — keep the diet diverse over time.
5. Pick ONE candidate that best matches `selection_bias` in the profile
   AND is not over-saturated in the topic distribution. Prefer mathematical
   depth, surprising ideas, last 2-4 weeks. If nothing matches, take the
   highest-scored candidate and note that in the hook.
6. For every candidate you considered but did NOT pick, call
   `paperteacher.mark_skipped(arxiv_id=..., title=..., tags=[...],
   reason=...)`. This builds a backlog you can revisit on slow news days.
7. Call `paperteacher.read_paper(arxiv_id=...)`. If `source == "arxiv_abs"`
   you only have the abstract — proceed but flag this in stage 1's outline
   and in the WhatsApp hook.

## Stage 1 — extract the outline (THIS IS WHERE DEPTH IS DECIDED)

5. Invoke the MCP prompt `paperteacher.extract_outline(arxiv_id=...)`.
   This returns a long instruction asking you to produce a structured YAML
   outline of every significant equation and concept, with full decomposition
   for the critical ones (problem solved, structure in words, role of each
   component, key trick, geometric picture, numerical walkthrough).
6. Produce the YAML. Be exhaustive on equations — if the paper has 12
   numbered equations, your outline references all 12. Better to over-extract.
7. Call `paperteacher.save_outline(arxiv_id=..., outline_yaml=...)`.
7a. **Claim the paper now** — call `paperteacher.mark_seen(arxiv_id=..., title=...,
    note="outline:saved", tags=[...])` *immediately* after the outline lands.
    This prevents the same paper looping back tomorrow if any later step
    (plan, script, audit, render, delivery) fails. If you legitimately abandon
    mid-run, switch to `mark_skipped` with a reason — DO NOT leave the paper
    unmarked.

## Stage 1.5 — plan the arc (recommended; prevents template-feel)

The teach prompt's default structure is paper-shaped (it gives you an arc
menu — mystery / build / detective / taxonomy — and tells you to pick or
invent). Without a plan, the model picks an arc inline during teach. With
a plan, the planner stage gets dedicated thinking budget for arc design
and produces *committed takes* the teach stage executes — which is the
single biggest lever against episodes that feel scripted or interchangeable.

8. Invoke `paperteacher.plan_episode(arxiv_id=...)`. Returns a long
   instruction asking you to design segments, callbacks, takes (2-5
   committed opinions, AT LEAST ONE methodological), and where this
   paper sits in the field.
9. Produce the YAML plan. Pick segment count for THIS paper (5-9
   typical; a dense theory paper might want 10, a sharp position paper
   4). Make sure at least one take is methodological — "the gain is from
   data filtering, not the architecture; section 4.2 makes this obvious"
   beats "this is an important contribution".
10. Call `paperteacher.save_plan(arxiv_id=..., plan_yaml=...)`. The teach
    prompt automatically picks up the plan from disk on the next call.

Skip this stage only when the paper's structure is genuinely obvious
(e.g., a single-finding bioRxiv preprint where there's nothing to plan).
For ML / physics / econ / theory-heavy papers — run it.

## Stage 2 — write the script with mandatory coverage

11. Invoke `paperteacher.teach_from_outline(arxiv_id=..., mode="two_host")`
    (or `mode="single_host"` if step 12 says so). Loads the saved outline
    AND any saved plan from stage 1.5, then returns a long instruction with
    strict coverage requirements: every `critical` item gets full
    decomposition, every `important` item gets the trick + bridge, every
    `mention` item gets one substantive sentence. Banned phrases are listed.
    The structure block is paper-shaped (arc menu, not 7-section template).
12. Produce the script. **Default is `mode="two_host"`** — Person1 is the
    "professor friend" mentor (warm, opinionated, broader-context takes,
    honest about what's actually new vs. clever reframing). Person2 is a
    peer-level interlocutor (clarifying, challenging, occasionally wrong
    and corrected — never cheerleading). Each turn is 1-3 sentences for
    natural cadence. Single-host is for episodes where the math is so
    dense that two-voice banter would dilute coverage; default to two_host.
13. Call `paperteacher.save_script(arxiv_id=..., script=...)`.

## Stage 3 — audit, and regenerate if glossed

14. Invoke `paperteacher.audit_coverage(arxiv_id=...)`. Loads outline + script
    and returns a YAML report ending with:
    `recommendation: ship | regenerate_with_gaps | regenerate_from_scratch`.
15. If `recommendation: ship`, proceed. If `regenerate_with_gaps`, re-invoke
    `teach_from_outline` with the gaps prepended to the prompt as additional
    forced-coverage items, save the new script, and audit ONE more time
    (no infinite loop). If `regenerate_from_scratch`, the outline itself is
    probably wrong — re-run stage 1, then stage 2, then audit.
16. Don't ship a script that audited as `poor`. Send a text-only message
    explaining why the paper was hard to teach and skip the voice note.

## Stage 4 — render and deliver

17. Call `paperteacher.render_audio(arxiv_id=..., mode="two_host")`. This
    KICKS OFF the render in a background thread and returns IMMEDIATELY with
    the deterministic path (`paper_<arxiv_id>.mp3`) and `status: "rendering"`.
    Vertex Chirp 3 HD on a ~2500-word two_host script takes 60–180 seconds;
    OpenClaw's MCP tool timeout is shorter than that, which is why we run the
    work in the background.

18. **Poll `paperteacher.audio_status(arxiv_id=...)` until `ready=true`**
    before sending the voice note. Suggested cadence: wait 60s, then poll
    every 15–20s for up to 5 minutes total. Three outcomes:
    - `ready=true` with `path` and `size_bytes` → audio is on disk, proceed
      to step 19.
    - `ready=false, status="rendering"` → keep waiting; a long script + a
      slow Vertex region can legitimately take 3 minutes.
    - `ready=false, status="error"` → the background render hit an exception
      (the `error` field has the message). Send the text hook (step 19a) but
      skip the voice note, and tell the user briefly why audio failed.

19. Send to WhatsApp as TWO SEPARATE messages, in this exact order. This is a
    HARD ordering requirement — the listener needs to know what they're about
    to hear before the audio starts playing in their headphones. Do not merge
    them, do not invert them, do not skip the text.

    a. **First, the text hook** — send this and wait for it to deliver
       before moving on. WhatsApp markup: `*bold*`, `_italic_`, `•` for
       bullets, blank line for paragraph break. NO Markdown `**bold**` or
       `## headers` — they don't render. Emoji selectively: 📄 once at
       the top, 🎧 for the audio tag, 🔗 for the link. Nothing else.
       Format:
       ```
       📄 *{title}*
       _{first author} et al. · arXiv:{arxiv_id}_

       {2-3 sentence hook drawn from outline.core_thesis and
       outline.stake_claim — the surprising claim, then why it matters.
       NEVER a paraphrase of the abstract.}

       *Highlights*
       • {one-line key finding or result, with the number if there is one}
       • {one-line method / mechanism / what's clever about it}
       • {one-line caveat, limitation, or what's still open}

       🎧 *{N} min · {single_host|two_host}*
       🔗 https://arxiv.org/abs/{arxiv_id}
       ```
       Hook content rules:
       - Pull from the outline, not the abstract. The outline is the
         structured truth; the abstract is the marketing.
       - Highlights are 3 bullets. Pick the most concrete, specific lines
         you can — a number, a named technique, a real caveat. Generic
         bullets ("the model performs well") are a failure.
       - Italics for the byline, bold for the title and section headers
         (Highlights, the duration tag). Don't bold whole sentences.

    b. **Then, the voice note** — send the mp3 at the path returned by
       `audio_status` (or by step-17's `audio_path`, same value) as the
       second message. Only after the text has been delivered AND
       `audio_status.ready=true` came back.
20. Call `paperteacher.mark_seen(arxiv_id=..., title=..., note="audit:complete",
    tags=[...])` again with the final tags and `audit:complete` note so the
    delivery is recorded. (The earlier 7a call already prevents loops; this
    one annotates the canonical record with topic tags for
    `topic_distribution`.) Use 2-4 topic tags from a stable vocabulary
    (`info-geometry`, `optimization`, `attention`, `rl`, `interpretability`,
    `architecture`, `theory`, `empirical`, etc.).

## Failure modes

- Any tool error: send a single text describing what failed at which step.
  Do NOT retry silently — the user wants to know about gaps.
- arXiv HTML 404 → the reader falls back automatically; flag in the hook
  ("Note: only the abstract was available, depth is limited.").
- Audit returns `poor` after one regeneration → ship text-only with a
  candid explanation. Do not deliver a glossed voice note.

## Conversational follow-up (Voice Wake / Talk Mode)

If the user replies to the voice note with a question, stay in the context
of TODAY's paper:
- Use `paperteacher.get_outline(arxiv_id=...)` to ground answers in the
  structured decomposition rather than paraphrasing the script.
- Use `paperteacher.read_paper(arxiv_id=...)` for direct quotes when the
  user asks "what does the paper actually say about X".
- Use `paperteacher.fetch_trending_papers(...)` if the user asks for
  related work.

## Style invariants (apply to every script you write)

- Voice-first. No LaTeX. No symbolic equation reading ("sigma sub i" is banned).
- Intuition before formalism, ALWAYS.
- 15-minute target, ~2500 words. Cut anything that doesn't earn its place.
- Honest about what's subtle. "This part is genuinely subtle" beats faking confidence.
- Banned phrases: "delve", "fascinating", "in conclusion", "let's explore",
  "in the realm of", and any phrase listed in the outline's `banned_glosses`.
