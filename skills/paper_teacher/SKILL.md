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

## Stage 2 — write the script with mandatory coverage

8. Invoke `paperteacher.teach_from_outline(arxiv_id=..., mode="two_host")`
   (or `mode="single_host"` if step 9 says so). This loads the saved outline
   and returns a long instruction with strict coverage requirements: every
   `critical` item gets full decomposition, every `important` item gets the
   trick + bridge, every `mention` item gets one substantive sentence.
   Banned phrases are listed.
9. Produce the script. **Default is `mode="two_host"`** — Person1 is the
   "professor friend" mentor (warm, opinionated, broader-context takes,
   honest about what's actually new vs. clever reframing). Person2 is a
   peer-level interlocutor (clarifying, challenging, occasionally wrong
   and corrected — never cheerleading). Each turn is 1-3 sentences for
   natural cadence. Single-host is for episodes where the math is so
   dense that two-voice banter would dilute coverage; default to two_host.
   Target ~1750 words (~10 min at the configured speaking rate).
10. Call `paperteacher.save_script(arxiv_id=..., script=...)`.

## Stage 3 — audit, and regenerate if glossed

11. Invoke `paperteacher.audit_coverage(arxiv_id=...)`. This loads outline
    + script and returns a YAML report ending with:
    `recommendation: ship | regenerate_with_gaps | regenerate_from_scratch`.
12. If `recommendation: ship`, proceed. If `regenerate_with_gaps`, re-invoke
    `teach_from_outline` with the gaps prepended to the prompt as additional
    forced-coverage items, save the new script, and audit ONE more time
    (no infinite loop). If `regenerate_from_scratch`, the outline itself is
    probably wrong — re-run stage 1, then stage 2, then audit.
13. Don't ship a script that audited as `poor`. Send a text-only message
    explaining why the paper was hard to teach and skip the voice note.

## Stage 4 — render and deliver

14. Call `paperteacher.render_audio(arxiv_id=..., mode="two_host")`. The
    server loads the saved script and renders via the configured TTS backend
    (defaults to Kokoro; set `PAPERTEACHER_TTS=vertex` or pass `backend="vertex"`
    to use Google Vertex AI Chirp 3 HD voices). Default speaking rate is 1.1x
    for a livelier pace. Returns an mp3 path.
15. Send to WhatsApp as TWO messages:
    - First (text): `*{title}* — {2-sentence hook}.\nhttps://arxiv.org/abs/{arxiv_id}`
    - Second (voice note): the mp3 from step 14.
16. Call `paperteacher.mark_seen(arxiv_id=..., title=..., note="audit:complete",
    tags=[...])`. Use 2-4 topic tags from a stable vocabulary
    (`info-geometry`, `optimization`, `attention`, `rl`, `interpretability`,
    `architecture`, `theory`, `empirical`, etc.) so `topic_distribution` is
    meaningful across episodes.

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
