# PaperTeacher

A personal research-paper podcast that doesn't gloss the math. One paper a
day, picked from arXiv / HuggingFace / NBER / bioRxiv, taught through a
three-stage pipeline that forces full coverage of every equation, finding,
and identifying assumption — then rendered to a voice note and delivered
over WhatsApp.

If NotebookLM is two cheerful AIs paraphrasing the abstract, this is a
working researcher walking you through every component of every critical
equation, with a stance about what's actually new versus clever reframing.

```
read paper → extract outline → teach script → audit coverage → render audio
   ar5iv    │     stage 1    │   stage 2    │    stage 3    │  Vertex TTS
            │  (Pydantic ✓)  │              │ (Pydantic ✓)  │
```

## Why three stages

Single-shot "read paper → write 15-min script" is exactly where models
gloss the hard math. The fix is to extract the structure first, then make
the script-writing pass treat it as a coverage contract, then audit and
regenerate if anything was skipped.

| Stage | Input | Output | Validated against |
|---|---|---|---|
| 1 — extract | full paper + listener profile | YAML outline (every equation, decomposed) | Pydantic schema on save |
| 2 — teach | outline + paper + profile | spoken script (~2500 words) | banned-phrases + voice rules |
| 3 — audit | outline + script | YAML report: ship \| regen-with-gaps \| regen-from-scratch | Pydantic schema on save |

Stage 1 is where depth is decided. The schema requires **at minimum 3
items marked `critical`**, and every critical equation carries: structure
in words, role of each component (the role, never the symbol), key trick,
geometric picture, numerical walkthrough, and bridge to the next equation.
Stage 2 has zero room to dodge — anything marked critical gets the full
chain spoken aloud, or the audit fails it.

This is structured *workflow*, not handcrafted output. Every word still
comes from the host LLM — we just make it impossible to skip the hard parts.

## Architecture

```
   cron @ 08:00
        │
        ▼
┌──────────────────┐    MCP    ┌──────────────────────────────┐
│ host agent       │◄─────────►│ paperteacher MCP server      │
│ (OpenClaw,       │           │                              │
│  Claude Code,    │           │ tools                        │
│  Claude Desktop) │           │  • fetch_trending_papers     │
│                  │           │  • read_paper                │
│ chains the three │           │  • save_outline / get_outline│
│ pipeline stages  │           │  • save_script  / get_script │
│ via the host's   │           │  • save_audit                │
│ LLM (Pro / Flash │           │  • render_audio (async)      │
│ / Claude / etc.) │           │  • audio_status (poll)       │
│                  │           │  • mark_seen / mark_skipped  │
│                  │           │  • topic_distribution        │
│ ↓ WhatsApp       │           │                              │
└──────────────────┘           │ prompts (the three stages)   │
                               │  • extract_outline           │
                               │  • plan_episode (opt-in)     │
                               │  • teach_from_outline        │
                               │  • audit_coverage            │
                               │                              │
                               │ resources                    │
                               │  • profile://taste           │
                               │  • outline://{arxiv_id}      │
                               │  • voice-guide://{domain}    │
                               └──────────────┬───────────────┘
                                              │ Vertex Chirp 3
                                              │  or Kokoro-82M
                                              ▼
                                            mp3
```

Three things to notice:

1. **PaperTeacher itself never calls an LLM.** The host drives inference;
   the server exposes tools, prompts, and Pydantic-validated saves. Provider
   choice (Gemini Pro, Claude, GPT, local Qwen) lives where it should — with
   the host. Subscription auth lives there too.

2. **`render_audio` is fire-and-poll.** Vertex TTS on a 2500-word script
   takes 60–180s; longer than most MCP timeouts. The tool returns
   immediately with the deterministic path; the host polls `audio_status`
   until `ready=true`. Atomic write means no partial files.

3. **Pydantic validates at the boundary.** Bad LLM output fails loudly with
   a usable error message instead of silently corrupting downstream stages.

## Domain packs

Each pack ships its own outline schema, prompt templates, discovery sources,
and reader. The framework (storage, audio, TTS, MCP server, CLI) is
domain-agnostic.

| Pack | Sources | Outline shape | What's audited |
|---|---|---|---|
| `ml` | HuggingFace Daily, arXiv (cs.LG, stat.ML, math.ST...), Semantic Scholar | equations + concepts, with prior-attempts, ablations, assumption-boundaries | concrete-then-abstract, anti-anthropomorphism, baseline-with-numbers, named techniques operationalized |
| `physics` | arXiv (hep-th, hep-ph, gr-qc, astro-ph, cond-mat, quant-ph), INSPIRE-HEP | equations with **dimensional check + limiting case + symmetries / Noether + Fermi estimate**, observables with falsifiability | no tensor notation aloud, regime named explicitly, "by symmetry" requires naming the symmetry |
| `neuro` | bioRxiv, Europe PMC, Semantic Scholar | findings with **method + key control by name**, behavioral tasks, subjects | what method physically measures, control alternative-explanation-then-logic, no p-values aloud |
| `econ` | NBER, arXiv (econ.*, q-fin.*) | identification block + specifications (voice_description) + estimates (economic_translation) + robustness checks | "X causes Y" requires identification in same beat, every coefficient paired with economic translation |

Adding a new domain is four files (`models.py`, `prompts.py`, `discovery.py`,
`reader.py`) and one `register_domain("name", ...)` call. The factory in
`domains/_common.make_domain()` wires the protocol.

## Listener profile

Your preferences live at `~/.paperteacher/profile.md` as markdown with a
few structured fields. **The host should never ask you about length or
mode — they're answered here.**

```yaml
name: ...
fields:
  - machine learning theory
  - mathematical ML (information geometry, optimal transport)
  - reasoning and interpretability

known_well:
  - transformers, attention, scaling laws
  - convex and stochastic optimization

skip_unless_unusually_strong:
  - benchmark-only papers without theory
  - prompt-engineering papers

voice:
  - PhD friend who connects fields
  - intuition before formalism
  - geometric and physical analogies preferred

# structured fields — read by code:
domains: ml, physics, neuro, econ
length_target_minutes: 15
script_mode: two_host
speaking_rate: 1.1
```

The free-form sections drive style; the structured fields drive
mechanics. Resolution order: `PAPERTEACHER_*` env var → profile line →
framework default. See `config/profile.example.md`.

## Quick start

```bash
# core install (no TTS yet — pick one below)
pip install -e .

# pick a TTS backend
pip install -e '.[tts-kokoro]'   # local, free, no keys
pip install -e '.[tts-vertex]'   # Google Vertex AI; auth via gcloud ADC

# copy the example profile and edit
mkdir -p ~/.paperteacher
cp config/profile.example.md ~/.paperteacher/profile.md
$EDITOR ~/.paperteacher/profile.md

# select a backend (default is kokoro)
export PAPERTEACHER_TTS=vertex
```

For Vertex TTS, run `gcloud auth application-default login` once — no API
key needed.

## End-to-end CLI run (no MCP host)

```bash
paperteacher-cli discover --categories cs.LG,stat.ML
paperteacher-cli prompt extract 2603.20105 | your-llm > outline.yaml
paperteacher-cli save-outline 2603.20105 -f outline.yaml
paperteacher-cli prompt teach 2603.20105   | your-llm > script.txt
paperteacher-cli save-script 2603.20105 -f script.txt
paperteacher-cli prompt audit 2603.20105   | your-llm > audit.yaml
paperteacher-cli save-audit 2603.20105 -f audit.yaml
paperteacher-cli render 2603.20105 --backend vertex
paperteacher-cli seen mark 2603.20105
```

Each `prompt` command renders to stdout; pipe through any LLM. Each `save-*`
command runs Pydantic validation and writes to `~/.paperteacher/`.

## MCP wiring

### Claude Code / Claude Desktop

```jsonc
// ~/.claude/mcp.json
{ "mcpServers": { "paperteacher": { "command": "paperteacher" } } }
```

The three stage prompts appear as slash commands. The host walks
`extract → save_outline → teach → save_script → audit → save_audit →
render_audio → audio_status → done`.

### OpenClaw (the daily-cron path)

```jsonc
// ~/.openclaw/openclaw.json
{
  "skills": {
    "load": { "extraDirs": ["/abs/path/to/PaperTeacher/skills"] }
  },
  "mcp": {
    "servers": {
      "paperteacher": {
        "command": "uv",
        "args": ["run", "--extra", "tts-vertex", "--directory",
                 "/abs/path/to/PaperTeacher",
                 "python", "-m", "paperteacher.server"],
        "env": { "PAPERTEACHER_TTS": "vertex" }
      }
    }
  }
}
```

`skills/paper_teacher/SKILL.md` chains every stage end-to-end and ships the
audio over WhatsApp. The MCP server runs in-place via `uv run` — pull the
repo, edit, restart OpenClaw.

For OpenClaw's main agent to know to invoke the skill on "give me a paper",
add a paper-teacher block to `~/.openclaw/workspace/AGENTS.md` telling it
to read `profile://taste` and never ask about length / mode. Otherwise the
chat lane will helpfully ask you preference questions whose answers live
in your profile.

## State on disk

```
~/.paperteacher/
├── profile.md                  # your taste / preferences (you write this)
├── preferred.yaml              # optional preferred-authors allowlist
├── seen.jsonl                  # delivered papers, with topic tags
├── skipped.jsonl               # the backlog (considered but not picked)
├── pipeline.jsonl              # structured event log
├── meta/{id}.json              # which domain pack handled which paper
├── outlines/{id}.yaml          # stage-1 output, Pydantic-validated
├── plans/{id}.yaml             # stage-1.5 output (opt-in planner)
├── scripts/{id}.txt            # stage-2 output
├── audits/{id}.yaml            # stage-3 output, Pydantic-validated
└── audio/paper_{id}.mp3        # rendered episodes
```

All filesystem. No DB. `cat`-debuggable.

## Preferred authors (optional)

Bias discovery toward specific researchers. When `~/.paperteacher/preferred.yaml`
exists, every candidate whose authors substring-match a configured name
gets a score boost so the host's selection step sees those papers near the
top. Off when the file is absent.

```bash
cp config/preferred.example.yaml ~/.paperteacher/preferred.yaml
$EDITOR ~/.paperteacher/preferred.yaml
```

Match is on author *name* — arXiv RSS doesn't expose institution, so to
bias toward a lab list its key researchers.

## TTS

Two backends, same render path:

| Backend | Cost | Quality | Setup |
|---|---|---|---|
| `kokoro` | free, local | solid for monologue, Apple Silicon ~real-time | `pip install -e '.[tts-kokoro]'` |
| `vertex` | per-character (Chirp 3 HD) | studio-grade prosody, two-host clean | `gcloud auth application-default login` |

Select via `PAPERTEACHER_TTS=kokoro|vertex` or `--backend` on the CLI. The
Vertex sync API has a 5000-byte chunk limit; the backend chunks
transparently and stitches the audio.

## MCP surface

**Tools** &nbsp; `fetch_trending_papers` · `read_paper` · `save_outline` ·
`get_outline` · `save_plan` · `get_plan` · `save_script` · `get_script` ·
`save_audit` · `render_audio` · `audio_status` · `list_seen` · `mark_seen` ·
`list_skipped` · `mark_skipped` · `topic_distribution`

**Prompts** &nbsp; `extract_outline` · `plan_episode` *(optional, per-pack
opt-in)* · `teach_from_outline` · `audit_coverage`

**Resources** &nbsp; `profile://taste` · `outline://{arxiv_id}` ·
`voice-guide://{domain}`

`save_*` calls run Pydantic validation and return decision-shaped responses
(e.g. `save_audit` returns `recommendation`, `coverage_status`, plus
counts — not the full per-item dump). The host can branch without re-parsing.

## Why this shape

Other open NotebookLM clones (podcastfy, open-notebooklm, NotebookLlama)
ship a single PDF-in / mp3-out flow with a generic two-host script writer.
The shallow banter is what makes them feel weak on technical content.

PaperTeacher inverts that:

- **Decompose-then-execute** for content depth (the three-stage pipeline).
- **Pydantic schemas validate stage-1 and stage-3 LLM output at the
  boundary** — schema-mismatched payloads fail loudly with usable errors,
  not silently three stages later.
- **The host LLM writes every word** — the server just structures the
  workflow so it can't avoid the hard parts. Provider-agnostic, model
  swappable, no LLM API key required by PaperTeacher itself.
- **Domain packs encode subject-specific failure modes** — physics
  enforces dimensional analysis and named limits; neuro enforces named
  controls and what-the-method-physically-measures; econ enforces
  identification before "X causes Y". Each pack's audit checks for its
  own glossing patterns.
- **Pluggable TTS** — Kokoro local for free / private / offline, or
  Vertex AI Chirp 3 HD when you want studio-grade prosody.

## Cost

PaperTeacher itself doesn't call an LLM, so its only direct cost is TTS:

- **Kokoro:** $0. Runs locally.
- **Vertex Chirp 3 HD:** ~$0.016 per 1000 characters. A 2500-word
  episode is ~15K characters → roughly $0.25/day, $7/month.

LLM cost lives with the host — Pro/Flash/Claude pricing applies there.
Provider choice is a host concern, not a PaperTeacher concern.
