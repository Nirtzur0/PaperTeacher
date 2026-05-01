# PaperTeacher

Daily research-paper podcast for WhatsApp. A personal NotebookLM, but actually
technical.

One paper a day, picked from HuggingFace Daily Papers and arXiv, taught
through a **3-stage decompose-then-execute pipeline** that forces full
coverage of every equation and concept (no glossing). Narrated by your
choice of [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (local,
free) or Google Vertex AI Text-to-Speech (Chirp 3 HD voices). Delivered
by [OpenClaw](https://openclaw.ai).

## Why decompose-then-execute

Single-shot "read paper → write 15-min script" is where models gloss the
hard math. The fix is to extract a structured outline of every equation
and concept FIRST, then make the script-writing pass treat that outline
as a mandatory coverage contract, then audit and regenerate if anything
was skipped.

```
read_paper
   ↓
extract_outline    ← stage 1: enumerate every equation, decompose each
   ↓                          (problem solved, role of each component, key
   save_outline              trick, geometric picture, numerical example).
   ↓                         Validated against a Pydantic schema on save.
teach_from_outline ← stage 2: write script with outline as coverage contract
   ↓                          (critical items get full decomposition;
   save_script               banned phrases listed; ~2500-word ceiling).
   ↓
audit_coverage     ← stage 3: blunt audit. recommendation: ship |
   save_audit                 regenerate_with_gaps | regenerate_from_scratch.
   ↓
render_audio (Kokoro local | Vertex AI TTS)
   ↓
WhatsApp: text hook + voice note
```

This is structured *workflow*, not handcrafted output. Every word still
comes from the model — we just make it impossible to skip the hard parts.

## Architecture

```
  cron @ 08:00
       │
       ▼
┌──────────────────┐    MCP    ┌────────────────────────────────┐
│ OpenClaw skill   │◄─────────►│ paperteacher MCP server        │
│ (openclaw/       │           │                                │
│  SKILL.md)       │           │ tools:                         │
│                  │           │  • fetch_trending_papers       │
│ chains the 3     │           │  • read_paper                  │
│ pipeline stages  │           │  • save_outline / get_outline  │
│ via the host's   │           │  • save_script  / get_script   │
│ LLM (the host's  │           │  • save_audit                  │
│ choice — Claude  │           │  • render_audio (local Kokoro) │
│ subscription,    │           │  • mark_seen / list_seen       │
│ Gemini API key,  │           │                                │
│ etc.)            │           │ prompts (the 3 stages):        │
│                  │           │  • extract_outline             │
│ ↓ WhatsApp       │           │  • teach_from_outline          │
└──────────────────┘           │  • audit_coverage              │
                               │                                │
                               │ resources:                     │
                               │  • profile://taste             │
                               │  • outline://{arxiv_id}        │
                               └──────────────────┬─────────────┘
                                                  │ Kokoro-82M (local)
                                                  ▼
                                                mp3
```

- **MCP server** = thin tools + the three pipeline prompts. Provider-agnostic.
  The LLM call lives in the host. Pydantic-validated saves catch bad LLM
  output at the boundary, not three stages later.
- **OpenClaw skill** = the orchestrator. Chains the three stages and a single
  retry loop, handles cron, sends to WhatsApp.
- **TTS** = pluggable backend (`paperteacher.tts`). Pick one:
  - `kokoro` — local Kokoro-82M. Free, runs on your laptop, no API keys.
  - `vertex` — Google Vertex AI Chirp 3 HD voices. Higher quality, needs ADC.
  Select via `PAPERTEACHER_TTS=kokoro|vertex` or the `--backend` CLI flag.

## Project layout

```
paperteacher/
├── __init__.py
├── paths.py        # filesystem layout + tunable constants
├── models.py       # Pydantic: Outline, Equation, Concept, AuditReport, …
├── storage.py      # unified persistence (seen / outlines / scripts / audits)
├── discovery.py    # HF Daily + arXiv RSS
├── reader.py       # arXiv HTML → HF → arXiv abstract fallback
├── prompts.py      # the three stage prompts
├── tts.py          # TTS backend abstraction (Kokoro | Vertex AI)
├── audio.py        # script orchestration: stitching, mp3/wav, atomic write
├── server.py       # MCP server (entry: `paperteacher`)
└── cli.py          # standalone CLI (entry: `paperteacher-cli`)

skills/paper_teacher/
└── SKILL.md        # OpenClaw skill (loaded via skills.load.extraDirs)
```

State lives under `~/.paperteacher/` (override with `PAPERTEACHER_HOME`):
- `profile.md` — listener taste profile (copy from `config/profile.example.md`)
- `seen.jsonl` — delivered papers
- `outlines/{id}.yaml` — stage 1 output, validated
- `scripts/{id}.txt` — stage 2 output
- `audits/{id}.yaml` — stage 3 output, validated
- `audio/paper_{arxiv_id}.mp3` — rendered episodes
- `pipeline.jsonl` — structured event log

## Quick start

```bash
# core install (no TTS yet — pick one below)
pip install -e .

# pick a TTS backend (you can install both and switch via env var)
pip install -e '.[tts-kokoro]'   # local, free, no keys
pip install -e '.[tts-vertex]'   # Google Vertex AI; auth via ADC

# copy the example profile to ~/.paperteacher/profile.md and edit it
mkdir -p ~/.paperteacher
cp config/profile.example.md ~/.paperteacher/profile.md
$EDITOR ~/.paperteacher/profile.md

# select a backend (default is kokoro)
export PAPERTEACHER_TTS=vertex

# one-paper end-to-end (manual, no MCP host) ----------------------------
paperteacher-cli discover --categories cs.LG,stat.ML
paperteacher-cli prompt extract 2603.20105 | your-llm > outline.yaml
paperteacher-cli save-outline 2603.20105 -f outline.yaml
paperteacher-cli prompt teach 2603.20105 | your-llm > script.txt
paperteacher-cli save-script 2603.20105 -f script.txt
paperteacher-cli prompt audit 2603.20105 | your-llm > audit.yaml
paperteacher-cli save-audit 2603.20105 -f audit.yaml
paperteacher-cli render 2603.20105 --backend vertex
paperteacher-cli seen mark 2603.20105
```

ffmpeg is needed for mp3 (via pydub). `brew install ffmpeg` /
`apt install ffmpeg`. Or pass `--output-format wav` to `render`.

For Vertex AI TTS, authenticate once with
`gcloud auth application-default login` (or set
`GOOGLE_APPLICATION_CREDENTIALS` to a service-account JSON path).

### Wire into Claude Code

```jsonc
// ~/.claude/mcp.json
{ "mcpServers": { "paperteacher": { "command": "paperteacher" } } }
```

Then in Claude Code, the three stage prompts appear as slash commands.
The host LLM walks through `extract → save_outline → teach → save_script
→ audit → save_audit → render_audio` automatically.

### Wire into OpenClaw

OpenClaw loads skills directly from the repo via `skills.load.extraDirs` in
`~/.openclaw/openclaw.json` — no copying required. Add two entries:

```jsonc
// ~/.openclaw/openclaw.json
{
  "skills": {
    "load": {
      "extraDirs": [
        "/absolute/path/to/PaperTeacher/skills"
      ]
    }
  },
  "mcp": {
    "servers": {
      "paperteacher": {
        "command": "uv",
        "args": ["run", "--directory", "/absolute/path/to/PaperTeacher",
                 "python", "-m", "paperteacher.server"],
        "description": "PaperTeacher MCP server"
      }
    }
  }
}
```

The skill at `skills/paper_teacher/SKILL.md` chains every stage and handles
the regenerate-on-gaps loop with a hard one-retry limit. The MCP server runs
in-place via `uv run` — pull the repo, edit, restart OpenClaw.

**Coexistence with other OpenClaw projects:** PaperTeacher's MCP server
(`paperteacher`) and skill (`paper-teacher`) are independently named, store
all state under `~/.paperteacher/`, and don't share filesystem or configuration
with sibling projects (e.g. AuctionScanner). Both can run on the same
WhatsApp allowlist and the same cron schedule without interfering.

## MCP surface

**Tools**
- `fetch_trending_papers(arxiv_categories?, limit?)` — HF Daily + arXiv RSS;
  seen papers filtered server-side.
- `read_paper(arxiv_id, max_chars?)` — full text via fallback chain
  (arXiv HTML → HF papers → arXiv abstract). Always returns; flags source.
- `save_outline(arxiv_id, outline_yaml)` — **validates against Pydantic**;
  returns `{ok: false, error, raw_preview}` on schema mismatch so the host can
  re-prompt with the error included.
- `get_outline(arxiv_id)` / `save_script(...)` / `get_script(...)`
- `save_audit(arxiv_id, audit_yaml)` — validates, returns the recommendation
  so the host can branch without re-parsing.
- `render_audio(arxiv_id, mode?, output_format?, backend?)` — render the
  saved script. `mode`: `single_host` | `two_host`. `output_format`: `mp3` |
  `wav`. `backend`: `kokoro` | `vertex` (defaults to `PAPERTEACHER_TTS`).
- `list_seen()` / `mark_seen(...)`

**Prompts (the three pipeline stages)**
- `extract_outline(arxiv_id)` — instructs the host LLM to produce a
  schema-conformant YAML outline (see `paperteacher.models.Outline`).
- `teach_from_outline(arxiv_id, mode?)` — instructs the host LLM to write
  the script using the saved outline as mandatory coverage. Includes the
  banned-phrases list, voice-first rules, and the ~2500-word ceiling.
- `audit_coverage(arxiv_id)` — instructs the host LLM to blunt-audit the
  saved script against the saved outline.

**Resources**
- `profile://taste` — the listener's taste profile (markdown).
- `outline://{arxiv_id}` — saved outline for any paper.

## Why this shape

Other open NotebookLM clones (podcastfy, open-notebooklm, NotebookLlama)
ship a single PDF-in / mp3-out flow with a generic two-host script writer.
The shallow banter is what makes them feel weak on technical content.

PaperTeacher inverts that:
- **Decompose-then-execute** for content depth (the 3-stage pipeline).
- **Pydantic schemas** validate stage-1 and stage-3 LLM output at the
  boundary — schema-mismatched payloads fail loudly with usable errors,
  not silently three stages later.
- **The host LLM (Claude / Gemini) writes every word** — we just structure
  the workflow so it can't avoid the hard parts.
- **Pluggable TTS** — Kokoro local for free / private / offline, or Vertex
  AI Chirp 3 HD when you want studio-grade prosody. Same render path either way.

## Subscription auth

PaperTeacher itself never calls an LLM API directly — the host (OpenClaw,
Claude Code, Claude Desktop) does. So if the host supports Claude Pro/Max
OAuth, you pay nothing per token; if it uses a Gemini or Anthropic API key,
you pay whatever the host pays. Provider choice lives where it should:
with the host.

TTS is a separate decision and is *yours* to make: Kokoro runs entirely on
your machine (no third-party keys), Vertex AI uses Google Cloud ADC and
your project's quota.
