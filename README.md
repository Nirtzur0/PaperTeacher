# PaperTeacher

Daily research-paper podcast, delivered to WhatsApp. A personal NotebookLM,
but actually technical.

One paper a day, picked from HuggingFace Daily Papers and arXiv, taught
through a **3-stage decompose-then-execute pipeline** that forces full
coverage of every equation and concept (no glossing). Narrated locally
by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), delivered by
[OpenClaw](https://openclaw.ai).

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
   ↓                          (problem solved, role of each term, key
   save_outline              trick, geometric picture, numerical example)
   ↓
teach_from_outline ← stage 2: write script with outline as coverage contract
   ↓                          (critical items get full decomposition;
   save_script               banned phrases listed; ~2500-word ceiling)
   ↓
audit_coverage     ← stage 3: blunt audit. recommendation: ship |
   ↓                          regenerate_with_gaps | regenerate_from_scratch
   render_audio (local Kokoro)
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
│ via the MCP host │           │  • save_script  / get_script   │
│ LLM (Claude or   │           │  • render_audio (local Kokoro) │
│ Gemini, the      │           │  • mark_seen / list_seen       │
│ host's choice)   │           │                                │
│                  │           │ prompts (the 3 stages):        │
│ ↓ WhatsApp       │           │  • extract_outline             │
└──────────────────┘           │  • teach_from_outline          │
                               │  • audit_coverage              │
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
  Works in any MCP host: Claude Desktop, Claude Code, OpenClaw.
- **OpenClaw skill** = the orchestrator. Chains the three stages, handles
  cron, sends to WhatsApp. The LLM provider is OpenClaw's choice — Gemini
  API key, Claude subscription via the host, etc.
- **Kokoro-82M** = local TTS. Free, runs on your laptop, no API keys.

## Quick start

```bash
pip install -e .

# fill in your taste profile
$EDITOR config/profile.md

# run the MCP server (stdio)
paperteacher
```

ffmpeg is needed for mp3 (via pydub). `brew install ffmpeg` /
`apt install ffmpeg`. Or pass `output_format="wav"` to `render_audio`.

Wire into Claude Code:

```jsonc
// ~/.claude/mcp.json
{
  "mcpServers": {
    "paperteacher": { "command": "paperteacher" }
  }
}
```

Then in Claude Code, run the pipeline by hand:

```
/extract_outline arxiv_id=2603.20105
# ...Claude produces YAML outline...
# call save_outline tool with the result

/teach_from_outline arxiv_id=2603.20105 mode=single_host
# ...Claude produces script...
# call save_script tool

/audit_coverage arxiv_id=2603.20105
# ...Claude returns ship | regenerate report...

# call render_audio with the final script
```

Wire into OpenClaw: copy `openclaw/SKILL.md` to
`~/.openclaw/workspace/skills/paper_teacher/SKILL.md`. The skill walks the
host through every step automatically.

## MCP surface

**Tools**
- `fetch_trending_papers(arxiv_categories?, limit?)` — HF Daily + arXiv RSS;
  seen papers filtered server-side.
- `read_paper(arxiv_id, max_chars?)` — full text via fallback chain
  (arXiv HTML → HF papers → arXiv abstract). Always returns; flags source.
- `save_outline(arxiv_id, outline_yaml)` / `get_outline(arxiv_id)`
- `save_script(arxiv_id, script)` / `get_script(arxiv_id)`
- `render_audio(script, mode, output_format?)` — local Kokoro. `mode` is
  `single_host` or `two_host`. `output_format` is `mp3` or `wav`.
- `list_seen()` / `mark_seen(arxiv_id, title?, note?)`

**Prompts (the three pipeline stages)**
- `extract_outline(arxiv_id)` → instructs the host LLM to produce a
  structured YAML outline with every equation decomposed (role of each
  component, key trick, geometric picture, numerical walkthrough).
- `teach_from_outline(arxiv_id, mode?)` → instructs the host LLM to write
  the script using the saved outline as mandatory coverage. Includes the
  full banned-phrases list, voice-first rules, and length ceiling.
- `audit_coverage(arxiv_id)` → instructs the host LLM to blunt-audit the
  saved script against the saved outline; returns a YAML report ending in
  `recommendation: ship | regenerate_with_gaps | regenerate_from_scratch`.

**Resources**
- `profile://taste` — the listener's taste profile (markdown).
- `outline://{arxiv_id}` — saved outline for any paper.

## Why this shape

Other open NotebookLM clones (podcastfy, open-notebooklm, NotebookLlama)
ship a single PDF-in / mp3-out flow with a generic two-host script writer.
The shallow banter is what makes them feel weak on technical content.

PaperTeacher inverts that:
- **Decompose-then-execute** for content depth (the 3-stage pipeline).
- **The host LLM (Claude / Gemini) writes every word** — we just structure
  the workflow so it can't avoid the hard parts.
- **Kokoro-82M narrates locally** — no cloud TTS, no API keys, no per-token cost.

The discovery + taste profile + structured pipeline + dedupe + delivery
layer is the actual value-add, and it's what nobody else has built.

## Subscription auth

PaperTeacher itself never calls an LLM API directly — the host (OpenClaw,
Claude Code, Claude Desktop) does. So if the host supports Claude Pro/Max
OAuth, you pay nothing per token; if it uses a Gemini or Anthropic API key,
you pay whatever the host pays. Provider choice lives where it should:
with the host.

TTS runs entirely on your machine via Kokoro — no third-party keys needed.
