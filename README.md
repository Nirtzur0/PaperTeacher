# PaperTeacher

Daily research-paper podcast, delivered to WhatsApp.

A personal NotebookLM, but actually technical. One paper a day, picked from
HuggingFace Daily Papers and arXiv, taught in your voice profile by Claude,
narrated locally by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(free, runs on your laptop), dropped into your chat app by
[OpenClaw](https://openclaw.ai).

## Architecture

Three pieces, deliberately split so the interesting parts stay portable:

```
  cron @ 08:00
       │
       ▼
┌─────────────────┐    MCP    ┌────────────────────────┐
│ OpenClaw skill  │◄─────────►│ paperteacher MCP server│
│ (this repo:     │           │ (this repo)            │
│  openclaw/      │           │  • fetch_trending      │
│  SKILL.md)      │           │  • read_paper          │
│                 │           │  • teach_paper prompt  │
│ ↓ WhatsApp      │           │  • render_audio        │
└─────────────────┘           │  • taste profile       │
                              └──────────┬─────────────┘
                                         │ Kokoro-82M (local)
                                         ▼
                                       mp3
```

- **MCP server** is the reusable brain-adjacent layer: paper discovery,
  full-text reading with fallback, the teaching prompt, audio rendering.
  Works in any MCP host: Claude Desktop, Claude Code, OpenClaw.
- **OpenClaw skill** is just the orchestrator: cron trigger, picks a paper,
  drives Claude through the prompt, sends the result to WhatsApp.
- **Claude** does discovery selection and script writing. Single-host by
  default (denser, better for math); two-host as a flag (parses
  `<Person1>/<Person2>` tags and stitches with two Kokoro voices).

## Quick start

```bash
pip install -e .

# fill in your taste profile
$EDITOR config/profile.md

# run the MCP server (stdio)
paperteacher
```

ffmpeg is needed for mp3 output (via pydub). Install with `brew install ffmpeg`
or `apt install ffmpeg`. Pass `output_format="wav"` to `render_audio` if you
want to skip ffmpeg.

Wire into Claude Code:

```jsonc
// ~/.claude/mcp.json
{
  "mcpServers": {
    "paperteacher": { "command": "paperteacher" }
  }
}
```

Then in Claude Code:

```
/teach_paper arxiv_id=2603.20105 mode=single_host
```

Wire into OpenClaw: copy `openclaw/SKILL.md` to
`~/.openclaw/workspace/skills/paper_teacher/SKILL.md`. OpenClaw's cron
runner will pick it up.

## MCP surface

**Tools**
- `fetch_trending_papers(arxiv_categories?, limit?)` — HF Daily + arXiv RSS,
  filters out anything in the seen-list.
- `read_paper(arxiv_id, max_chars?)` — full text via the fallback chain
  (arXiv HTML → HF papers → arXiv abstract). Always returns; flags source.
- `list_seen()` / `mark_seen(arxiv_id, title?, note?)` — dedupe state.
- `render_audio(script, mode, output_format?)` — local Kokoro TTS.
  `mode`: `single_host` or `two_host`. `output_format`: `mp3` (default) or `wav`.

**Prompt**
- `teach_paper(arxiv_id, mode?)` — the full teaching prompt, with the
  listener's profile baked in. Returns a prompt the agent should fill by
  calling `read_paper` for the paper text.

**Resource**
- `profile://taste` — the listener's taste profile (markdown).

## Why this shape

Other open NotebookLM clones (podcastfy, open-notebooklm, NotebookLlama)
ship a single PDF-in / mp3-out flow with a generic two-host script writer.
The shallow banter is what makes them feel weak on technical content.

PaperTeacher inverts that: **Claude (with your teaching prompt) writes the
script, Kokoro-82M just narrates it locally.** That's where the depth comes
from. The audio side is ~100 lines of Python — no cloud TTS, no API keys,
no per-token cost.

The discovery + taste profile + dedupe + delivery layer is the actual
value-add and it's what nobody else has built.

## Subscription auth

If your MCP host (OpenClaw, Claude Code) supports Claude Pro/Max OAuth, you
don't need an Anthropic API key — the host authenticates and you pay
nothing per token. Check your host's docs. PaperTeacher itself never calls
the Anthropic API directly; the host does.

TTS runs entirely on your machine via Kokoro — no third-party keys needed.
