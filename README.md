# PaperTeacher

Daily research-paper podcast, delivered to WhatsApp.

A personal NotebookLM, but actually technical. One paper a day, picked from
HuggingFace Daily Papers and arXiv, taught in your voice profile by Claude,
narrated by ElevenLabs, dropped into your chat app by [OpenClaw](https://openclaw.ai).

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
                                         │ podcastfy + ElevenLabs
                                         ▼
                                       mp3
```

- **MCP server** is the reusable brain-adjacent layer: paper discovery,
  full-text reading with fallback, the teaching prompt, audio rendering.
  Works in any MCP host: Claude Desktop, Claude Code, OpenClaw.
- **OpenClaw skill** is just the orchestrator: cron trigger, picks a paper,
  drives Claude through the prompt, sends the result to WhatsApp.
- **Claude** does discovery selection and script writing. Single-host by
  default (denser, better for math); two-host as a config flag.

## Quick start

```bash
pip install -e .

# fill in your taste profile
$EDITOR config/profile.md

# set keys
export ELEVENLABS_API_KEY=...        # for audio
export ANTHROPIC_API_KEY=...         # only if you're not using subscription auth

# run the MCP server (stdio)
paperteacher
```

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
- `render_audio(script, mode, tts_model?)` — script → mp3 via podcastfy.
  `mode` is `single_host` or `two_host`.

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

PaperTeacher uses **podcastfy as the audio engine only** — Claude (with
your teaching prompt) writes the script, podcastfy just does TTS and
stitching. That's where the depth comes from.

The discovery + taste profile + dedupe + delivery layer is the actual
value-add and it's what nobody else has built.

## Subscription auth

If your MCP host (OpenClaw, Claude Code) supports Claude Pro/Max OAuth, you
don't need an Anthropic API key — the host authenticates and you pay
nothing per token. Check your host's docs. PaperTeacher itself never calls
the Anthropic API directly; the host does.

ElevenLabs and OpenAI TTS still need their own keys.
