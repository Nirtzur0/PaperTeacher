"""MCP server exposing paper discovery, reading, the teach_paper prompt, and audio rendering."""
from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import audio, discovery, prompts, reader, state

mcp = FastMCP("paperteacher")

PROFILE_PATH = Path(os.environ.get("PAPERTEACHER_PROFILE", "config/profile.md"))
PODCAST_CONFIG = Path(os.environ.get("PAPERTEACHER_PODCAST_CONFIG", "config/podcast_config.yaml"))


def _profile_text() -> str:
    if PROFILE_PATH.exists():
        return PROFILE_PATH.read_text()
    return "(no profile.md found — using generic ML/CS defaults)"


# ---------- resources ----------


@mcp.resource("profile://taste")
def taste_profile() -> str:
    """The listener's taste profile (fields, known topics, voice preferences)."""
    return _profile_text()


# ---------- tools ----------


@mcp.tool()
async def fetch_trending_papers(
    arxiv_categories: list[str] | None = None,
    limit: int = 20,
) -> list[dict]:
    """Discover candidate papers. Returns ranked list combining HF Daily + arXiv RSS.

    arxiv_categories: optional list like ["cs.LG", "stat.ML", "math-ph"].
    Already-seen papers are filtered out.
    """
    cands = await discovery.discover(
        arxiv_categories=arxiv_categories or [],
        limit=limit,
    )
    return [c.to_dict() for c in cands if not state.is_seen(c.arxiv_id)]


@mcp.tool()
async def read_paper(arxiv_id: str, max_chars: int = 120_000) -> dict:
    """Fetch the paper's full text using fallback: arXiv HTML -> HF -> arXiv abstract."""
    p = await reader.read_paper(arxiv_id, max_chars=max_chars)
    return {
        "arxiv_id": p.arxiv_id,
        "title": p.title,
        "text": p.text,
        "source": p.source,
        "truncated": p.truncated,
    }


@mcp.tool()
def list_seen() -> list[dict]:
    """All papers that have already been delivered."""
    return state.list_seen()


@mcp.tool()
def mark_seen(arxiv_id: str, title: str = "", note: str = "") -> dict:
    """Record that a paper has been delivered. Call after successful render+send."""
    state.mark_seen(arxiv_id, title=title, note=note)
    return {"ok": True, "arxiv_id": arxiv_id}


@mcp.tool()
def render_audio(
    script: str,
    mode: str = "single_host",
    tts_model: str = "elevenlabs",
) -> dict:
    """Render a script to mp3. mode is "single_host" or "two_host"."""
    out = audio.render(
        script=script,
        mode=mode,
        tts_model=tts_model,
        config_path=PODCAST_CONFIG if PODCAST_CONFIG.exists() else None,
    )
    return {"ok": True, "audio_path": str(out)}


# ---------- prompts ----------


@mcp.prompt()
def teach_paper(arxiv_id: str, mode: str = "single_host") -> str:
    """Render the teaching prompt for a given arXiv id.

    The host (Claude) is expected to call read_paper(arxiv_id) first and pass
    its `text` field as the paper body. This prompt loads the listener's
    taste profile from the profile resource.
    """
    # We embed an instruction for the host on how to fill the paper text.
    return (
        f"Use the `read_paper` tool with arxiv_id={arxiv_id} to fetch the full text, "
        f"then internalize this teaching prompt and write the script. "
        f"Mode: {mode}.\n\n"
        + prompts.render_teach_prompt(
            arxiv_id=arxiv_id,
            title="(read with read_paper)",
            taste_profile=_profile_text(),
            paper_text="(fill in from read_paper.text)",
            mode=mode,
        )
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
