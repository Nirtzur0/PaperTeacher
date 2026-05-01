"""MCP server: paper discovery + reading + 3-stage teaching pipeline + audio rendering.

The pipeline is decompose-then-execute: stage 1 extracts a structured outline of
every concept and equation, stage 2 writes the script with that outline as a
mandatory coverage contract, stage 3 audits the script against the outline.

The MCP host (OpenClaw, Claude Code, etc.) runs each stage's prompt and uses
this server's tools to persist intermediate state.
"""
from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import audio, discovery, outlines, prompts, reader, state

mcp = FastMCP("paperteacher")

PROFILE_PATH = Path(os.environ.get("PAPERTEACHER_PROFILE", "config/profile.md"))


def _profile_text() -> str:
    if PROFILE_PATH.exists():
        return PROFILE_PATH.read_text()
    return "(no profile.md found — using generic ML/CS defaults)"


# ---------- resources ----------


@mcp.resource("profile://taste")
def taste_profile() -> str:
    """The listener's taste profile (fields, known topics, voice preferences)."""
    return _profile_text()


@mcp.resource("outline://{arxiv_id}")
def outline_resource(arxiv_id: str) -> str:
    """Load the saved outline for a paper. Empty if not yet extracted."""
    body = outlines.load_outline(arxiv_id)
    return body or f"(no outline saved for {arxiv_id} — run the extract_outline prompt first)"


# ---------- tools: discovery + reading ----------


@mcp.tool()
async def fetch_trending_papers(
    arxiv_categories: list[str] | None = None,
    limit: int = 20,
) -> list[dict]:
    """Discover candidate papers from HF Daily + arXiv RSS. Filters out seen papers."""
    cands = await discovery.discover(arxiv_categories=arxiv_categories or [], limit=limit)
    return [c.to_dict() for c in cands if not state.is_seen(c.arxiv_id)]


@mcp.tool()
async def read_paper(arxiv_id: str, max_chars: int = 120_000) -> dict:
    """Fetch full text via fallback: arXiv HTML -> HF papers -> arXiv abstract."""
    p = await reader.read_paper(arxiv_id, max_chars=max_chars)
    return {
        "arxiv_id": p.arxiv_id,
        "title": p.title,
        "text": p.text,
        "source": p.source,
        "truncated": p.truncated,
    }


# ---------- tools: pipeline state ----------


@mcp.tool()
def save_outline(arxiv_id: str, outline_yaml: str) -> dict:
    """Persist the YAML outline produced by stage 1 (extract_outline)."""
    p = outlines.save_outline(arxiv_id, outline_yaml)
    return {"ok": True, "path": str(p)}


@mcp.tool()
def get_outline(arxiv_id: str) -> dict:
    """Load the outline previously saved for this paper."""
    body = outlines.load_outline(arxiv_id)
    return {"ok": body is not None, "outline_yaml": body or ""}


@mcp.tool()
def save_script(arxiv_id: str, script: str) -> dict:
    """Persist the script produced by stage 2 (teach_from_outline)."""
    p = outlines.save_script(arxiv_id, script)
    return {"ok": True, "path": str(p)}


@mcp.tool()
def get_script(arxiv_id: str) -> dict:
    """Load the script previously saved for this paper."""
    body = outlines.load_script(arxiv_id)
    return {"ok": body is not None, "script": body or ""}


@mcp.tool()
def list_seen() -> list[dict]:
    """Papers already delivered."""
    return state.list_seen()


@mcp.tool()
def mark_seen(arxiv_id: str, title: str = "", note: str = "") -> dict:
    """Record that a paper has been delivered."""
    state.mark_seen(arxiv_id, title=title, note=note)
    return {"ok": True, "arxiv_id": arxiv_id}


# ---------- tools: audio ----------


@mcp.tool()
def render_audio(
    script: str,
    mode: str = "single_host",
    output_format: str = "mp3",
) -> dict:
    """Render a script to audio via local Kokoro-82M.

    mode: "single_host" (denser, math-friendly) or "two_host" (parses
          <Person1>/<Person2> tags and stitches with a short pause).
    output_format: "mp3" or "wav".
    """
    out = audio.render(script=script, mode=mode, output_format=output_format)
    return {"ok": True, "audio_path": str(out)}


# ---------- prompts: the three pipeline stages ----------


@mcp.prompt()
async def extract_outline(arxiv_id: str) -> str:
    """STAGE 1. Read the paper and produce a structured YAML outline of every
    concept and equation. The outline becomes the coverage contract for stage 2.
    """
    p = await reader.read_paper(arxiv_id)
    return prompts.render_extract(
        arxiv_id=arxiv_id,
        title=p.title,
        taste_profile=_profile_text(),
        paper_text=p.text,
    )


@mcp.prompt()
async def teach_from_outline(arxiv_id: str, mode: str = "single_host") -> str:
    """STAGE 2. Write the spoken script using the previously-saved outline as
    a mandatory coverage contract. Run extract_outline + save_outline first.
    """
    outline_yaml = outlines.load_outline(arxiv_id)
    if not outline_yaml:
        return (
            f"ERROR: no outline saved for {arxiv_id}. "
            f"Run the `extract_outline` prompt for this paper, then call "
            f"`save_outline(arxiv_id={arxiv_id!r}, outline_yaml=...)` with the result, "
            f"then re-invoke this prompt."
        )
    p = await reader.read_paper(arxiv_id)
    return prompts.render_teach(
        arxiv_id=arxiv_id,
        title=p.title,
        taste_profile=_profile_text(),
        paper_text=p.text,
        outline_yaml=outline_yaml,
        mode=mode,
    )


@mcp.prompt()
def audit_coverage(arxiv_id: str) -> str:
    """STAGE 3. Audit the saved script against the saved outline. Returns a
    YAML report with `recommendation: ship | regenerate_with_gaps | regenerate_from_scratch`.
    """
    outline_yaml = outlines.load_outline(arxiv_id)
    script = outlines.load_script(arxiv_id)
    if not outline_yaml or not script:
        missing = []
        if not outline_yaml:
            missing.append("outline")
        if not script:
            missing.append("script")
        return f"ERROR: missing {', '.join(missing)} for {arxiv_id}. Run earlier stages first."
    return prompts.render_audit(outline_yaml=outline_yaml, script=script)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
