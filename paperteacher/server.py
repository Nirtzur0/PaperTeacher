"""MCP server: paper discovery + reading + 3-stage teaching pipeline + audio rendering.

The MCP host (OpenClaw, Claude Code, etc.) drives the LLM. This server:
  - exposes tools for discovery, reading, persistence, and audio rendering;
  - exposes the three pipeline-stage prompts;
  - validates LLM output (outline, audit) at save-time via Pydantic, so
    schema-mismatched payloads fail loudly with usable error messages
    instead of silently corrupting downstream stages.
"""
from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from . import audio, discovery, paths, prompts, reader, storage, tts
from .models import ParseError

log = logging.getLogger(__name__)
mcp = FastMCP("paperteacher")


def _profile_text() -> str:
    if paths.PROFILE_PATH.exists():
        return paths.PROFILE_PATH.read_text()
    return "(no profile.md found — using generic ML/CS defaults)"


# ---- resources ------------------------------------------------------------


@mcp.resource("profile://taste")
def taste_profile() -> str:
    """The listener's taste profile (fields, known topics, voice preferences)."""
    return _profile_text()


@mcp.resource("outline://{arxiv_id}")
def outline_resource(arxiv_id: str) -> str:
    """Saved outline for a paper, or a hint if it hasn't been extracted."""
    body = storage.load_outline_yaml(arxiv_id)
    return body or f"(no outline saved for {arxiv_id} — run the extract_outline prompt first)"


# ---- tools: discovery + reading ------------------------------------------


@mcp.tool()
async def fetch_trending_papers(
    arxiv_categories: list[str] | None = None,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[dict]:
    """Discover candidate papers from HF Daily + arXiv RSS. Filters out seen."""
    cands = await discovery.discover(arxiv_categories=arxiv_categories or [], limit=limit)
    seen = storage.seen_ids()
    return [c.to_dict() for c in cands if c.arxiv_id not in seen]


@mcp.tool()
async def read_paper(arxiv_id: str, max_chars: int = paths.DEFAULT_MAX_PAPER_CHARS) -> dict:
    """Fetch full text via fallback: arXiv HTML -> HF papers -> arXiv abstract."""
    p = await reader.read_paper(arxiv_id, max_chars=max_chars)
    return {
        "arxiv_id": p.arxiv_id,
        "title": p.title,
        "text": p.text,
        "source": p.source,
        "truncated": p.truncated,
    }


# ---- tools: pipeline state -----------------------------------------------


@mcp.tool()
def save_outline(arxiv_id: str, outline_yaml: str) -> dict:
    """Validate + persist the YAML outline from stage 1.

    Returns {ok: false, error: ..., raw: ...} if the YAML doesn't match the
    Outline schema — the host should regenerate stage 1 with the error included.
    """
    try:
        path, outline = storage.save_outline(arxiv_id, outline_yaml)
    except ParseError as e:
        return {"ok": False, "error": str(e), "raw_preview": outline_yaml[:500]}
    return {
        "ok": True,
        "path": str(path),
        "stats": {
            "equations": len(outline.key_equations),
            "concepts": len(outline.key_concepts),
            "critical_ids": outline.critical_ids(),
            "important_ids": outline.important_ids(),
        },
    }


@mcp.tool()
def get_outline(arxiv_id: str) -> dict:
    """Load the canonical outline YAML previously saved."""
    body = storage.load_outline_yaml(arxiv_id)
    return {"ok": body is not None, "outline_yaml": body or ""}


@mcp.tool()
def save_script(arxiv_id: str, script: str) -> dict:
    """Persist the script from stage 2."""
    path = storage.save_script(arxiv_id, script)
    return {"ok": True, "path": str(path), "word_count": len(script.split())}


@mcp.tool()
def get_script(arxiv_id: str) -> dict:
    """Load the script previously saved."""
    body = storage.load_script(arxiv_id)
    return {"ok": body is not None, "script": body or ""}


@mcp.tool()
def save_audit(arxiv_id: str, audit_yaml: str) -> dict:
    """Validate + persist the audit report from stage 3.

    Returns the decision so the host can branch without re-parsing.
    """
    try:
        path, audit = storage.save_audit(arxiv_id, audit_yaml)
    except ParseError as e:
        return {"ok": False, "error": str(e), "raw_preview": audit_yaml[:500]}
    return {
        "ok": True,
        "path": str(path),
        "coverage_status": audit.coverage_status,
        "recommendation": audit.recommendation,
        "items_missing": [m.model_dump() for m in audit.items_missing],
        "items_glossed": [g.model_dump() for g in audit.items_glossed],
    }


@mcp.tool()
def list_seen() -> list[dict]:
    """Papers already delivered."""
    return storage.list_seen()


@mcp.tool()
def mark_seen(arxiv_id: str, title: str = "", note: str = "") -> dict:
    """Record that a paper has been delivered."""
    storage.mark_seen(arxiv_id, title=title, note=note)
    return {"ok": True, "arxiv_id": arxiv_id}


# ---- tools: audio --------------------------------------------------------


@mcp.tool()
def render_audio(
    arxiv_id: str,
    mode: str = "single_host",
    output_format: str = "mp3",
    backend: str | None = None,
) -> dict:
    """Render the saved script for `arxiv_id` to audio.

    mode: "single_host" (denser, math-friendly) or "two_host" (parses
          <Person1>/<Person2> tags, stitches with a short pause).
    output_format: "mp3" (needs ffmpeg) or "wav".
    backend: "kokoro" (local, default) or "vertex" (Google Vertex AI TTS).
             Falls back to PAPERTEACHER_TTS env var when omitted.
    """
    script = storage.load_script(arxiv_id)
    if script is None:
        return {"ok": False, "error": f"no script saved for {arxiv_id}"}
    out = audio.render(
        script=script,
        mode=mode,
        output_format=output_format,
        backend=tts.get_backend(backend) if backend else None,
        filename=f"paper_{arxiv_id}.{output_format}",
    )
    return {"ok": True, "audio_path": str(out)}


# ---- prompts: the three pipeline stages ----------------------------------


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
    outline_yaml = storage.load_outline_yaml(arxiv_id)
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
    YAML report whose `recommendation` is one of:
    ship | regenerate_with_gaps | regenerate_from_scratch.
    """
    outline_yaml = storage.load_outline_yaml(arxiv_id)
    script = storage.load_script(arxiv_id)
    if not outline_yaml or not script:
        missing = []
        if not outline_yaml:
            missing.append("outline")
        if not script:
            missing.append("script")
        return f"ERROR: missing {', '.join(missing)} for {arxiv_id}. Run earlier stages first."
    return prompts.render_audit(outline_yaml=outline_yaml, script=script)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=__import__("sys").stderr,  # MCP uses stdout for protocol
    )
    paths.ensure_layout()
    mcp.run()


if __name__ == "__main__":
    main()
