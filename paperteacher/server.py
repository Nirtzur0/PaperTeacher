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

from . import audio, paths, preferred as preferred_mod, profile, storage, tts
from .domain import ParseError, discover_all, domain_for
from .domain import read_paper as _read_routed

log = logging.getLogger(__name__)
mcp = FastMCP("paperteacher")


# ---- resources ------------------------------------------------------------


@mcp.resource("profile://taste")
def taste_profile() -> str:
    """The listener's taste profile (fields, known topics, voice preferences).

    Hosts should fetch this once at the start of a pipeline run rather than
    expecting it to be re-inlined into every stage's prompt — plan and teach
    no longer carry the profile body in their prompts (saves ~1.5K tokens
    per pipeline run; see paperteacher.domains._prompts).
    """
    return profile.read_text_for_prompt()


@mcp.resource("outline://{arxiv_id}")
def outline_resource(arxiv_id: str) -> str:
    """Saved outline for a paper, or a hint if it hasn't been extracted."""
    body = storage.load_outline_yaml(arxiv_id)
    return body or f"(no outline saved for {arxiv_id} — run the extract_outline prompt first)"


@mcp.resource("voice-guide://{domain}")
def voice_guide_resource(domain: str) -> str:
    """Per-domain voice guide — pronunciation tables, numerical rewrites,
    banned phrases, anti-anthropomorphism rules. The teach prompt references
    this resource by URI rather than re-shipping its ~1K tokens on every
    (re)generation; hosts should fetch the relevant pack's guide once.

    Currently shipped for `ml`; physics/neuro/econ inline their (smaller)
    voice rules in the teach prompt body and return an empty body here.
    """
    from .domain import get_domain

    try:
        pack = get_domain(domain)
    except ValueError:
        return f"(unknown domain {domain!r})"
    guide = getattr(pack, "voice_guide_text", "") or ""
    if not guide:
        return (
            f"(domain {domain!r} inlines its voice rules in the teach prompt — "
            f"no separate guide to fetch)"
        )
    return guide


# ---- tools: discovery + reading ------------------------------------------


@mcp.tool()
async def fetch_trending_papers(
    arxiv_categories: list[str] | None = None,
    limit: int = paths.DEFAULT_DISCOVERY_LIMIT,
) -> list[dict]:
    """Discover candidate papers across all active domain packs in parallel.
    For ML: HF Daily + arXiv RSS. Other configured packs route through their
    own fetchers. Each candidate carries `domain` so the host can route
    follow-up calls back to the originating pack. Filters out seen and
    skipped IDs server-side.
    """
    cands = await discover_all(arxiv_categories=arxiv_categories or [], limit=limit)
    excluded = storage.excluded_ids()
    cands = [c for c in cands if c.arxiv_id not in excluded]
    pref = preferred_mod.load()
    if pref is not None:
        preferred_mod.apply(cands, pref)
    return [c.to_dict() for c in cands]


@mcp.tool()
async def read_paper(
    arxiv_id: str,
    max_chars: int = paths.DEFAULT_MAX_PAPER_CHARS,
    domain: str | None = None,
) -> dict:
    """Fetch full paper text. `domain` is the originating pack (from a
    Candidate's `domain` field); if omitted, routing falls back to the
    paper's recorded pack or the first active one.
    """
    p = await _read_routed(arxiv_id, max_chars=max_chars, hint=domain)
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
    # Domain-specific stat fields. Each domain's outline can expose any
    # combination of `critical_ids`/`important_ids`/`key_equations`/etc.;
    # we surface whatever is present and skip the rest.
    stats: dict = {}
    for fn in ("critical_ids", "important_ids"):
        if callable(getattr(outline, fn, None)):
            stats[fn] = getattr(outline, fn)()
    for attr in ("key_equations", "key_concepts"):
        if isinstance(getattr(outline, attr, None), list):
            stats[attr.replace("key_", "") + "_count"] = len(getattr(outline, attr))
    return {"ok": True, "path": str(path), "stats": stats}


@mcp.tool()
def get_outline(arxiv_id: str) -> dict:
    """Load the canonical outline YAML previously saved."""
    body = storage.load_outline_yaml(arxiv_id)
    return {"ok": body is not None, "outline_yaml": body or ""}


@mcp.tool()
def save_plan(arxiv_id: str, plan_yaml: str) -> dict:
    """Validate + persist the YAML episode plan from stage 1.5.

    The planner stage is OPT-IN per domain pack — returns
    {ok: false, error: "no planner stage"} for packs that haven't implemented
    `parse_plan`. When a plan is saved, the teach prompt drops its default
    arc and follows the plan's structure instead.
    """
    try:
        path, plan = storage.save_plan(arxiv_id, plan_yaml)
    except ParseError as e:
        return {"ok": False, "error": str(e), "raw_preview": plan_yaml[:500]}
    except NotImplementedError as e:
        return {"ok": False, "error": str(e)}
    stats: dict = {}
    arc = getattr(plan, "arc", None)
    if isinstance(arc, list):
        stats["segments"] = len(arc)
        stats["roles"] = [getattr(s, "role", None) for s in arc]
    takes = getattr(plan, "takes", None)
    if isinstance(takes, list):
        stats["takes_count"] = len(takes)
    return {"ok": True, "path": str(path), "stats": stats}


@mcp.tool()
def get_plan(arxiv_id: str) -> dict:
    """Load the canonical episode plan YAML previously saved (if any)."""
    body = storage.load_plan_yaml(arxiv_id)
    return {"ok": body is not None, "plan_yaml": body or ""}


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

    Returns just the decision and counts so the host can branch without
    re-parsing — the full per-item breakdown lives in the saved YAML
    (`audits/{arxiv_id}.yaml`) and can be loaded if needed. Slimming the
    response saves ~1.5K tokens per audit when the LLM gets it back as
    a tool result.
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
        "missing_count": len(audit.items_missing),
        "glossed_count": len(audit.items_glossed),
    }


@mcp.tool()
def list_seen() -> list[dict]:
    """Papers already delivered. Each row carries `tags` (topic tags) so the
    host can compute topic distribution for balanced selection.
    """
    return storage.list_seen()


@mcp.tool()
def mark_seen(
    arxiv_id: str,
    title: str = "",
    note: str = "",
    tags: list[str] | None = None,
) -> dict:
    """Record that a paper has been delivered. `tags` is a list of free-form
    topic tags (e.g. ["info-geometry", "optimization"]) used to balance
    coverage across episodes.
    """
    storage.mark_seen(arxiv_id, title=title, note=note, tags=tags)
    return {"ok": True, "arxiv_id": arxiv_id}


@mcp.tool()
def list_skipped() -> list[dict]:
    """Candidates considered but NOT delivered. The backlog. Use this to
    revisit interesting papers that lost the daily slot.
    """
    return storage.list_skipped()


@mcp.tool()
def mark_skipped(
    arxiv_id: str,
    title: str = "",
    tags: list[str] | None = None,
    reason: str = "",
) -> dict:
    """Record a candidate that was considered but skipped today. Adds it to
    the backlog. Server filters skipped IDs out of fetch_trending_papers by
    default, but you can revisit a skipped paper by calling read_paper /
    extract_outline directly with its ID.
    """
    storage.mark_skipped(arxiv_id, title=title, tags=tags, reason=reason)
    return {"ok": True, "arxiv_id": arxiv_id}


@mcp.tool()
def topic_distribution(window: int = 30) -> dict:
    """Tag → count over the last `window` delivered papers. Use when picking
    the next paper to favor underrepresented topics.
    """
    return storage.topic_distribution(window=window)


# ---- tools: audio --------------------------------------------------------
#
# Vertex Chirp 3 TTS on a ~2500-word two_host script can take 60–180 seconds
# end to end (multiple sync calls × the 5000-byte chunk limit + mp3 encode).
# OpenClaw's MCP tool timeout is well below that, which previously caused
# render_audio to surface as -32001 to the host, the agent to fall back to
# text-only delivery, and the file to land on disk seconds after delivery
# already shipped without the audio. Fix: render_audio fires the work in a
# background thread and returns IMMEDIATELY with the deterministic path; the
# host polls `audio_status(arxiv_id)` until ready before sending the voice
# note. The atomic .mp3.partial → final rename in audio.render() guarantees
# the host never sees a half-written file.

import threading

# In-flight render state, keyed by arxiv_id. Lives in process memory only;
# the canonical truth is whether the file exists on disk. This dict just
# records error reasons so audio_status can surface them.
_render_state: dict[str, dict] = {}
_render_lock = threading.Lock()


def _audio_path(arxiv_id: str, output_format: str) -> "Path":
    from pathlib import Path
    return Path(paths.AUDIO_DIR) / f"paper_{arxiv_id}.{output_format}"


def _render_in_background(
    arxiv_id: str,
    script: str,
    mode: str,
    output_format: str,
    backend_name: str | None,
) -> None:
    try:
        audio.render(
            script=script,
            mode=mode,
            output_format=output_format,
            backend=tts.get_backend(backend_name) if backend_name else None,
            filename=f"paper_{arxiv_id}.{output_format}",
        )
        with _render_lock:
            _render_state[arxiv_id] = {"status": "done"}
    except Exception as e:  # noqa: BLE001 — errors must reach audio_status
        log.exception("render_audio background task failed for %s", arxiv_id)
        with _render_lock:
            _render_state[arxiv_id] = {"status": "error", "error": str(e)}


@mcp.tool()
def render_audio(
    arxiv_id: str,
    mode: str = "single_host",
    output_format: str = "mp3",
    backend: str | None = None,
) -> dict:
    """Kick off audio rendering for the saved script. Returns IMMEDIATELY
    with the path the file will land at; the actual TTS runs in a background
    thread. Poll `audio_status(arxiv_id)` until `ready=true` before sending
    the voice note.

    mode: "single_host" (denser, math-friendly) or "two_host" (parses
          <Person1>/<Person2> tags, stitches with a short pause).
    output_format: "mp3" (needs ffmpeg) or "wav".
    backend: "kokoro" (local, default) or "vertex" (Google Vertex AI TTS).
             Falls back to PAPERTEACHER_TTS env var when omitted.
    """
    script = storage.load_script(arxiv_id)
    if script is None:
        return {"ok": False, "error": f"no script saved for {arxiv_id}"}
    target = _audio_path(arxiv_id, output_format)
    # If the file already exists, return ready immediately (re-runs are cheap).
    if target.exists():
        return {"ok": True, "audio_path": str(target), "status": "done"}
    with _render_lock:
        _render_state[arxiv_id] = {"status": "rendering"}
    threading.Thread(
        target=_render_in_background,
        args=(arxiv_id, script, mode, output_format, backend),
        daemon=True,
        name=f"render_audio:{arxiv_id}",
    ).start()
    return {
        "ok": True,
        "audio_path": str(target),
        "status": "rendering",
        "hint": "poll audio_status(arxiv_id) until ready=true (typically 60–180s for two_host on Vertex).",
    }


@mcp.tool()
def audio_status(arxiv_id: str, output_format: str = "mp3") -> dict:
    """Check whether the rendered audio file exists for `arxiv_id`. Pair with
    `render_audio` for a fire-and-poll workflow.

    Returns:
      {"ready": True,  "path": str, "size_bytes": int}      when the file is on disk
      {"ready": False, "status": "rendering"}               while a background render is running
      {"ready": False, "status": "error", "error": str}     when the background render failed
      {"ready": False, "status": "absent"}                  when nothing has been rendered yet
    """
    target = _audio_path(arxiv_id, output_format)
    if target.exists():
        return {
            "ready": True,
            "path": str(target),
            "size_bytes": target.stat().st_size,
        }
    with _render_lock:
        state = _render_state.get(arxiv_id, {})
    status = state.get("status", "absent")
    out: dict = {"ready": False, "status": status}
    if status == "error":
        out["error"] = state.get("error", "")
    return out


# ---- prompts: the pipeline stages ----------------------------------------


@mcp.prompt()
async def extract_outline(arxiv_id: str) -> str:
    """STAGE 1. Read the paper and produce a structured YAML outline. The
    outline shape is determined by the paper's domain pack — for ML this is
    the equation+concept schema; other domains use their own contracts.
    """
    p = await _read_routed(arxiv_id)
    domain = domain_for(arxiv_id)
    return domain.render_extract(
        arxiv_id=arxiv_id,
        title=p.title,
        taste_profile=profile.read_text_for_prompt(),
        paper_text=p.text,
    )


@mcp.prompt()
async def plan_episode(arxiv_id: str) -> str:
    """STAGE 1.5 (optional). Design the macro arc + persona stance for the
    episode. Requires a saved outline. When the resulting plan is saved via
    `save_plan`, the teach prompt drops its default arc and follows the plan
    instead — different papers get different shapes.

    Domain packs that haven't opted into the planner stage return an error
    pointing the host at `teach_from_outline` directly.
    """
    outline_yaml = storage.load_outline_yaml(arxiv_id)
    if not outline_yaml:
        return (
            f"ERROR: no outline saved for {arxiv_id}. "
            f"Run `extract_outline` + `save_outline` first."
        )
    domain = domain_for(arxiv_id)
    render = getattr(domain, "render_plan", None)
    if render is None:
        return (
            f"ERROR: domain {domain.name!r} does not implement the planner stage. "
            f"Skip this prompt and run `teach_from_outline` directly."
        )
    p = await _read_routed(arxiv_id)
    # Plan body drops paper_text (outline carries the structural claims; saves
    # ~30K tokens) but keeps taste_profile — the listener voice anchors which
    # adjacent works the takes draw from and what depth to land at.
    return render(
        arxiv_id=arxiv_id,
        title=p.title,
        taste_profile=profile.read_text_for_prompt(),
        outline_yaml=outline_yaml,
    )


@mcp.prompt()
async def teach_from_outline(arxiv_id: str, mode: str = "single_host") -> str:
    """STAGE 2. Write the spoken script using the previously-saved outline as
    a mandatory coverage contract. Run extract_outline + save_outline first.
    If a plan has also been saved (stage 1.5), it is loaded automatically and
    drives the macro structure instead of the pack's default arc.
    """
    outline_yaml = storage.load_outline_yaml(arxiv_id)
    if not outline_yaml:
        return (
            f"ERROR: no outline saved for {arxiv_id}. "
            f"Run the `extract_outline` prompt for this paper, then call "
            f"`save_outline(arxiv_id={arxiv_id!r}, outline_yaml=...)` with the result, "
            f"then re-invoke this prompt."
        )
    plan_yaml = storage.load_plan_yaml(arxiv_id)  # None when no plan saved
    p = await _read_routed(arxiv_id)
    domain = domain_for(arxiv_id)
    # taste_profile is the voice anchor — keeps the script as a working-
    # researcher deep dive instead of a generic abstract paraphrase. Earlier
    # diet had pulled it out and the script collapsed; restored.
    return domain.render_teach(
        arxiv_id=arxiv_id,
        title=p.title,
        taste_profile=profile.read_text_for_prompt(),
        paper_text=p.text,
        outline_yaml=outline_yaml,
        mode=mode,
        plan_yaml=plan_yaml,
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
    return domain_for(arxiv_id).render_audit(outline_yaml=outline_yaml, script=script)


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
