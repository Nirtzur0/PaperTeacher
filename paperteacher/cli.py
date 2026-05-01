"""CLI for running pipeline stages without an MCP host.

Designed for: testing prompts on a real paper, debugging a glossed episode,
manual end-to-end runs. The LLM stages print prompts to stdout — pipe them
into Claude/Gemini/whatever, then feed the result back via `save-outline` etc.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer

from . import audio, discovery, paths, prompts, reader, storage, tts

app = typer.Typer(
    add_completion=False,
    help="PaperTeacher: 3-stage paper-to-podcast pipeline.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _profile() -> str:
    if paths.PROFILE_PATH.exists():
        return paths.PROFILE_PATH.read_text()
    return "(no profile.md found — generic ML defaults)"


# ---- discover -------------------------------------------------------------


@app.command()
def discover(
    categories: str = typer.Option("cs.LG,stat.ML", help="comma-separated arXiv categories"),
    limit: int = typer.Option(paths.DEFAULT_DISCOVERY_LIMIT),
    json_output: bool = typer.Option(False, "--json", help="emit JSON instead of a table"),
) -> None:
    """List candidate papers (HF Daily + arXiv RSS), filtering already-seen."""
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    cands = asyncio.run(discovery.discover(arxiv_categories=cats, limit=limit))
    seen = storage.seen_ids()
    cands = [c for c in cands if c.arxiv_id not in seen]
    if json_output:
        typer.echo(json.dumps([c.to_dict() for c in cands], indent=2))
        return
    for c in cands:
        typer.echo(f"{c.arxiv_id}  [{c.source:>14}]  score={c.score:>5.0f}  {c.title}")


# ---- read -----------------------------------------------------------------


@app.command()
def read(arxiv_id: str, max_chars: int = paths.DEFAULT_MAX_PAPER_CHARS) -> None:
    """Print the paper's full text (or abstract if HTML unavailable)."""
    p = asyncio.run(reader.read_paper(arxiv_id, max_chars=max_chars))
    typer.echo(f"# source: {p.source}  truncated: {p.truncated}\n# title: {p.title}\n")
    typer.echo(p.text)


# ---- prompts (stage 1, 2, 3) ---------------------------------------------

prompt = typer.Typer(help="Render a stage's prompt to stdout.")
app.add_typer(prompt, name="prompt")


@prompt.command("extract")
def prompt_extract(arxiv_id: str) -> None:
    """STAGE 1 prompt — feed paper to your LLM, expect YAML outline back."""
    paper = asyncio.run(reader.read_paper(arxiv_id))
    typer.echo(prompts.render_extract(
        arxiv_id=arxiv_id,
        title=paper.title,
        taste_profile=_profile(),
        paper_text=paper.text,
    ))


@prompt.command("teach")
def prompt_teach(
    arxiv_id: str,
    mode: str = typer.Option("single_host", help="single_host or two_host"),
) -> None:
    """STAGE 2 prompt — requires a saved outline."""
    outline_yaml = storage.load_outline_yaml(arxiv_id)
    if outline_yaml is None:
        typer.echo(f"error: no outline for {arxiv_id} — run extract + save-outline first", err=True)
        raise typer.Exit(1)
    paper = asyncio.run(reader.read_paper(arxiv_id))
    typer.echo(prompts.render_teach(
        arxiv_id=arxiv_id,
        title=paper.title,
        taste_profile=_profile(),
        paper_text=paper.text,
        outline_yaml=outline_yaml,
        mode=mode,
    ))


@prompt.command("audit")
def prompt_audit(arxiv_id: str) -> None:
    """STAGE 3 prompt — requires saved outline AND saved script."""
    outline_yaml = storage.load_outline_yaml(arxiv_id)
    script = storage.load_script(arxiv_id)
    if outline_yaml is None or script is None:
        missing = [n for n, v in [("outline", outline_yaml), ("script", script)] if v is None]
        typer.echo(f"error: missing {', '.join(missing)} for {arxiv_id}", err=True)
        raise typer.Exit(1)
    typer.echo(prompts.render_audit(outline_yaml=outline_yaml, script=script))


# ---- save (stages 1, 2, 3 outputs) ---------------------------------------


def _read_input(file: Path | None) -> str:
    if file is None:
        return sys.stdin.read()
    return file.read_text()


@app.command("save-outline")
def save_outline(
    arxiv_id: str,
    file: Path | None = typer.Option(None, "-f", "--file", help="read YAML from file (else stdin)"),
) -> None:
    """Validate + save the LLM's outline output."""
    raw = _read_input(file)
    p, outline = storage.save_outline(arxiv_id, raw)
    typer.echo(
        f"saved {p}  ({len(outline.key_equations)} equations, "
        f"{len(outline.critical_ids())} critical)"
    )


@app.command("save-script")
def save_script(
    arxiv_id: str,
    file: Path | None = typer.Option(None, "-f", "--file"),
) -> None:
    """Save the LLM's script output."""
    p = storage.save_script(arxiv_id, _read_input(file))
    typer.echo(f"saved {p}")


@app.command("save-audit")
def save_audit(
    arxiv_id: str,
    file: Path | None = typer.Option(None, "-f", "--file"),
) -> None:
    """Validate + save the LLM's audit YAML output."""
    raw = _read_input(file)
    p, audit = storage.save_audit(arxiv_id, raw)
    typer.echo(f"saved {p}  recommendation={audit.recommendation}  status={audit.coverage_status}")


# ---- render audio --------------------------------------------------------


@app.command()
def render(
    arxiv_id: str,
    mode: str = typer.Option("single_host"),
    output_format: str = typer.Option("mp3"),
    backend: str = typer.Option(
        "", help="kokoro | vertex (default: $PAPERTEACHER_TTS or kokoro)"
    ),
) -> None:
    """Render the saved script for `arxiv_id` to audio."""
    script = storage.load_script(arxiv_id)
    if script is None:
        typer.echo(f"error: no script for {arxiv_id}", err=True)
        raise typer.Exit(1)
    out = audio.render(
        script=script,
        mode=mode,
        output_format=output_format,
        backend=tts.get_backend(backend or None),
        filename=f"paper_{arxiv_id}.{output_format}",
    )
    typer.echo(str(out))


# ---- seen ----------------------------------------------------------------

seen = typer.Typer(help="Manage the seen-papers list.")
app.add_typer(seen, name="seen")


@seen.command("list")
def seen_list() -> None:
    for row in storage.list_seen():
        typer.echo(f"{row['arxiv_id']}  {row.get('ts','')}  {row.get('title','')}")


@seen.command("mark")
def seen_mark(
    arxiv_id: str,
    title: str = typer.Option(""),
    note: str = typer.Option(""),
) -> None:
    storage.mark_seen(arxiv_id, title=title, note=note)
    typer.echo(f"marked {arxiv_id} as seen")


# ---- entry point ---------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
