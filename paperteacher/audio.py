"""Render a Claude-written script to audio via podcastfy.

We bypass podcastfy's own LLM script writer. Claude writes the script in
its native voice (single host or two host), and we hand it to podcastfy as
a transcript so podcastfy only does TTS + stitching.
"""
from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path


def render(
    script: str,
    *,
    out_dir: Path | str | None = None,
    mode: str = "single_host",
    tts_model: str = "elevenlabs",
    config_path: Path | str | None = None,
) -> Path:
    """Render `script` to an mp3 and return its path.

    Two host modes the script writer should produce:
      - "single_host": plain narration, one voice. Best for technical depth.
      - "two_host":   "<Person1>...</Person1><Person2>...</Person2>" XML-ish tags
                       in the transcript, which podcastfy expects for multi-speaker.
    """
    try:
        from podcastfy.client import generate_podcast
    except ImportError as e:
        raise RuntimeError(
            "podcastfy not installed. `pip install podcastfy` (and set ELEVENLABS_API_KEY)."
        ) from e

    out = Path(out_dir or Path.home() / ".paperteacher" / "out")
    out.mkdir(parents=True, exist_ok=True)

    transcript_path = Path(tempfile.mkstemp(suffix=".txt")[1])
    transcript_path.write_text(_format_transcript(script, mode))

    kwargs: dict = {
        "transcript_file": str(transcript_path),
        "tts_model": tts_model,
    }
    if config_path:
        kwargs["conversation_config"] = str(config_path)

    audio_path = generate_podcast(**kwargs)
    final = out / f"paper_{dt.date.today().isoformat()}.mp3"
    Path(audio_path).rename(final)
    return final


def _format_transcript(script: str, mode: str) -> str:
    """Podcastfy expects single-speaker text or <Person1>/<Person2> tags."""
    if mode == "single_host":
        if "<Person1>" in script:
            return script
        return f"<Person1>{script}</Person1>"
    if mode == "two_host":
        if "<Person1>" not in script:
            raise ValueError(
                "two_host mode requires <Person1>...</Person1><Person2>...</Person2> tags"
            )
        return script
    raise ValueError(f"Unknown mode: {mode}")
