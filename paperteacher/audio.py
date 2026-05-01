"""Script → audio orchestration. Backend-agnostic.

Parses two-host tags, stitches turns with a short pause, encodes to wav/mp3,
writes atomically. The actual TTS lives in `paperteacher.tts`.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import paths, tts

log = logging.getLogger(__name__)

_SEG_RE = re.compile(r"<(Person[12])>(.*?)</\1>", re.DOTALL)


def _parse_two_host(script: str) -> list[tuple[str, str]]:
    return [
        (m.group(1), m.group(2).strip())
        for m in _SEG_RE.finditer(script)
        if m.group(2).strip()
    ]


def _write_mp3(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    """Float32 [-1, 1] → mp3 at speech-friendly bitrate. Needs ffmpeg via pydub."""
    from pydub import AudioSegment

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    AudioSegment(
        pcm.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    ).export(path, format="mp3", bitrate=paths.MP3_BITRATE)


def render(
    script: str,
    *,
    out_dir: Path | str | None = None,
    mode: str = "single_host",
    voices: dict | None = None,
    backend: tts.TTSBackend | None = None,
    output_format: str = "mp3",
    filename: str | None = None,
) -> Path:
    """Render `script` to an audio file and return its path.

    mode:
      - "single_host": one voice. Strips any speaker tags. Best for math-heavy
                       papers — denser, no banter overhead.
      - "two_host":   parses <Person1>/<Person2> tags, renders each turn with
                       the matching voice, stitches with a short pause.

    backend: TTS backend instance. Defaults to `tts.get_backend()` which honors
             PAPERTEACHER_TTS (kokoro | vertex).

    output_format: "mp3" (compact, requires ffmpeg) or "wav".
    """
    paths.ensure_layout()
    out = Path(out_dir or paths.AUDIO_DIR)
    out.mkdir(parents=True, exist_ok=True)
    backend = backend or tts.get_backend()
    voice_map = {**tts.voices_for(backend), **(voices or {})}
    sample_rate = backend.sample_rate_hz

    if mode == "single_host":
        text = re.sub(r"</?Person[12]>", "", script).strip()
        if not text:
            raise ValueError("single_host mode received an empty script")
        audio_data = backend.synth(text, voice_map["Person1"])
    elif mode == "two_host":
        segments = _parse_two_host(script)
        if not segments:
            raise ValueError(
                "two_host mode requires <Person1>...</Person1><Person2>...</Person2> tags"
            )
        gap = np.zeros(int(sample_rate * paths.INTER_TURN_PAUSE_S), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for speaker, text in segments:
            chunks.append(backend.synth(text, voice_map[speaker]))
            chunks.append(gap)
        audio_data = np.concatenate(chunks)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected single_host or two_host)")

    name = filename or f"paper_{dt.date.today().isoformat()}.{output_format}"
    final = out / name
    if output_format == "wav":
        sf.write(final, audio_data, sample_rate)
    elif output_format == "mp3":
        # write to a temp file first, then atomic-rename, so partial files
        # don't masquerade as complete renders.
        with tempfile.NamedTemporaryFile(
            dir=out, suffix=".mp3.partial", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _write_mp3(audio_data, tmp_path, sample_rate)
            tmp_path.replace(final)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    else:
        raise ValueError(f"Unknown output_format: {output_format!r}")

    log.info(
        "rendered %s (backend=%s, mode=%s, %d samples, %.1fs)",
        final, backend.name, mode, len(audio_data), len(audio_data) / sample_rate,
    )
    return final
