"""Local TTS via Kokoro-82M. Single-host or stitched two-host."""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import soundfile as sf

# American English voices that ship with Kokoro-82M.
# Override per-call via the `voices` arg if you want British, etc.
DEFAULT_VOICES = {
    "Person1": "af_bella",    # mentor
    "Person2": "bm_george",   # interlocutor
}
SAMPLE_RATE = 24_000
INTER_TURN_PAUSE_S = 0.35

_pipeline = None


def _pipeline_for(lang_code: str = "a"):
    """Lazy-load Kokoro. lang_code 'a' = American English, 'b' = British."""
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline  # imported lazily so the MCP boots fast
        _pipeline = KPipeline(lang_code=lang_code)
    return _pipeline


def _synth(text: str, voice: str) -> np.ndarray:
    pipeline = _pipeline_for()
    chunks: list[np.ndarray] = []
    for _, _, audio in pipeline(text, voice=voice, speed=1):
        chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


SEG_RE = re.compile(r"<(Person[12])>(.*?)</\1>", re.DOTALL)


def _parse_two_host(script: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    for m in SEG_RE.finditer(script):
        text = m.group(2).strip()
        if text:
            segments.append((m.group(1), text))
    return segments


def _write_mp3(audio: np.ndarray, path: Path, bitrate: str = "64k") -> None:
    """Write float32 audio in [-1, 1] as mp3 at speech-friendly bitrate."""
    from pydub import AudioSegment

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    AudioSegment(
        pcm.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1
    ).export(path, format="mp3", bitrate=bitrate)


def render(
    script: str,
    *,
    out_dir: Path | str | None = None,
    mode: str = "single_host",
    voices: dict | None = None,
    output_format: str = "mp3",
) -> Path:
    """Render `script` to an audio file and return its path.

    mode:
      - "single_host": one voice. Best for math-heavy papers. Strips any tags.
      - "two_host":   parses <Person1>/<Person2> tags, renders each turn with
                      the matching voice, stitches with a short pause.
    output_format: "mp3" (compact, needs ffmpeg via pydub) or "wav".
    """
    out = Path(out_dir or Path.home() / ".paperteacher" / "out")
    out.mkdir(parents=True, exist_ok=True)
    voice_map = {**DEFAULT_VOICES, **(voices or {})}

    if mode == "single_host":
        text = re.sub(r"</?Person[12]>", "", script).strip()
        audio = _synth(text, voice_map["Person1"])
    elif mode == "two_host":
        segments = _parse_two_host(script)
        if not segments:
            raise ValueError(
                "two_host mode requires <Person1>...</Person1><Person2>...</Person2> tags"
            )
        gap = np.zeros(int(SAMPLE_RATE * INTER_TURN_PAUSE_S), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for speaker, text in segments:
            chunks.append(_synth(text, voice_map[speaker]))
            chunks.append(gap)
        audio = np.concatenate(chunks)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    final = out / f"paper_{dt.date.today().isoformat()}.{output_format}"
    if output_format == "wav":
        sf.write(final, audio, SAMPLE_RATE)
    elif output_format == "mp3":
        _write_mp3(audio, final)
    else:
        raise ValueError(f"Unknown output_format: {output_format}")
    return final
