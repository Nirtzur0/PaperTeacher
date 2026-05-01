"""TTS backend abstraction.

Two backends:

  - kokoro: local Kokoro-82M. Free, no keys, runs on the user's machine.
            Heavy import (pulls torch); deferred until first synth.
  - vertex: Google Vertex AI Text-to-Speech. Chirp 3 HD voices. Needs
            Application Default Credentials (`gcloud auth application-default
            login`) and the `google-cloud-texttospeech` package.

Both expose the same surface: `synth(text, voice) -> float32 mono numpy array
at SAMPLE_RATE_HZ`. The orchestration in `audio.py` (two-host stitching, mp3
encoding, atomic write) is backend-agnostic.

Voice names are backend-specific. The `DEFAULT_VOICES` dict in `audio.py` maps
the Person1/Person2 abstraction onto a backend's voice IDs.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from . import paths

log = logging.getLogger(__name__)


# Voice maps, keyed by backend. Person1 = mentor, Person2 = interlocutor.
VOICES: dict[str, dict[str, str]] = {
    "kokoro": {
        "Person1": "af_bella",
        "Person2": "bm_george",
    },
    "vertex": {
        # Chirp 3 HD voices — most natural prosody Vertex offers as of 2026.
        "Person1": "en-US-Chirp3-HD-Aoede",
        "Person2": "en-US-Chirp3-HD-Charon",
    },
}


class TTSBackend(Protocol):
    name: str
    sample_rate_hz: int

    def synth(self, text: str, voice: str) -> np.ndarray:
        """Return float32 mono audio in [-1, 1] at `sample_rate_hz`."""
        ...


# ---- Kokoro ---------------------------------------------------------------


class KokoroBackend:
    name = "kokoro"
    sample_rate_hz = paths.SAMPLE_RATE_HZ

    def __init__(self, lang_code: str = paths.DEFAULT_LANG) -> None:
        self.lang_code = lang_code
        self._pipeline: object | None = None

    def _ensure_pipeline(self) -> object:
        if self._pipeline is None:
            try:
                from kokoro import KPipeline  # heavy: torch, transformers
            except ImportError as e:
                raise RuntimeError(
                    "Kokoro TTS backend requested but `kokoro` is not installed. "
                    "Install with: pip install 'paperteacher[tts-kokoro]'"
                ) from e
            self._pipeline = KPipeline(lang_code=self.lang_code)
        return self._pipeline

    def synth(self, text: str, voice: str) -> np.ndarray:
        pipeline = self._ensure_pipeline()
        chunks: list[np.ndarray] = []
        for _, _, audio_chunk in pipeline(text, voice=voice, speed=1):
            chunks.append(np.asarray(audio_chunk, dtype=np.float32))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)


# ---- Vertex AI ------------------------------------------------------------


class VertexBackend:
    """Google Vertex AI Text-to-Speech (Chirp 3 HD voices).

    Auth: Application Default Credentials. Run `gcloud auth application-default
    login` once, or set GOOGLE_APPLICATION_CREDENTIALS to a service-account
    JSON path. Project is inferred from ADC; override with GOOGLE_CLOUD_PROJECT.

    The synchronous API caps inputs at 5000 bytes; longer scripts are split on
    paragraph/sentence boundaries and the resulting audio is concatenated.
    """

    name = "vertex"
    # Vertex returns LINEAR16 PCM at the rate we ask for. Match Kokoro for
    # mixing-friendly stitching (so the inter-turn pause has the same rate).
    sample_rate_hz = paths.SAMPLE_RATE_HZ
    # Vertex sync TTS hard limit is 5000 bytes; leave headroom for safety.
    MAX_BYTES_PER_REQUEST = 4500

    def __init__(
        self,
        language_code: str = "en-US",
        speaking_rate: float = paths.DEFAULT_TTS_SPEAKING_RATE,
    ) -> None:
        self.language_code = language_code
        self.speaking_rate = speaking_rate
        self._client: object | None = None

    def _ensure_client(self) -> object:
        if self._client is None:
            try:
                from google.cloud import texttospeech
            except ImportError as e:
                raise RuntimeError(
                    "Vertex TTS backend requested but `google-cloud-texttospeech` "
                    "is not installed. Install with: "
                    "pip install 'paperteacher[tts-vertex]'"
                ) from e
            self._client = texttospeech.TextToSpeechClient()
            self._mod = texttospeech
        return self._client

    def _chunk(self, text: str) -> list[str]:
        """Split text into chunks under MAX_BYTES_PER_REQUEST, on paragraph
        first, then sentence boundaries if a paragraph is still too long.
        """
        max_bytes = self.MAX_BYTES_PER_REQUEST
        if len(text.encode("utf-8")) <= max_bytes:
            return [text]
        import re
        chunks: list[str] = []
        # paragraphs first
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para.encode("utf-8")) <= max_bytes:
                chunks.append(para)
                continue
            # split paragraph on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for s in sentences:
                candidate = (buf + " " + s).strip() if buf else s
                if len(candidate.encode("utf-8")) > max_bytes:
                    if buf:
                        chunks.append(buf)
                    if len(s.encode("utf-8")) > max_bytes:
                        # pathologically long sentence: hard-split by bytes
                        encoded = s.encode("utf-8")
                        for i in range(0, len(encoded), max_bytes):
                            chunks.append(encoded[i:i+max_bytes].decode("utf-8", "ignore"))
                        buf = ""
                    else:
                        buf = s
                else:
                    buf = candidate
            if buf:
                chunks.append(buf)
        return chunks

    def synth(self, text: str, voice: str) -> np.ndarray:
        chunks = self._chunk(text)
        if len(chunks) == 1:
            return self._synth_chunk(chunks[0], voice)
        # synth each chunk, concatenate (no inter-chunk gap; chunks are adjacent prose)
        parts = [self._synth_chunk(c, voice) for c in chunks]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _synth_chunk(self, text: str, voice: str) -> np.ndarray:
        client = self._ensure_client()
        tts = self._mod  # set in _ensure_client
        synthesis_input = tts.SynthesisInput(text=text)
        voice_params = tts.VoiceSelectionParams(
            language_code=self.language_code,
            name=voice,
        )
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate_hz,
            speaking_rate=self.speaking_rate,
        )
        resp = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )
        # `audio_content` is a WAV-wrapped LINEAR16 PCM blob. Strip the 44-byte
        # RIFF header (Vertex returns standard PCM WAV) and decode int16 → float32.
        pcm_bytes = resp.audio_content
        if pcm_bytes[:4] == b"RIFF":
            pcm_bytes = pcm_bytes[44:]
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        return (pcm.astype(np.float32) / 32768.0)


# ---- factory --------------------------------------------------------------


def get_backend(name: str | None = None) -> TTSBackend:
    """Construct a TTS backend by name.

    Defaults to `paths.DEFAULT_TTS_BACKEND` (env-overridable via PAPERTEACHER_TTS).
    """
    selected = (name or paths.DEFAULT_TTS_BACKEND or "kokoro").lower()
    if selected == "kokoro":
        return KokoroBackend()
    if selected == "vertex":
        return VertexBackend()
    raise ValueError(
        f"Unknown TTS backend: {selected!r} (expected 'kokoro' or 'vertex')"
    )


def voices_for(backend: TTSBackend) -> dict[str, str]:
    """Default Person1/Person2 voice map for the given backend."""
    return dict(VOICES[backend.name])
