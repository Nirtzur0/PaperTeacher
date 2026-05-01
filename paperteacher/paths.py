"""Single source of truth for filesystem layout and tunable constants.

Override the root via PAPERTEACHER_HOME env var.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- versioning -----------------------------------------------------------

USER_AGENT = "PaperTeacher/0.2 (+https://github.com/Nirtzur0/PaperTeacher)"

# ---- filesystem layout ----------------------------------------------------

ROOT = Path(os.environ.get("PAPERTEACHER_HOME", Path.home() / ".paperteacher"))

OUTLINES_DIR = ROOT / "outlines"
SCRIPTS_DIR = ROOT / "scripts"
AUDITS_DIR = ROOT / "audits"
AUDIO_DIR = ROOT / "audio"
SEEN_FILE = ROOT / "seen.jsonl"
SKIPPED_FILE = ROOT / "skipped.jsonl"  # candidates considered but not delivered
EVENT_LOG = ROOT / "pipeline.jsonl"

# Profile lives under PAPERTEACHER_HOME by default — NOT in the cwd, so the
# MCP server finds it regardless of where the host launches it. The repo
# ships `config/profile.example.md` as a starting template.
PROFILE_PATH = Path(os.environ.get("PAPERTEACHER_PROFILE", ROOT / "profile.md"))


def ensure_layout() -> None:
    """Create all subdirectories. Cheap to call repeatedly."""
    for d in (ROOT, OUTLINES_DIR, SCRIPTS_DIR, AUDITS_DIR, AUDIO_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---- audio constants ------------------------------------------------------

SAMPLE_RATE_HZ = 24_000
INTER_TURN_PAUSE_S = 0.35
MP3_BITRATE = "64k"
DEFAULT_LANG = "a"  # 'a' = American English, 'b' = British. Kokoro convention.

# ---- TTS backend ----------------------------------------------------------

# Backend selection. Override with PAPERTEACHER_TTS=kokoro|vertex.
# Kokoro = local, free, no keys (default).
# Vertex = Google Vertex AI Text-to-Speech (Chirp 3 voices). Needs ADC.
DEFAULT_TTS_BACKEND = os.environ.get("PAPERTEACHER_TTS", "kokoro")

# ---- pipeline constants ---------------------------------------------------

DEFAULT_MAX_PAPER_CHARS = 120_000
DEFAULT_DISCOVERY_LIMIT = 20
# ~10 min target at Vertex Chirp 3 HD with speaking_rate 1.1 (~165 wpm × 1.1 ≈ 180 wpm).
TARGET_SCRIPT_WORDS = 1750
# Slightly faster than default — keeps the listener leaning forward without rushing.
DEFAULT_TTS_SPEAKING_RATE = 1.1
