"""Single source of truth for filesystem layout and tunable constants.

Override the root via PAPERTEACHER_HOME env var.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- filesystem layout ----------------------------------------------------

ROOT = Path(os.environ.get("PAPERTEACHER_HOME", Path.home() / ".paperteacher"))

OUTLINES_DIR = ROOT / "outlines"
SCRIPTS_DIR = ROOT / "scripts"
AUDITS_DIR = ROOT / "audits"
AUDIO_DIR = ROOT / "audio"
SEEN_FILE = ROOT / "seen.jsonl"
EVENT_LOG = ROOT / "pipeline.jsonl"

PROFILE_PATH = Path(os.environ.get("PAPERTEACHER_PROFILE", "config/profile.md"))


def ensure_layout() -> None:
    """Create all subdirectories. Cheap to call repeatedly."""
    for d in (ROOT, OUTLINES_DIR, SCRIPTS_DIR, AUDITS_DIR, AUDIO_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---- audio constants ------------------------------------------------------

SAMPLE_RATE_HZ = 24_000
INTER_TURN_PAUSE_S = 0.35
MP3_BITRATE = "64k"
DEFAULT_LANG = "a"  # 'a' = American English, 'b' = British. Kokoro convention.

# ---- pipeline constants ---------------------------------------------------

DEFAULT_MAX_PAPER_CHARS = 120_000
DEFAULT_DISCOVERY_LIMIT = 20
TARGET_SCRIPT_WORDS = 2500
