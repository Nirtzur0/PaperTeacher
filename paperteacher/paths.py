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
PLANS_DIR = ROOT / "plans"
SCRIPTS_DIR = ROOT / "scripts"
AUDITS_DIR = ROOT / "audits"
AUDIO_DIR = ROOT / "audio"
# Per-paper metadata sidecars (e.g. which domain pack produced/read it).
# One tiny JSON per arxiv_id; the routing layer is the only writer/reader.
META_DIR = ROOT / "meta"
SEEN_FILE = ROOT / "seen.jsonl"
SKIPPED_FILE = ROOT / "skipped.jsonl"  # candidates considered but not delivered
EVENT_LOG = ROOT / "pipeline.jsonl"

# Profile lives under PAPERTEACHER_HOME by default — NOT in the cwd, so the
# MCP server finds it regardless of where the host launches it. The repo
# ships `config/profile.example.md` as a starting template.
PROFILE_PATH = Path(os.environ.get("PAPERTEACHER_PROFILE", ROOT / "profile.md"))

# Preferred-authors allowlist. Off by default — when this file is absent,
# discovery scoring is unchanged. Copy `config/preferred.example.yaml` to
# this path to activate. See paperteacher.preferred for the schema.
PREFERRED_PATH = Path(os.environ.get("PAPERTEACHER_PREFERRED", ROOT / "preferred.yaml"))


def ensure_layout() -> None:
    """Create all subdirectories. Cheap to call repeatedly."""
    for d in (ROOT, OUTLINES_DIR, PLANS_DIR, SCRIPTS_DIR, AUDITS_DIR, AUDIO_DIR, META_DIR):
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
# Framework mechanism, not user preference. User-facing knobs (script length,
# mode, speaking rate, discovery categories) live in `paperteacher.profile` —
# see config/profile.example.md. Don't add user-facing knobs here.

DEFAULT_MAX_PAPER_CHARS = 120_000
DEFAULT_DISCOVERY_LIMIT = 20
