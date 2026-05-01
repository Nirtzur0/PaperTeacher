"""Listener profile — the single source of truth for user-shaped preferences.

The profile lives at `PROFILE_PATH` as markdown. Most of it is free-form text
that the host LLM interprets via prompt-template injection (`{taste_profile}`).
A few fields are *structured*: extracted from the markdown by simple line
regex and consumed programmatically. Those structured fields are the answer
to "where does the X come from" — they exist in exactly one place.

Resolution order for any structured field:
  1. The matching env var (PAPERTEACHER_<FIELD>) when defined for that field
  2. The profile.md line
  3. The framework fallback in paths.py

The split — "what's structured vs. what's free-form" — is documented per
field below and mirrored in `config/profile.example.md` so users know which
keys actually do something vs. which are advisory text the LLM reads.

Structured fields (read by code):
  - domain / domains  (handled in paperteacher.domain)
  - length_target_minutes  (drives target_script_words)
  - script_mode             (single_host | two_host)
  - speaking_rate           (TTS speed multiplier)
  - arxiv_categories        (default discovery categories for ML)

Advisory fields (interpreted only by the host LLM through the prompt):
  - name, fields, known_well, voice, skip_unless_unusually_strong,
    selection_bias, discovery_sources_priority — all stay as raw markdown.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import cache
from typing import Literal

from . import paths

log = logging.getLogger(__name__)

ScriptMode = Literal["single_host", "two_host"]

# Vertex Chirp 3 HD speaks at roughly 165 wpm at speaking_rate=1.0.
# target_script_words ≈ minutes × 165 × speaking_rate. The constant lives
# here (not paths.py) because the conversion is profile logic, not a tunable.
_BASE_WPM = 165


@dataclass(frozen=True)
class Profile:
    """Resolved listener preferences. Use `profile.load()` to get the
    cached singleton; tests can call `profile.reset()` to re-resolve.
    """

    raw: str
    length_target_minutes: int
    script_mode: ScriptMode
    speaking_rate: float
    arxiv_categories: list[str]

    @property
    def target_script_words(self) -> int:
        """Words to target for stage 2. Derived from minutes × wpm × rate
        so changing length_target_minutes or speaking_rate stays consistent.
        """
        return int(self.length_target_minutes * _BASE_WPM * self.speaking_rate)


# ---- parsing -------------------------------------------------------------


def _read_profile_text() -> str:
    if not paths.PROFILE_PATH.exists():
        return ""
    return paths.PROFILE_PATH.read_text()


def read_text_for_prompt() -> str:
    """Profile markdown for `{taste_profile}` injection — never raises.
    Returns a descriptive fallback string when the file is absent so the
    LLM still gets a coherent prompt instead of an empty section.
    """
    text = _read_profile_text()
    return text or "(no profile.md found — using generic ML/CS defaults)"


def _line(text: str, key: str) -> str | None:
    """Extract a `key: value` line from the profile, value-only."""
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _csv_list(text: str, key: str) -> list[str] | None:
    """Extract `key: a, b, c` as a list."""
    val = _line(text, key)
    if val is None:
        return None
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return parts or None


def _int(text: str, key: str) -> int | None:
    val = _line(text, key)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        log.warning("profile.md: %s=%r is not an integer; ignoring", key, val)
        return None


def _float(text: str, key: str) -> float | None:
    val = _line(text, key)
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        log.warning("profile.md: %s=%r is not a float; ignoring", key, val)
        return None


# ---- resolution ----------------------------------------------------------


_FALLBACK_LENGTH_MIN = 10
_FALLBACK_MODE: ScriptMode = "two_host"
_FALLBACK_RATE = 1.1
# Categories chosen for ML/CS theory + interpretability + alignment coverage.
# math-ph (mathematical physics) used to be here by mistake — it's almost
# entirely off-topic for ML. The replacements are:
#   - cs.CL: NLP + most LLM/alignment/interp papers cross-list here
#   - cs.AI: RL, planning, agents that don't tag cs.LG
#   - math.ST: statistical theory (replaces the math-ph slot for theory work)
#   - math.OC: optimization, the math underpinning of training dynamics
_FALLBACK_CATEGORIES = ["cs.LG", "cs.CL", "cs.AI", "stat.ML", "math.ST", "math.OC"]


@cache
def load() -> Profile:
    """Resolve the profile once, cache for the process. Tests should call
    `reset()` to force re-resolution after monkeypatching env vars or the
    profile path.
    """
    raw = _read_profile_text()

    # length_target_minutes
    length = _int(raw, "length_target_minutes")
    if length is None:
        length = _FALLBACK_LENGTH_MIN

    # script_mode (single_host | two_host)
    mode_raw = os.environ.get("PAPERTEACHER_SCRIPT_MODE") or _line(raw, "script_mode")
    if mode_raw not in (None, "single_host", "two_host"):
        log.warning("profile/env script_mode=%r invalid; using %s", mode_raw, _FALLBACK_MODE)
        mode_raw = None
    mode: ScriptMode = mode_raw or _FALLBACK_MODE  # type: ignore[assignment]

    # speaking_rate
    rate_env = os.environ.get("PAPERTEACHER_SPEAKING_RATE")
    rate = float(rate_env) if rate_env else _float(raw, "speaking_rate")
    if rate is None:
        rate = _FALLBACK_RATE

    # arxiv_categories
    cats = _csv_list(raw, "arxiv_categories")
    if cats is None:
        cats = list(_FALLBACK_CATEGORIES)

    return Profile(
        raw=raw,
        length_target_minutes=length,
        script_mode=mode,
        speaking_rate=rate,
        arxiv_categories=cats,
    )


def reset() -> None:
    """Clear the cached resolved profile. Tests use this after env-var or
    profile-path monkeypatching."""
    load.cache_clear()


__all__ = ["Profile", "ScriptMode", "load", "read_text_for_prompt", "reset"]
