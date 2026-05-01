"""Shared fixtures: redirect PAPERTEACHER_HOME to a tmp dir per test, so
storage tests don't pollute the user's real ~/.paperteacher.

Importing the package modules AFTER setting the env var ensures `paths.ROOT`
picks up the tmp path. We force a re-import in the fixture rather than relying
on lazy module state.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


@pytest.fixture
def paperteacher_home(tmp_path, monkeypatch):
    """Isolate filesystem state under tmp_path; reload paths-aware modules."""
    monkeypatch.setenv("PAPERTEACHER_HOME", str(tmp_path))
    monkeypatch.setenv("PAPERTEACHER_PROFILE", str(tmp_path / "profile.md"))
    # Reload so module-level constants pick up the env vars.
    import paperteacher.paths as paths_mod
    importlib.reload(paths_mod)
    import paperteacher.storage as storage_mod
    importlib.reload(storage_mod)
    # Reset cached active domain so tests that switch PAPERTEACHER_DOMAIN
    # via monkeypatch.setenv get a fresh resolution.
    import paperteacher.domain as domain_mod
    domain_mod.reset_active()
    paths_mod.ensure_layout()
    return tmp_path
