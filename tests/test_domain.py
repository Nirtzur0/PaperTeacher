"""Domain registry + active-domain resolution."""
from __future__ import annotations

import pytest


def _reset(monkeypatch):
    """Clear the cached active domain so we re-resolve."""
    import paperteacher.domain as d
    d.reset_active()


def test_ml_domain_registered():
    from paperteacher.domain import list_domains
    # Importing paperteacher.domains triggers ml registration.
    import paperteacher.domains  # noqa: F401
    assert "ml" in list_domains()


def test_active_domain_defaults_to_ml(monkeypatch):
    monkeypatch.delenv("PAPERTEACHER_DOMAIN", raising=False)
    monkeypatch.delenv("PAPERTEACHER_PROFILE", raising=False)
    _reset(monkeypatch)
    from paperteacher.domain import active_domain
    assert active_domain().name == "ml"


def test_active_domain_from_env(monkeypatch):
    monkeypatch.setenv("PAPERTEACHER_DOMAIN", "ml")
    _reset(monkeypatch)
    from paperteacher.domain import active_domain
    assert active_domain().name == "ml"


def test_unknown_domain_raises(monkeypatch):
    monkeypatch.setenv("PAPERTEACHER_DOMAIN", "nonexistent")
    _reset(monkeypatch)
    from paperteacher.domain import active_domain
    with pytest.raises(ValueError, match="Unknown domain"):
        active_domain()


def test_register_custom_domain(monkeypatch):
    """Plugin path: a third-party can register their own domain."""
    from paperteacher.domain import register_domain, get_domain

    class FakeDomain:
        name = "fake-test-domain"
        OutlineModel = None

    register_domain("fake-test-domain", FakeDomain)
    assert get_domain("fake-test-domain").name == "fake-test-domain"


def test_domain_from_profile_file(tmp_path, monkeypatch):
    """A `domain: <name>` line in profile.md is honored."""
    profile = tmp_path / "profile.md"
    profile.write_text("name: Test\nfields: [whatever]\ndomain: ml\n")
    monkeypatch.setenv("PAPERTEACHER_PROFILE", str(profile))
    monkeypatch.delenv("PAPERTEACHER_DOMAIN", raising=False)
    # paths.PROFILE_PATH is read at module-import time, so reload.
    import importlib
    import paperteacher.paths
    importlib.reload(paperteacher.paths)
    _reset(monkeypatch)
    from paperteacher.domain import active_domain
    assert active_domain().name == "ml"


def test_env_overrides_profile(tmp_path, monkeypatch):
    """env var beats profile.md."""
    profile = tmp_path / "profile.md"
    profile.write_text("domain: nonexistent\n")
    monkeypatch.setenv("PAPERTEACHER_PROFILE", str(profile))
    monkeypatch.setenv("PAPERTEACHER_DOMAIN", "ml")
    import importlib
    import paperteacher.paths
    importlib.reload(paperteacher.paths)
    _reset(monkeypatch)
    from paperteacher.domain import active_domain
    assert active_domain().name == "ml"
