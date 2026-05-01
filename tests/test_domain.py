"""Domain registry + active-domain resolution."""
from __future__ import annotations

import pytest


def _reset(monkeypatch):
    """Clear the cached active domain so we re-resolve."""
    import paperteacher.domain as d
    d.reset_active()


def test_ml_domain_registered(monkeypatch):
    """Lazy load runs on first `active_domain()` call (avoids a circular
    import at module-load time — see domain.py for the rationale)."""
    monkeypatch.delenv("PAPERTEACHER_DOMAIN", raising=False)
    monkeypatch.delenv("PAPERTEACHER_PROFILE", raising=False)
    _reset(monkeypatch)
    from paperteacher.domain import active_domain, list_domains
    active_domain()  # trigger _ensure_bundled_domains_loaded()
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


# ---- multi-pack routing --------------------------------------------------


def _register_fake(name: str):
    """Register a minimal fake domain pack for routing tests."""
    from paperteacher.domain import register_domain
    from paperteacher.domains._common import Candidate, PaperText

    class _Fake:
        def __init__(self):
            self.name = name
            self.OutlineModel = None

        async def discover(self, arxiv_categories=None, limit=20):
            return [Candidate(
                arxiv_id=f"{name}-001",
                title=f"{name} paper",
                authors=[],
                summary="",
                source=f"{name}_src",
            )]

        async def read(self, arxiv_id, max_chars=None):
            return PaperText(
                arxiv_id=arxiv_id,
                title=f"{name} paper",
                text="hello",
                source=f"{name}_reader",
            )

    register_domain(name, _Fake)


def test_active_domains_from_profile_list(tmp_path, monkeypatch):
    """A `domains: a, b` line resolves to multiple packs in order."""
    _register_fake("fake-a")
    _register_fake("fake-b")
    profile = tmp_path / "profile.md"
    profile.write_text("name: Test\ndomains: fake-a, fake-b\n")
    monkeypatch.setenv("PAPERTEACHER_PROFILE", str(profile))
    monkeypatch.delenv("PAPERTEACHER_DOMAIN", raising=False)
    import importlib
    import paperteacher.paths
    importlib.reload(paperteacher.paths)
    _reset(monkeypatch)
    from paperteacher.domain import active_domains
    names = [d.name for d in active_domains()]
    assert names == ["fake-a", "fake-b"]


def test_env_supports_comma_list(monkeypatch):
    """PAPERTEACHER_DOMAIN accepts a comma-separated list."""
    _register_fake("fake-a")
    _register_fake("fake-b")
    monkeypatch.setenv("PAPERTEACHER_DOMAIN", "fake-b,fake-a")
    monkeypatch.delenv("PAPERTEACHER_PROFILE", raising=False)
    _reset(monkeypatch)
    from paperteacher.domain import active_domains
    assert [d.name for d in active_domains()] == ["fake-b", "fake-a"]


def test_discover_all_fans_out_and_stamps_domain(monkeypatch):
    """discover_all() runs every active pack and tags each candidate."""
    import asyncio
    _register_fake("fake-a")
    _register_fake("fake-b")
    monkeypatch.setenv("PAPERTEACHER_DOMAIN", "fake-a,fake-b")
    monkeypatch.delenv("PAPERTEACHER_PROFILE", raising=False)
    _reset(monkeypatch)
    from paperteacher.domain import discover_all
    cands = asyncio.run(discover_all())
    assert {c.arxiv_id for c in cands} == {"fake-a-001", "fake-b-001"}
    assert {c.domain for c in cands} == {"fake-a", "fake-b"}


def test_read_paper_records_meta_and_routes(tmp_path, monkeypatch):
    """read_paper() with a hint routes to that pack and stamps meta/<id>.json."""
    import asyncio
    _register_fake("fake-a")
    _register_fake("fake-b")
    monkeypatch.setenv("PAPERTEACHER_HOME", str(tmp_path))
    monkeypatch.setenv("PAPERTEACHER_DOMAIN", "fake-a,fake-b")
    monkeypatch.delenv("PAPERTEACHER_PROFILE", raising=False)
    import importlib
    import paperteacher.paths
    importlib.reload(paperteacher.paths)
    _reset(monkeypatch)
    from paperteacher.domain import domain_for, read_paper
    paper = asyncio.run(read_paper("xyz-7", hint="fake-b"))
    assert paper.source == "fake-b_reader"
    # meta sidecar got written, so subsequent routing returns fake-b.
    assert domain_for("xyz-7").name == "fake-b"
