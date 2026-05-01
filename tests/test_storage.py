"""Storage round-trip + seen-filter behavior."""
from __future__ import annotations

import importlib

import pytest


def _fresh_storage():
    """Re-import storage so it reads the monkeypatched paths.ROOT."""
    import paperteacher.storage as storage_mod
    return importlib.reload(storage_mod)


VALID_OUTLINE_YAML = """\
paper_id: 2603.20105
type: theoretical
core_thesis: Two sentences. Really two.
gap_filled: One thing.
key_concepts: []
key_equations: []
"""


def test_seen_round_trip(paperteacher_home):
    storage = _fresh_storage()
    assert storage.list_seen() == []
    assert storage.seen_ids() == set()

    storage.mark_seen("2603.20105", title="Paper A")
    storage.mark_seen("2603.99999", title="Paper B", note="audit:partial")

    rows = storage.list_seen()
    assert {r["arxiv_id"] for r in rows} == {"2603.20105", "2603.99999"}
    assert storage.is_seen("2603.20105")
    assert not storage.is_seen("0000.00000")
    assert storage.seen_ids() == {"2603.20105", "2603.99999"}


def test_seen_ids_is_set_not_per_call_reread(paperteacher_home, monkeypatch):
    """seen_ids should hit the file once per call, not once per id checked."""
    storage = _fresh_storage()
    storage.mark_seen("a")
    storage.mark_seen("b")

    reads = {"n": 0}
    real_list = storage.list_seen

    def counting_list_seen():
        reads["n"] += 1
        return real_list()

    monkeypatch.setattr(storage, "list_seen", counting_list_seen)
    ids = storage.seen_ids()
    assert ids == {"a", "b"}
    assert reads["n"] == 1


def test_outline_save_validates_and_returns_canonical(paperteacher_home):
    storage = _fresh_storage()
    path, outline = storage.save_outline("2603.20105", VALID_OUTLINE_YAML)
    assert path.exists()
    assert outline.paper_id == "2603.20105"

    # Re-load: the saved form is parseable and equivalent.
    loaded = storage.load_outline("2603.20105")
    assert loaded.paper_id == "2603.20105"
    assert storage.load_outline_yaml("2603.20105") is not None

    # Bad YAML raises ParseError, no file written.
    from paperteacher.models import ParseError
    with pytest.raises(ParseError):
        storage.save_outline("2603.99999", "paper_id: [bad")
    assert storage.load_outline("2603.99999") is None


def test_script_save_load(paperteacher_home):
    storage = _fresh_storage()
    assert storage.load_script("2603.20105") is None
    storage.save_script("2603.20105", "hello world")
    assert storage.load_script("2603.20105") == "hello world"


VALID_AUDIT_YAML = """\
coverage_status: complete
items_missing: []
items_glossed: []
banned_phrases_used: []
voice_first_violations: []
overall_assessment: ok
recommendation: ship
"""


def test_audit_save_returns_decision(paperteacher_home):
    storage = _fresh_storage()
    path, audit = storage.save_audit("2603.20105", VALID_AUDIT_YAML)
    assert path.exists()
    assert audit.recommendation == "ship"


def test_seen_carries_tags(paperteacher_home):
    storage = _fresh_storage()
    storage.mark_seen("a", title="Paper A", tags=["info-geometry", "optimization"])
    storage.mark_seen("b", title="Paper B", tags=["rl"])
    rows = storage.list_seen()
    assert rows[0]["tags"] == ["info-geometry", "optimization"]
    assert rows[1]["tags"] == ["rl"]


def test_topic_distribution_counts_tags(paperteacher_home):
    storage = _fresh_storage()
    storage.mark_seen("a", tags=["info-geometry", "optimization"])
    storage.mark_seen("b", tags=["info-geometry", "rl"])
    storage.mark_seen("c", tags=["rl"])
    dist = storage.topic_distribution()
    assert dist == {"info-geometry": 2, "optimization": 1, "rl": 2}


def test_topic_distribution_window_only_recent(paperteacher_home):
    storage = _fresh_storage()
    for i in range(40):
        storage.mark_seen(str(i), tags=["old"] if i < 30 else ["recent"])
    dist = storage.topic_distribution(window=10)
    assert dist == {"recent": 10}  # last 10 deliveries only


def test_skipped_round_trip(paperteacher_home):
    storage = _fresh_storage()
    assert storage.list_skipped() == []
    storage.mark_skipped("a", title="Paper A", tags=["empirical"], reason="benchmark-only")
    storage.mark_skipped("b", title="Paper B", tags=["theory"], reason="too-narrow")
    rows = storage.list_skipped()
    assert {r["arxiv_id"] for r in rows} == {"a", "b"}
    assert storage.skipped_ids() == {"a", "b"}
    assert rows[0]["reason"] == "benchmark-only"


def test_event_log_appended(paperteacher_home):
    storage = _fresh_storage()
    storage.mark_seen("2603.20105")
    storage.save_outline("2603.20105", VALID_OUTLINE_YAML)
    storage.save_script("2603.20105", "x")
    storage.save_audit("2603.20105", VALID_AUDIT_YAML)

    import paperteacher.paths as paths_mod
    log_lines = paths_mod.EVENT_LOG.read_text().strip().splitlines()
    events = [line.split('"event":')[1].split('"')[1] for line in log_lines]
    # All four writes left a structured event.
    assert {"seen_marked", "outline_saved", "script_saved", "audit_saved"} <= set(events)
