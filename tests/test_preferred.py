"""Tests for the preferred-authors allowlist."""
from __future__ import annotations

import importlib

import pytest


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERTEACHER_HOME", str(tmp_path))
    monkeypatch.setenv("PAPERTEACHER_PREFERRED", str(tmp_path / "preferred.yaml"))
    import paperteacher.paths as paths_mod
    importlib.reload(paths_mod)
    import paperteacher.preferred as pref_mod
    importlib.reload(pref_mod)
    return pref_mod


def _candidate(arxiv_id, authors, score=0.0):
    from paperteacher.domains._common import Candidate
    return Candidate(
        arxiv_id=arxiv_id,
        title=f"paper {arxiv_id}",
        authors=authors,
        summary="",
        source="arxiv_cs.LG",
        score=score,
    )


def test_load_returns_none_when_file_missing(tmp_path, monkeypatch):
    pref_mod = _reload(monkeypatch, tmp_path)
    assert pref_mod.load() is None


def test_load_returns_none_when_authors_empty(tmp_path, monkeypatch):
    pref_mod = _reload(monkeypatch, tmp_path)
    (tmp_path / "preferred.yaml").write_text("authors: []\nboost: 50\n")
    assert pref_mod.load() is None


def test_load_returns_none_on_invalid_yaml(tmp_path, monkeypatch):
    pref_mod = _reload(monkeypatch, tmp_path)
    (tmp_path / "preferred.yaml").write_text("authors: [unterminated\n")
    assert pref_mod.load() is None


def test_load_returns_none_when_top_level_not_mapping(tmp_path, monkeypatch):
    pref_mod = _reload(monkeypatch, tmp_path)
    (tmp_path / "preferred.yaml").write_text("- just\n- a\n- list\n")
    assert pref_mod.load() is None


def test_load_parses_authors_and_boost(tmp_path, monkeypatch):
    pref_mod = _reload(monkeypatch, tmp_path)
    (tmp_path / "preferred.yaml").write_text(
        "authors:\n  - Chris Olah\n  - Neel Nanda\nboost: 75.0\n"
    )
    pref = pref_mod.load()
    assert pref is not None
    assert pref.authors == ("Chris Olah", "Neel Nanda")
    assert pref.boost == 75.0


def test_load_uses_default_boost_when_omitted(tmp_path, monkeypatch):
    pref_mod = _reload(monkeypatch, tmp_path)
    (tmp_path / "preferred.yaml").write_text("authors:\n  - Chris Olah\n")
    pref = pref_mod.load()
    assert pref is not None
    assert pref.boost == pref_mod.DEFAULT_BOOST


def test_matches_case_insensitive_substring(tmp_path, monkeypatch):
    pref_mod = _reload(monkeypatch, tmp_path)
    pref = pref_mod.Preferred(authors=("Olah",), boost=100.0)
    assert pref.matches(_candidate("1", ["Christopher Olah", "Other"]))
    assert pref.matches(_candidate("2", ["chris olah"]))
    assert not pref.matches(_candidate("3", ["Geoff Hinton"]))


def test_matches_handles_empty_authors_field(tmp_path, monkeypatch):
    pref_mod = _reload(monkeypatch, tmp_path)
    pref = pref_mod.Preferred(authors=("Olah",), boost=100.0)
    assert not pref.matches(_candidate("1", []))


def test_apply_boosts_matched_and_sorts(tmp_path, monkeypatch):
    pref_mod = _reload(monkeypatch, tmp_path)
    pref = pref_mod.Preferred(authors=("Olah",), boost=100.0)
    cands = [
        _candidate("1", ["No Match"], score=10.0),
        _candidate("2", ["Chris Olah"], score=5.0),
        _candidate("3", ["Other"], score=20.0),
    ]
    out = pref_mod.apply(cands, pref)
    assert out[0].arxiv_id == "2"
    assert out[0].score == 105.0
    assert out[1].arxiv_id == "3"
    assert out[2].arxiv_id == "1"


def test_apply_no_match_preserves_relative_order(tmp_path, monkeypatch):
    pref_mod = _reload(monkeypatch, tmp_path)
    pref = pref_mod.Preferred(authors=("Olah",), boost=100.0)
    cands = [_candidate("1", ["A"], score=5.0), _candidate("2", ["B"], score=10.0)]
    out = pref_mod.apply(cands, pref)
    assert out[0].arxiv_id == "2" and out[0].score == 10.0
    assert out[1].arxiv_id == "1" and out[1].score == 5.0
