"""TTS backend chunking + factory tests. The actual TTS calls are not made;
we exercise the deterministic chunking logic and the factory routing.
"""
from __future__ import annotations

import pytest

from paperteacher import tts


def test_factory_routes_kokoro_and_vertex():
    assert tts.get_backend("kokoro").name == "kokoro"
    assert tts.get_backend("vertex").name == "vertex"


def test_factory_rejects_unknown():
    with pytest.raises(ValueError):
        tts.get_backend("nonsense")


def test_factory_default_falls_back_to_kokoro(monkeypatch):
    monkeypatch.setenv("PAPERTEACHER_TTS", "")
    # Reload paths so DEFAULT_TTS_BACKEND picks up the empty env
    import importlib
    import paperteacher.paths as paths_mod
    importlib.reload(paths_mod)
    importlib.reload(tts)
    assert tts.get_backend().name == "kokoro"


# ---- Vertex chunking -----------------------------------------------------


def test_vertex_short_text_is_one_chunk():
    b = tts.VertexBackend()
    chunks = b._chunk("This is short.")
    assert chunks == ["This is short."]


def test_vertex_chunks_long_text_under_limit():
    b = tts.VertexBackend()
    para = "Hello world. " * 100  # ~1.3 KB; well under limit
    chunks = b._chunk(para * 5)  # ~6.5 KB; over the 4500-byte limit
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.encode("utf-8")) <= b.MAX_BYTES_PER_REQUEST


def test_vertex_chunks_split_on_paragraph_boundaries_when_possible():
    b = tts.VertexBackend()
    # Two paragraphs each ~3 KB — should split between them
    p1 = "First paragraph. " * 200
    p2 = "Second paragraph. " * 200
    chunks = b._chunk(f"{p1}\n\n{p2}")
    assert len(chunks) >= 2
    # Each chunk fits
    for c in chunks:
        assert len(c.encode("utf-8")) <= b.MAX_BYTES_PER_REQUEST


def test_vertex_chunks_oversized_paragraph_on_sentences():
    b = tts.VertexBackend()
    # One paragraph far over the limit; only sentence-level splitting can save it
    text = ("This is a sentence. " * 500)  # ~10 KB single paragraph
    chunks = b._chunk(text)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c.encode("utf-8")) <= b.MAX_BYTES_PER_REQUEST


def test_vertex_chunks_concatenated_text_roundtrip():
    """No content should be lost when chunking — the concatenated chunks should
    contain (modulo whitespace normalization) the same words as the input."""
    b = tts.VertexBackend()
    text = ("Sentence one. Sentence two. " * 300)
    chunks = b._chunk(text)
    rejoined_words = " ".join(chunks).split()
    original_words = text.split()
    assert rejoined_words == original_words
