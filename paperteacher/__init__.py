"""PaperTeacher: 3-stage paper-to-podcast pipeline with local TTS.

Submodules import on demand — `paperteacher.audio` pulls numpy/kokoro,
`paperteacher.models` does not. This keeps the import surface small for
callers that only need part of the system.
"""
from __future__ import annotations

__version__ = "0.2.0"
