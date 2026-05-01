"""Domain packs namespace.

This `__init__.py` is intentionally empty (no imports). Bundled packs
(`paperteacher.domains.ml`, etc.) are imported lazily by
`paperteacher.domain.active_domain()` so that domain.py and _common.py can
import each other without triggering pack registration during module load
(which would create a circular import).

Add a new domain by:
  1. Creating `paperteacher/domains/<name>/__init__.py` with a class that
     conforms to `paperteacher.domain.Domain` and calls `register_domain(
     "<name>", YourClass)` at module level.
  2. Adding `from . import <name>` to `_BUNDLED` in `paperteacher.domain`.
"""
