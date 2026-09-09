"""
conftest.py — sys.path setup for selection tests.

Adds the repo root (so `selection.lib.*` and `scripts.lib.*` resolve as
namespace packages) and this tests/ dir (for `util`).  Does NOT add
`selection/` itself to sys.path — that would make `lib` ambiguous with
`scripts/lib` when curation + selection tests run in one pytest session.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TESTS = os.path.dirname(os.path.abspath(__file__))

for _p in (_ROOT, _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
