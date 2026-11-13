"""Which tasks the served `eval` split holds: those whose gold patch grades 1 offline.

Upstream validated its gold patches with network at grading time (`network_mode =
"public"` for the verifier) and a 3,000 s verifier budget. A signed-episode sandbox grades
with no network and within 810 s, so a task is served only after `scripts/gold_check.py`
(run on a gVisor host with the images) has graded its gold patch 1 and the empty patch 0
through this package's own hooks, inside the budget.

`VERIFIED` is that list (`gold_check.py --write-gate` rewrites this file); None until the
gold check has run, and the `eval` split then refuses to load (`eval_all` serves all 642,
for the gold check and conformance runs). `EXCLUDED` names tasks left out for a reason
known before the gold check, gold passing or not. A `.py` file on purpose: the package's
identity digests its `.py` files, so the served set cannot change under the same identity.
"""

from __future__ import annotations

VERIFIED: frozenset[str] | None = None
"""Task ids whose gold patch graded 1 offline within the budget; None: not gated yet."""

GOLD_CHECK: dict[str, str] = {}
"""Provenance of `VERIFIED`: where and when the gold check ran (host, runtime, date)."""

EXCLUDED: dict[str, str] = {}
"""Task id -> why it is never served."""
