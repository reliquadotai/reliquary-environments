"""Which tasks the served `eval` split holds: those whose reference scores 1 offline.

Upstream graded with network (`pip install -e .` at grading time) and without a budget.
A signed-episode sandbox grades with no network and within 810 s, so a task is served
only after `scripts/gold_check.py` (run on a gVisor host with the images) has graded the
reference source 1 and an empty workspace 0 through this package's own hooks, inside the
budget.

`VERIFIED` is that list (`gold_check.py --write-gate` rewrites this file); None until the
gold check has run, and the `eval` split then refuses to load (`eval_all` serves all 104,
for the gold check and conformance runs). `EXCLUDED` names tasks left out for a reason
known before the gold check. A `.py` file on purpose: the package's identity digests its
`.py` files, so the served set cannot change under the same identity.
"""

from __future__ import annotations

VERIFIED: frozenset[str] | None = None
"""Task ids whose reference scored 1 and empty workspace 0 offline within the budget."""

GOLD_CHECK: dict[str, str] = {}
"""Provenance of `VERIFIED`: where and when the gold check ran (host, runtime, date)."""

EXCLUDED: dict[str, str] = {
    "pyautogui": "the image runs as appuser, which cannot create /logs (no artifact can be "
                 "captured), and needs an X display; no reference survives in the image",
}
"""Task id -> why it is never served."""
