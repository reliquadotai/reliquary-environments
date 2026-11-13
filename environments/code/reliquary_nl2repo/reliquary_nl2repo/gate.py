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

VERIFIED: frozenset[str] | None = frozenset({
    'arguably',
    'arxiv-mcp-server',
    'asteval',
    'autojump',
    'bleach',
    'boltons',
    'box',
    'cachier',
    'cherry',
    'dbutils',
    'deslib',
    'dictdatabase',
    'docopt-ng',
    'emoji',
    'fastapi-users',
    'frontmatter',
    'funcy',
    'humanize',
    'ipytest',
    'jinja',
    'jsonlines',
    'justext',
    'markupsafe',
    'math-verify',
    'more-Itertools',
    'paillier',
    'parse',
    'pathlib2',
    'pdfplumber-stable',
    'plac',
    'pss',
    'pyperclip',
    'python-jose',
    'python-pathspec',
    'python-patterns',
    'python-slugify',
    'pythonprojecttemplate',
    'pytorch-grad-cam',
    'records',
    'retrying',
    'schedule-master',
    'schema',
    'six',
    'sortedcontainers',
    'sqlparse',
    'tenacity',
    'trimming',
    'typing_extensions',
    'uiautomator',
    'unidecode',
    'verifiers',
    'voluptuous',
    'wsgidav',
})
"""Task ids whose reference scored 1 and empty workspace 0 offline within the budget."""

GOLD_CHECK: dict[str, str] = {
    'host': 'nl2r-gold-01 (Targon cpu-medium-amd, 8 vCPU, gVisor 20260928.0)',
    'passes': 'references of 30 tasks below 1 re-run after 3e02a93; the rest at 221f276',
    'runtime': 'runsc',
    'date': '2026-11-13',
    'budget_s': '700.0',
    'rule': 'reference >= 1.0, empty <= 0.0',
    'results_sha256': 'db08f071c6e173646e7a5ccc861836c354f6f1ec481dac722121b12af11d5f3d',
    'verified': '53/104',
}
"""Provenance of `VERIFIED`: where and when the gold check ran (host, runtime, date)."""

EXCLUDED: dict[str, str] = {'pyautogui': 'the image runs as appuser, which cannot create /logs (no artifact can be captured), and needs an X display; no reference survives in the image'}
"""Task id -> why it is never served."""
