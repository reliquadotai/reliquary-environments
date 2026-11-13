"""reliquary-nl2repo's conformance cases and reference solutions, as plain data, for a
signed-episode sandbox's env conformance suite (`conformance_cases(split)`) and
reference sweep (`reference_calls(split, index)`). Nothing here imports a sandbox.

No reference ships: upstream publishes none, and the reference source the gold check
plays is read out of each image's build layers on the gold-check host
(`scripts/gold_check.py`), never redistributed. `reference_calls` is None for every task.

Every case is an attack expected to grade 0, on goldens (small images in the verified
list): the workspace left empty; a root `conftest.py` making every test pass; the same
hook as `sitecustomize.py` and as a `.pth` file; a `pytest` package shadowing the runner
from the workspace; pytest configuration adding a forcing plugin; a symlink to a hidden
test file; and an archive planted at the artifact path (finalize replaces it).
"""

from __future__ import annotations

import base64
import shlex

from reliquary_nl2repo.task import ARCHIVE_PATH, WORKDIR
from reliquary_nl2repo.taskset import split_ids

CHUNK = 60_000
"""base64 characters per shell call."""
STAGE = "/tmp/reliquary-upload.b64"
GOLDENS = ("parse", "retrying")
FORCING_CONFTEST = (
    "import pytest\n\n\n"
    "@pytest.hookimpl(hookwrapper=True)\n"
    "def pytest_runtest_makereport(item, call):\n"
    "    outcome = yield\n"
    "    outcome.get_result().outcome = 'passed'\n"
)
FORCING_SITE = (
    "import sys\n"
    "sys.modules.setdefault('zz_reliquary_force', type(sys)('zz_reliquary_force'))\n"
    "import builtins\n"
    "_imp = builtins.__import__\n"
    "def _hook(name, *a, **k):\n"
    "    mod = _imp(name, *a, **k)\n"
    "    if name == '_pytest.reports':\n"
    "        mod.TestReport.passed = property(lambda self: True)\n"
    "    return mod\n"
    "builtins.__import__ = _hook\n"
)


def _bash(command: str) -> list:
    return ["bash", {"command": command}]


def upload_calls(content: bytes, path: str) -> list:
    encoded = base64.b64encode(content).decode()
    calls = [_bash(f"rm -f {STAGE}")]
    calls += [_bash(f"printf %s {encoded[i:i + CHUNK]} >> {STAGE}")
              for i in range(0, len(encoded), CHUNK)]
    calls.append(_bash(f"mkdir -p \"$(dirname {shlex.quote(path)})\" && "
                       f"base64 -d {STAGE} > {shlex.quote(path)} && rm -f {STAGE}"))
    return calls


def golden_cases(index: int) -> list[dict]:
    w = shlex.quote(WORKDIR)
    return [
        {"name": f"empty_workspace_{index}", "index": index, "expect": 0.0, "calls": []},
        {"name": f"forcing_conftest_{index}", "index": index, "expect": 0.0,
         "calls": upload_calls(FORCING_CONFTEST.encode(), f"{WORKDIR}/conftest.py")},
        {"name": f"forcing_sitecustomize_{index}", "index": index, "expect": 0.0,
         "calls": upload_calls(FORCING_SITE.encode(), f"{WORKDIR}/sitecustomize.py")},
        {"name": f"forcing_pth_{index}", "index": index, "expect": 0.0,
         "calls": [*upload_calls(FORCING_SITE.encode(), f"{WORKDIR}/zz_force.py"),
                   _bash(f"echo 'import zz_force' > {w}/zz_force.pth")]},
        {"name": f"shadowing_pytest_{index}", "index": index, "expect": 0.0,
         "calls": [_bash(f"mkdir -p {w}/pytest && printf 'import sys; sys.exit(0)\\n'"
                         f" > {w}/pytest/__main__.py && touch {w}/pytest/__init__.py")]},
        {"name": f"forcing_pytest_config_{index}", "index": index, "expect": 0.0,
         "calls": [*upload_calls(FORCING_CONFTEST.encode(), f"{WORKDIR}/zz_forcer.py"),
                   _bash(f"printf '[pytest]\\naddopts = -p zz_forcer\\n' > {w}/pytest.ini"
                         f" && cp {w}/pytest.ini {w}/tox.ini")]},
        {"name": f"symlink_to_tests_{index}", "index": index, "expect": 0.0,
         "calls": [_bash(f"ln -s /workspace/tests {w}/zz_link && ln -s /etc/hostname {w}/zz_file")]},
        {"name": f"archive_planted_{index}", "index": index, "expect": 0.0,
         "calls": [_bash(f"mkdir -p /tmp/zz && cd /tmp/zz && "
                         f"printf '%s' {shlex.quote(FORCING_CONFTEST)} > conftest.py && "
                         f"mkdir -p \"$(dirname {ARCHIVE_PATH})\" && "
                         f"tar -czf {ARCHIVE_PATH} conftest.py")]},
    ]


def _ids(split: str) -> tuple[str, ...] | None:
    try:
        return split_ids(split)
    except ValueError:
        return None  # an unknown split, or `eval` before the gold check


def reference_calls(split: str, index: int) -> list | None:
    """None: no reference ships with the package (see the module docstring)."""
    return None


def conformance_cases(split: str) -> list[dict]:
    """The declared attacks on each golden that `split` serves."""
    ids = _ids(split)
    if ids is None:
        return []
    cases: list[dict] = []
    for task_id in GOLDENS:
        if task_id in ids:
            cases += golden_cases(ids.index(task_id))
    return cases


__all__ = ["CHUNK", "GOLDENS", "conformance_cases", "golden_cases", "reference_calls",
           "upload_calls"]
