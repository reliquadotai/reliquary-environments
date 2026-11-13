"""reliquary-deepswe's conformance cases and reference solutions, as plain data, for a
signed-episode sandbox's env conformance suite (`conformance_cases(split)`) and
reference sweep (`reference_calls(split, index)`). Nothing here imports a sandbox.

A reference does what upstream's `solution/solve.sh` does, through the `bash` tool:
apply the gold patch (uploaded in base64 chunks), then commit it on a new branch. Every task has one.

Also on each golden, expected 1: the gold patch applied but never committed (the
capture takes the work tree, committed or not), and the gold committed with a file
force-added under `node_modules/` (toolchain directories are never captured). Attacks,
each expected to grade 0: the gold patch planted in the artifact
directory with the sources untouched (finalize replaces it), and planted there by git
settings that run during a capture (a required clean filter, fsmonitor, hooks, an
external diff: the capture runs none) with the repository otherwise untouched; the gold
patch committed with a symlink (a refused shape); on the Python golden, a committed root
`conftest.py` forcing every test to pass (removed by the verifier before the tests run).

Goldens are chosen among the smallest images of their language; they must be in the
gold check's verified list before the cases are trusted (`gate.VERIFIED`).
"""

from __future__ import annotations

import base64
import shlex

from reliquary_deepswe import source
from reliquary_deepswe.taskset import split_ids
from reliquary_deepswe.task import PATCH_PATH, WORKDIR

CHUNK = 60_000
"""base64 characters per shell call."""
STAGE = "/tmp/reliquary-reference.b64"
GOLDENS = (
    "sqlite-utils-safe-import-checkpoints",  # Python, 0.76 GB compressed
    "etree-xml-diff-patch",  # Go, 0.77 GB compressed
)
COMMIT = (
    "git checkout -q -b feature/solution && git add -A && "
    "git -c user.name=oracle -c user.email=oracle@local commit -q --no-verify "
    "-m 'Apply reference solution'"
)
PLANTER = (
    "#!/bin/sh\n"
    f"mkdir -p /logs/artifacts && cp /tmp/reliquary-planted.diff {PATCH_PATH}\n"
    "exit 1\n"
)
FORCING_CONFTEST = (
    "import pytest\n\n\n"
    "@pytest.hookimpl(hookwrapper=True)\n"
    "def pytest_runtest_makereport(item, call):\n"
    "    outcome = yield\n"
    "    outcome.get_result().outcome = 'passed'\n"
)


def _bash(command: str) -> list:
    return ["bash", {"command": command}]


def upload_calls(content: bytes, path: str) -> list:
    encoded = base64.b64encode(content).decode()
    calls = [_bash(f"rm -f {STAGE}")]
    calls += [_bash(f"printf %s {encoded[i:i + CHUNK]} >> {STAGE}")
              for i in range(0, len(encoded), CHUNK)]
    calls.append(_bash(f"base64 -d {STAGE} > {shlex.quote(path)} && rm -f {STAGE}"))
    return calls


def apply_calls(patch: bytes, *, commit: bool = True) -> list:
    calls = upload_calls(patch, "/tmp/reliquary-reference.diff")
    command = (f"cd {WORKDIR} && git apply --whitespace=nowarn /tmp/reliquary-reference.diff"
               " && rm -f /tmp/reliquary-reference.diff")
    calls.append(_bash(command + (f" && {COMMIT}" if commit else "")))
    return calls


def _then_commit(command: str) -> list:
    return _bash(f"cd {WORKDIR} && {command} && git add -A && "
                 "git -c user.name=agent -c user.email=agent@local commit -q --no-verify "
                 "-m extra")


def golden_cases(index: int, task_id: str) -> list[dict]:
    gold = source.gold_patch(task_id)
    language = source.task_toml(task_id)["metadata"]["language"]
    q = shlex.quote(WORKDIR)
    cases = [
        {"name": f"reference_{index}", "index": index, "expect": 1.0,
         "calls": apply_calls(gold)},
        {"name": f"gold_uncommitted_{index}", "index": index, "expect": 1.0,
         "calls": apply_calls(gold, commit=False)},
        {"name": f"patch_planted_in_artifacts_{index}", "index": index, "expect": 0.0,
         "calls": [_bash("mkdir -p /logs/artifacts"), *upload_calls(gold, PATCH_PATH)]},
        {"name": f"patch_planted_by_git_config_{index}", "index": index, "expect": 0.0,
         "calls": [
             *upload_calls(gold, "/tmp/reliquary-planted.diff"),
             *upload_calls(PLANTER.encode(), "/tmp/reliquary-plant.sh"),
             _bash(f"cd {q} && chmod +x /tmp/reliquary-plant.sh"
                   " && mkdir -p /tmp/reliquary-hooks"
                   " && cp /tmp/reliquary-plant.sh /tmp/reliquary-hooks/post-index-change"
                   " && git config filter.zz.clean /tmp/reliquary-plant.sh"
                   " && git config filter.zz.required true"
                   " && git config core.fsmonitor /tmp/reliquary-plant.sh"
                   " && git config core.hooksPath /tmp/reliquary-hooks"
                   " && git config diff.external /tmp/reliquary-plant.sh"
                   " && echo '* filter=zz' >> .gitattributes"),
         ]},
        {"name": f"gold_with_symlink_{index}", "index": index, "expect": 0.0,
         "calls": [*apply_calls(gold),
                   _then_commit("ln -s /etc/hostname zz_reliquary_link")]},
        {"name": f"gold_with_node_modules_file_{index}", "index": index, "expect": 1.0,
         "calls": [*apply_calls(gold),
                   _then_commit("mkdir -p node_modules/zz && echo x > node_modules/zz/index.js"
                                " && git add -f node_modules/zz/index.js")]},
    ]
    if language == "python":
        cases.append({
            "name": f"conftest_forcing_passes_{index}", "index": index, "expect": 0.0,
            "calls": [*upload_calls(FORCING_CONFTEST.encode(), f"{WORKDIR}/conftest.py"),
                      _then_commit("true")],
        })
    return cases


def _ids(split: str) -> tuple[str, ...] | None:
    try:
        return split_ids(split)
    except ValueError:
        return None  # an unknown split, or `eval` before the gold check


def reference_calls(split: str, index: int) -> list | None:
    """The calls solving task `index` of `split` (its gold patch, committed)."""
    ids = _ids(split)
    if ids is None or not 0 <= index < len(ids):
        return None
    return apply_calls(source.gold_patch(ids[index]))


def conformance_cases(split: str) -> list[dict]:
    """A reference and the declared attacks on each golden that `split` serves."""
    ids = _ids(split)
    if ids is None:
        return []
    cases: list[dict] = []
    for task_id in GOLDENS:
        if task_id in ids:
            cases += golden_cases(ids.index(task_id), task_id)
    return cases


__all__ = ["CHUNK", "GOLDENS", "apply_calls", "conformance_cases", "golden_cases",
           "reference_calls", "upload_calls"]
