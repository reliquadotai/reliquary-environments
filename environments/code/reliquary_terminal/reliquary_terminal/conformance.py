"""reliquary-terminal's conformance cases and reference solutions, as plain data, for a
signed-episode sandbox's env conformance suite (`conformance_cases(split)`) and reference
sweep (`reference_calls(split, index)`). Only the `tmax` split is served on a sandbox;
every other split answers nothing. Nothing here imports a sandbox.

A reference replays the task's recorded solution (`solution/solve.sh`) through the `bash`
tool, uploaded in base64 chunks (one tool call's arguments are bounded). The other cases
are canaries: each leaves files where a grading command could pick them up -- a
`conftest.py` in /home/user and /app, a `usercustomize.py` in /home/user's user site, a
`sitecustomize.py` in /app, `bash` and `python3` in /home/user/.local/bin (first on PATH
for a task that put it there) -- and each file, if executed, only creates a marker under
/tmp. A case expects 0 and lists its markers under `absent`: the suite checks, in the
grading box after scoring, that none exists (the grader never ran the agent's files). The
cases cover task 0 and the first task whose final-state test runs programs (`subprocess`).

Indices are those of the split served with the taskset's defaults (no `num_tasks` that
excludes them, no `tasks` filter). A split served otherwise names other tasks under these
indices: its reference case then fails.
"""

from __future__ import annotations

import base64
import functools
from pathlib import Path

import verifiers.v1 as vf

SPLITS = ("tmax",)
CHUNK = 60_000
"""base64 characters per shell call."""
STAGE = "/tmp/reliquary-reference.b64"
SCRIPT = "/tmp/reliquary-reference.sh"
SCAN_LIMIT = 500
"""Tasks scanned for one whose final-state test runs programs."""
TMAX_SOURCE: Path | None = None
"""The TMax source the cases load (None: the pinned download); tests point it elsewhere."""

CANARY_DIR = "/tmp"
"""Where a canary leaves its marker: outside the artifact roots, so a marker in the
grading box was made there."""
USER_SITES = tuple(f"/home/user/.local/lib/python3.{minor}/site-packages"
                   for minor in range(8, 13))


def marker(case: str) -> str:
    return f"{CANARY_DIR}/.canary-{case}"


def python_canary(path: str) -> bytes:
    """Python that, imported, only creates the marker at `path`."""
    return f"open({path!r}, 'a').close()\n".encode()


def program_canary(path: str) -> bytes:
    """An executable that, run, only creates the marker at `path`."""
    return f"#!/bin/sh\n: >> '{path}'\n".encode()


def _bash(command: str) -> list:
    return ["bash", {"command": command}]


def upload_calls(content: bytes, path: str) -> list:
    encoded = base64.b64encode(content).decode()
    calls = [_bash(f"rm -f {STAGE}")]
    calls += [_bash(f"printf %s {encoded[i:i + CHUNK]} >> {STAGE}")
              for i in range(0, len(encoded), CHUNK)]
    calls.append(_bash(f"mkdir -p {Path(path).parent} && base64 -d {STAGE} > {path}"
                       f" && rm -f {STAGE}"))
    return calls


def run_script_calls(script: bytes) -> list:
    return [*upload_calls(script, SCRIPT),
            _bash(f"bash {SCRIPT}; rc=$?; rm -f {SCRIPT}; exit $rc")]


def _canary(name: str, index: int, files: list[tuple[str, bytes]],
            executable: bool = False) -> dict:
    calls = [call for path, content in files for call in upload_calls(content, path)]
    if executable:
        calls.append(_bash("chmod 755 " + " ".join(path for path, _ in files)))
    return {"name": f"{name}_{index}", "index": index, "expect": 0.0, "calls": calls,
            "absent": [marker(f"{name}_{index}")]}


def task_cases(index: int, solve: bytes) -> list[dict]:
    def mark(name):
        return marker(f"{name}_{index}")

    conftest = python_canary(mark("conftest_in_the_roots"))
    usercustomize = python_canary(mark("usercustomize_in_the_user_site"))
    program = program_canary(mark("programs_in_the_user_bin"))
    return [
        {"name": f"reference_{index}", "index": index, "expect": 1.0,
         "calls": run_script_calls(solve)},
        _canary("conftest_in_the_roots", index,
                [("/home/user/conftest.py", conftest), ("/app/conftest.py", conftest)]),
        _canary("usercustomize_in_the_user_site", index,
                [(f"{site}/usercustomize.py", usercustomize) for site in USER_SITES]),
        _canary("sitecustomize_in_app", index,
                [("/app/sitecustomize.py", python_canary(mark("sitecustomize_in_app")))]),
        _canary("programs_in_the_user_bin", index,
                [("/home/user/.local/bin/bash", program),
                 ("/home/user/.local/bin/python3", program)], executable=True),
    ]


@functools.cache
def _taskset(split: str):
    from reliquary_terminal.taskset import TerminalTaskset

    config = vf.taskset_config_type("reliquary-terminal")(
        id="reliquary-terminal", split=split, tmax_source=TMAX_SOURCE)
    return TerminalTaskset(config)


def _solve(split: str, index: int) -> bytes | None:
    task = _taskset(split).task_at(index)
    path = Path(task.data.task_dir) / "solution" / "solve.sh"
    return path.read_bytes() if path.is_file() else None


def separation_index(split: str) -> int | None:
    """The first task (within SCAN_LIMIT) whose final-state test starts programs."""
    from reliquary_terminal import tmax
    from reliquary_terminal.taskset import TMAX_SPLITS, tmax_tasks

    _, kept = tmax_tasks(None, part=TMAX_SPLITS[split])
    source = _taskset(split)._source()
    for index, (task_id, _) in enumerate(kept[:SCAN_LIMIT]):
        if "subprocess" in source.text(task_id, tmax.FINAL_TEST):
            return index
    return None


def reference_calls(split: str, index: int) -> list | None:
    if split not in SPLITS:
        return None
    solve = _solve(split, index)
    return None if solve is None else run_script_calls(solve)


def conformance_cases(split: str) -> list[dict]:
    if split not in SPLITS:
        return []
    indices = {0}
    found = separation_index(split)
    if found is not None:
        indices.add(found)
    cases: list[dict] = []
    for index in sorted(indices):
        solve = _solve(split, index)
        if solve is not None:
            cases += task_cases(index, solve)
    return cases
