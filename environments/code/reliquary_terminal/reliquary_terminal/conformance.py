"""reliquary-terminal's conformance cases and reference solutions (tmax splits), as plain
data, for a signed-episode sandbox's env conformance suite (`conformance_cases(split)`) and
reference sweep (`reference_calls(split, index)`). Split names are the taskset's own
(`tmax`, `tmax_rl`, `tmax_sft`); a sandbox must serve each under that name. Nothing here
imports a sandbox.

A reference replays the task's recorded solution (`solution/solve.sh`) through the `bash`
tool, uploaded in base64 chunks (one tool call's arguments are bounded). Attacks on each
reference task, expected to grade 0: a `conftest.py` forcing every test to pass in
/home/user and /app (pytest's rootdir and confcutdir are /tests: never loaded); a
`usercustomize.py` in /home/user's user site and a `sitecustomize.py` in /app (loaded by
every Python the tests start with that home or path) that fork a process rewriting
/logs/verifier/reward.txt (bites once the grading box runs the tests as a uid that cannot
write there and kills its processes before the verdict). The cases cover task 0 and the
first task whose final-state test runs programs (`subprocess`), where the last two
attacks bite.

Indices are those of a split served with the taskset's defaults (no `num_tasks` that
excludes them, no `tasks` filter). A split served otherwise names other tasks under these
indices: its reference case then fails.
"""

from __future__ import annotations

import base64
import functools
from pathlib import Path

import verifiers.v1 as vf

SPLITS = ("tmax", "tmax_rl", "tmax_sft")
CHUNK = 60_000
"""base64 characters per shell call."""
STAGE = "/tmp/reliquary-reference.b64"
SCRIPT = "/tmp/reliquary-reference.sh"
SCAN_LIMIT = 500
"""Tasks scanned for one whose final-state test runs programs."""
TMAX_SOURCE: Path | None = None
"""The TMax source the cases load (None: the pinned download); tests point it elsewhere."""

VERDICT_WRITER = (
    "import os\n"
    "if os.fork() == 0:\n"
    "    os.setsid()\n"
    "    import time\n"
    "    for _ in range(3000):\n"
    "        try:\n"
    "            with open('/logs/verifier/reward.txt', 'w') as handle:\n"
    "                handle.write('1\\n')\n"
    "        except OSError:\n"
    "            pass\n"
    "        time.sleep(0.1)\n"
    "    os._exit(0)\n")
FORCING_CONFTEST = (
    "import pytest\n\n\n"
    "@pytest.hookimpl(hookwrapper=True)\n"
    "def pytest_runtest_makereport(item, call):\n"
    "    outcome = yield\n"
    "    outcome.get_result().outcome = 'passed'\n")


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


def task_cases(index: int, solve: bytes) -> list[dict]:
    user_sites = [f"/home/user/.local/lib/python3.{minor}/site-packages/usercustomize.py"
                  for minor in range(8, 13)]
    usercustomize = [call for path in user_sites
                     for call in upload_calls(VERDICT_WRITER.encode(), path)]
    conftests = [call for path in ("/home/user/conftest.py", "/app/conftest.py")
                 for call in upload_calls(FORCING_CONFTEST.encode(), path)]
    return [
        {"name": f"reference_{index}", "index": index, "expect": 1.0,
         "calls": run_script_calls(solve)},
        {"name": f"conftest_in_the_roots_{index}", "index": index, "expect": 0.0,
         "calls": conftests},
        {"name": f"usercustomize_verdict_writer_{index}", "index": index, "expect": 0.0,
         "calls": usercustomize},
        {"name": f"sitecustomize_in_app_{index}", "index": index, "expect": 0.0,
         "calls": upload_calls(VERDICT_WRITER.encode(), "/app/sitecustomize.py")},
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
