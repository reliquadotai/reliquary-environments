"""Per-test results next to the binary reward.

Every task of every split grades with pytest through `pytest-json-ctrf`
(`--ctrf /logs/verifier/ctrf.json`, in all 89 Terminal-Bench 2.1 `test.sh`,
all 64 MiMo ones and the one `tmax` generates), then writes 1 or 0 to
`reward.txt`. verifiers reads
`reward.txt` and discards the rest: `test.sh`'s exit code, its output, and the
CTRF report that says which tests failed. `TerminalTask` keeps them, on the
trace that gets the reward:

- `trace.info["grading"]`: `test.sh`'s exit code and the tail of its output
  (where `anti_hack_guard.py` says why it rejected a tree), and the CTRF
  tests -- name, status, and the failure message, truncated;
- `trace.metrics`: `tests_total`, `tests_passed`, `tests_failed`, when a
  report was written.

The reward is still `reward.txt`, read by verifiers' own code: this only adds
what used to be thrown away, and nothing it does after the reward is read --
a malformed or hostile report, a failing read -- can change the outcome.

Two things can only lower it, never raise it:

- `reward.json` is never read, in either split. verifiers prefers it over
  `reward.txt`, but no `test.sh` of either split writes it (all 89 + 64 write
  `reward.txt`), so one in the box was planted: by the agent, or by a process
  it left running;
- in a separate grading box (`train`), a nonzero `reward.txt` next to a
  nonzero `test.sh` exit is 0, recorded as `reward_without_success`: a MiMo
  `test.sh` writes 1 only when pytest passed (about half then exit with
  pytest's status, the rest exit 0 regardless), so a 1 with a failing exit was
  written by something else -- a background process a planted file started;
- in a separate grading box, a nonzero reward also needs the CTRF report to
  show no failed test, at least one passed, and every test pytest collects by
  default from the row's `tests/test_*.py` (`collected_tests`: `test_*`
  functions at module level, `test_*` methods of `Test*` classes without
  `__init__`) among the passed ones; otherwise 0, recorded as
  `ctrf_incomplete`. Agent code imported by the tests can end pytest early
  with `os._exit(0)`: exit 0, reward 1, and a report without the tests that
  never ran. Every MiMo `test.sh` passes `--ctrf /logs/verifier/ctrf.json`
  (pytest-json-ctrf 0.5.2 in all 64 images: names are `<path>::[Class::]test`,
  parameters dropped).

Terminal-Bench (`eval`) keeps its own semantics: it is graded as it ships, in
the agent's box, untrusted anyway (below).

**Untrusted in the `eval` split.** There `test.sh` runs in the box the agent
worked in, as Terminal-Bench ships it, and anything the agent left running
there can write `/logs/verifier/ctrf.json` or the output this records. The
per-test data is fit for debugging there, never for partial credit. In the
`train` split it comes from a fresh grading box that received only `/app`,
and in `tmax` from one that received `/app` and `/home/user`.
"""

from __future__ import annotations

import ast
import functools
import json
import logging
from collections import Counter
from pathlib import Path

from verifiers.v1.runtimes import DockerRuntime, Runtime
from verifiers.v1.tasksets.harbor.taskset import HarborTask
from verifiers.v1.trace import Trace

from reliquary_terminal import containers, tmax

logger = logging.getLogger(__name__)

CTRF = "/logs/verifier/ctrf.json"
VERIFIER = ["bash", "/tests/test.sh"]
MAX_CTRF_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_CHARS = 4000
MAX_MESSAGE_CHARS = 1000
# Kept per trace; counts still cover every test.
MAX_TESTS = 1000
MAX_NAME_CHARS = 300
MAX_STATUS_CHARS = 32


def summarize_ctrf(raw: bytes) -> dict:
    """The tests of a CTRF report: name, status, and for any test that did not
    pass its message (truncated), plus counts by status over every test. At
    most MAX_TESTS are listed; `omitted` says how many more there were.
    Raises on anything that is not a CTRF report."""
    doc = json.loads(raw)
    tests = doc["results"]["tests"]
    if not isinstance(tests, list):
        raise ValueError("CTRF `results.tests` is not a list")
    counts: Counter = Counter()
    out = []
    for test in tests:
        status = str(test["status"])[:MAX_STATUS_CHARS]
        counts[status] += 1
        if len(out) >= MAX_TESTS:
            continue
        entry = {"name": str(test["name"])[:MAX_NAME_CHARS], "status": status}
        message = test.get("message") or test.get("trace")
        if status != "passed" and message:
            entry["message"] = str(message)[:MAX_MESSAGE_CHARS]
        out.append(entry)
    return {"counts": dict(counts), "tests": out, "omitted": len(tests) - len(out)}


@functools.lru_cache(maxsize=256)
def collected_tests(task_dir: str | Path) -> frozenset[tuple[str, ...]]:
    """The tests pytest collects by default from `<task_dir>/tests/test_*.py`, read
    statically: `(file, function)` for module-level `test_*` functions and
    `(file, class, method)` for `test_*` methods of `Test*` classes that have no
    `__init__` (pytest skips those). Parametrized tests count once."""
    found = set()
    for path in sorted((Path(task_dir) / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    found.add((path.name, node.name))
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                methods = [m.name for m in node.body
                           if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if "__init__" not in methods:
                    found.update((path.name, node.name, m) for m in methods
                                 if m.startswith("test_"))
    return frozenset(found)


def _ctrf_key(name: str) -> tuple[str, ...]:
    parts = name.split("::")
    return (parts[0].rsplit("/", 1)[-1], *(p.split("[", 1)[0] for p in parts[1:]))


def ctrf_incomplete(raw: bytes | None, expected: frozenset[tuple[str, ...]]) -> str | None:
    """Why a CTRF report cannot back a reward of 1, or None when it can."""
    if raw is None:
        return "no report"
    try:
        tests = json.loads(raw)["results"]["tests"]
        if not isinstance(tests, list):
            raise ValueError("`results.tests` is not a list")
        statuses = [(str(t["name"]), str(t["status"])) for t in tests]
    except Exception as e:  # noqa: BLE001 - any malformed report backs nothing
        return f"unreadable report: {type(e).__name__}"
    failed = sum(status == "failed" for _, status in statuses)
    if failed:
        return f"{failed} failed"
    passed = {_ctrf_key(name) for name, status in statuses if status == "passed"}
    if not passed:
        return "no test passed"
    missing = sorted("::".join(key) for key in expected - passed)
    if missing:
        return f"not passed: {', '.join(missing)}"[:MAX_MESSAGE_CHARS]
    return None


class _Watched:
    """The runtime, recording the result of the run that executes `test.sh`."""

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self.verifier = None

    async def run(self, argv, env):
        result = await self._runtime.run(argv, env)
        if list(argv) == VERIFIER:
            self.verifier = result
        return result

    def __getattr__(self, name):
        return getattr(self._runtime, name)


class TerminalTask(HarborTask):
    """A Harbor task that records its containers (see `containers`) and keeps
    its per-test results (see this module's docstring).

    A `tmax` task also carries a setup bundle (`<task_dir>/setup/`), which
    generates the task's data in every box at start (docs/tmax.md, decisions
    1 and 3). `setup_role` says which box this is: "agent", where the
    bundle deletes the hidden inputs, or "grade", where it stashes the
    protected ones for `test.sh` to put back. `TerminalEnv.finalize` sets
    "grade" on the task it builds for the grading box."""

    setup_role: str = "agent"

    async def setup(self, runtime: Runtime) -> None:
        if isinstance(runtime, DockerRuntime):
            try:
                containers.record(runtime.name)
            except Exception:  # noqa: BLE001 - a missing safety net must not fail the rollout
                logger.warning(
                    "reliquary-terminal: could not record container %s for cleanup",
                    runtime.name, exc_info=True,
                )
        await super().setup(runtime)
        bundle = Path(self.data.task_dir) / "setup" if self.data.task_dir else None
        if bundle is not None and (bundle / "setup.sh").is_file():
            await tmax.run_setup(runtime, bundle, self.setup_role)

    async def _graded(self, runtime: Runtime, trace: Trace) -> float | dict[str, float]:
        grading: dict = {"ctrf": None}
        # A report left from before -- planted in the agent's own box, or in
        # the image -- must not be read as this run's.
        try:
            cleared = await runtime.run(["rm", "-f", CTRF], {})
            stale = None if cleared.exit_code == 0 else cleared.stderr.strip()[-500:] or "rm failed"
        except Exception as e:  # noqa: BLE001 - detail only; grading itself decides below
            stale = f"{type(e).__name__}: {str(e)[-500:]}"
        watched = _Watched(runtime)
        score = await super()._graded(watched, trace)
        raw, read_error = None, None
        if stale is None:
            try:
                raw = await runtime.read(CTRF, max_bytes=MAX_CTRF_BYTES)
            except Exception as e:  # noqa: BLE001 - recorded; a missing report backs nothing
                read_error = f"{type(e).__name__}: {str(e)[-500:]}"
        if self.data.verifier is not None and score != 0:
            if watched.verifier is not None and watched.verifier.exit_code != 0:
                grading["reward_without_success"] = {"reward": score,
                                                     "exit_code": watched.verifier.exit_code}
                score = 0.0
            else:
                why = ctrf_incomplete(raw, collected_tests(self.data.task_dir))
                if why is not None:
                    grading["ctrf_incomplete"] = why
                    score = 0.0
        # Nothing below may change the outcome: the score is already decided.
        try:
            if watched.verifier is not None:
                output = watched.verifier.stdout + watched.verifier.stderr
                grading["exit_code"] = watched.verifier.exit_code
                grading["output_tail"] = output[-MAX_OUTPUT_CHARS:]
            if stale is not None:
                grading["ctrf_error"] = f"could not clear a stale report: {stale}"
            elif read_error is not None:
                grading["ctrf_error"] = read_error
            else:
                try:
                    report = summarize_ctrf(raw)
                except Exception as e:  # noqa: BLE001 - any report problem is recorded, not raised
                    grading["ctrf_error"] = f"{type(e).__name__}: {str(e)[-500:]}"
                else:
                    grading["ctrf"] = report
                    counts = report["counts"]
                    trace.record_metrics(
                        {
                            "tests_total": float(sum(counts.values())),
                            "tests_passed": float(counts.get("passed", 0)),
                            "tests_failed": float(counts.get("failed", 0)),
                        }
                    )
        except Exception as e:  # noqa: BLE001 - see above
            grading = {"ctrf": None, "error": f"{type(e).__name__}: {str(e)[-500:]}",
                       **{key: grading[key] for key in ("reward_without_success",
                                                        "ctrf_incomplete") if key in grading}}
        trace.info["grading"] = grading
        return score

    async def _reward_json(self, runtime: Runtime) -> None:
        """Never read: see this module's docstring."""
        return None


__all__ = ["CTRF", "TerminalTask", "collected_tests", "ctrf_incomplete", "summarize_ctrf"]
