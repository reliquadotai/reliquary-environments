"""Per-test results next to the binary reward.

Every task of both splits grades with pytest through `pytest-json-ctrf`
(`--ctrf /logs/verifier/ctrf.json`, in all 89 Terminal-Bench 2.1 `test.sh`
and all 64 MiMo ones), then writes 1 or 0 to `reward.txt`. verifiers reads
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

**Untrusted in the `eval` split.** There `test.sh` runs in the box the agent
worked in, as Terminal-Bench ships it, and anything the agent left running
there can write `/logs/verifier/ctrf.json` or the output this records. The
per-test data is fit for debugging there, never for partial credit. In the
`train` split it comes from a fresh grading box that received only `/app`.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from verifiers.v1.runtimes import DockerRuntime, Runtime
from verifiers.v1.tasksets.harbor.taskset import HarborTask
from verifiers.v1.trace import Trace

from reliquary_terminal import containers

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
    its per-test results (see this module's docstring)."""

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
        # Nothing below may change the outcome: the score is already decided.
        try:
            if watched.verifier is not None:
                output = watched.verifier.stdout + watched.verifier.stderr
                grading["exit_code"] = watched.verifier.exit_code
                grading["output_tail"] = output[-MAX_OUTPUT_CHARS:]
            if stale is not None:
                grading["ctrf_error"] = f"could not clear a stale report: {stale}"
            else:
                try:
                    report = summarize_ctrf(await runtime.read(CTRF, max_bytes=MAX_CTRF_BYTES))
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
            grading = {"ctrf": None, "error": f"{type(e).__name__}: {str(e)[-500:]}"}
        trace.info["grading"] = grading
        return score


__all__ = ["CTRF", "TerminalTask", "summarize_ctrf"]
