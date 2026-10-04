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
what used to be thrown away.
"""

from __future__ import annotations

import json
from collections import Counter

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import DockerRuntime, Runtime
from verifiers.v1.tasksets.harbor.taskset import HarborTask
from verifiers.v1.trace import Trace

from reliquary_terminal import containers

CTRF = "/logs/verifier/ctrf.json"
VERIFIER = ["bash", "/tests/test.sh"]
MAX_CTRF_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_CHARS = 4000
MAX_MESSAGE_CHARS = 1000


def summarize_ctrf(raw: bytes) -> dict:
    """The tests of a CTRF report: name, status, and for any test that did not
    pass its message (truncated), plus counts by status. Raises ValueError on
    anything that is not a CTRF report."""
    doc = json.loads(raw)
    tests = doc["results"]["tests"]
    if not isinstance(tests, list):
        raise ValueError("CTRF `results.tests` is not a list")
    out = []
    for test in tests:
        entry = {"name": str(test["name"]), "status": str(test["status"])}
        message = test.get("message") or test.get("trace")
        if entry["status"] != "passed" and message:
            entry["message"] = str(message)[:MAX_MESSAGE_CHARS]
        out.append(entry)
    return {"counts": dict(Counter(t["status"] for t in out)), "tests": out}


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
            containers.record(runtime.name)
        await super().setup(runtime)

    async def _graded(self, runtime: Runtime, trace: Trace) -> float | dict[str, float]:
        # A report left from before -- planted in the agent's own box, or in
        # the image -- must not be read as this run's.
        cleared = await runtime.run(["rm", "-f", CTRF], {})
        watched = _Watched(runtime)
        score = await super()._graded(watched, trace)
        grading: dict = {}
        if watched.verifier is not None:
            output = watched.verifier.stdout + watched.verifier.stderr
            grading["exit_code"] = watched.verifier.exit_code
            grading["output_tail"] = output[-MAX_OUTPUT_CHARS:]
        grading["ctrf"] = None
        if cleared.exit_code != 0:
            grading["ctrf_error"] = f"could not clear a stale report: {cleared.stderr.strip()[-500:]}"
        else:
            try:
                report = summarize_ctrf(await runtime.read(CTRF, max_bytes=MAX_CTRF_BYTES))
            except (SandboxError, OSError, ValueError, KeyError, TypeError) as e:
                grading["ctrf_error"] = f"{type(e).__name__}: {str(e)[-500:]}"
            else:
                grading["ctrf"] = report
                counts = report["counts"]
                trace.record_metrics(
                    {
                        "tests_total": float(len(report["tests"])),
                        "tests_passed": float(counts.get("passed", 0)),
                        "tests_failed": float(counts.get("failed", 0)),
                    }
                )
        trace.info["grading"] = grading
        return score


__all__ = ["CTRF", "TerminalTask", "summarize_ctrf"]
