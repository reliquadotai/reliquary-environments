"""Run a submission on tests, one fresh subprocess per test.

The child receives `guest.py`'s text plus a small driver, sets its own CPU,
address-space and file-size limits as its first statements (never a
`preexec_fn`: forking from a threaded caller deadlocks, see reliquary-code),
runs the submission through `guest.run`, and prints one JSON line. A wall
clock above the CPU limit and a process-group kill cover what sleeps or forks.

Not a sandbox: no network isolation, no filesystem isolation. Production runs
the same `guest.run` inside gVisor; this runner serves development, the build
and qualification.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from reliquary_competitive_code.judge.compare import outputs_match

MEMORY_BYTES = 2 << 30
OUTPUT_CAP_SLACK = 64 * 1024
WALL_FACTOR = 2.0
WALL_SLACK_S = 1.0

_GUEST_SOURCE = Path(__file__).with_name("guest.py").read_text(encoding="utf-8")
_DRIVER = """

if __name__ == "__main__":
    import json as _json
    _request = _json.loads(sys.stdin.buffer.read().decode("utf-8"))
    _result = run(_request["code"], _request["stdin"], _request["output_cap"])
    sys.__stdout__.write("\\n" + _json.dumps(_result) + "\\n")
    sys.__stdout__.flush()
"""


@dataclass(frozen=True, slots=True)
class TestCase:
    stdin: str
    stdout: str


@dataclass(frozen=True, slots=True)
class RunResult:
    status: str
    stdout: str
    cpu_seconds: float


@dataclass(frozen=True, slots=True)
class Verdict:
    passed: bool
    status: str
    tests_run: int
    max_cpu_seconds: float


def output_cap(expected: str) -> int:
    return 4 * len(expected.encode("utf-8")) + OUTPUT_CAP_SLACK


def _program(cpu_seconds: int) -> str:
    return (
        "import resource\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({MEMORY_BYTES}, {MEMORY_BYTES}))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))\n"
        f"exec(compile({_GUEST_SOURCE + _DRIVER!r}, '<guest>', 'exec'), "
        "{'__name__': '__main__'})\n"
    )


def _kill_group(proc: subprocess.Popen) -> tuple[bytes, bytes]:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        return proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.communicate()


def run_test(code: str, stdin: str, *, time_limit_s: float, output_cap: int) -> RunResult:
    request = json.dumps({"code": code, "stdin": stdin, "output_cap": output_cap})
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", _program(math.ceil(time_limit_s) + 1)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
            cwd="/",
            start_new_session=True,
        )
    except OSError:
        return RunResult("runtime_error", "", 0.0)
    timed_out = False
    try:
        out, _ = proc.communicate(
            input=request.encode("utf-8"),
            timeout=WALL_FACTOR * time_limit_s + WALL_SLACK_S,
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        out, _ = _kill_group(proc)
    if timed_out or proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
        return RunResult("timeout", "", time_limit_s)
    try:
        result = json.loads(out.decode("utf-8", errors="replace").rstrip().rsplit("\n", 1)[-1])
        status, text, cpu = str(result["status"]), str(result["stdout"]), float(result["cpu_seconds"])
    except (ValueError, KeyError, TypeError, IndexError):
        return RunResult("runtime_error", "", 0.0)
    if cpu > time_limit_s:
        return RunResult("timeout", "", cpu)
    return RunResult(status, text, cpu)


def judge(code: str | None, tests: Sequence[TestCase], *, time_limit_s: float) -> Verdict:
    """Binary verdict over `tests`, stopping at the first failure."""
    if code is None:
        return Verdict(False, "no_code", 0, 0.0)
    slowest = 0.0
    for count, test in enumerate(tests, start=1):
        result = run_test(code, test.stdin, time_limit_s=time_limit_s, output_cap=output_cap(test.stdout))
        slowest = max(slowest, result.cpu_seconds)
        if result.status != "ok":
            return Verdict(False, result.status, count, slowest)
        if not outputs_match(test.stdout, result.stdout):
            return Verdict(False, "wrong_answer", count, slowest)
    return Verdict(True, "ok", len(tests), slowest)
