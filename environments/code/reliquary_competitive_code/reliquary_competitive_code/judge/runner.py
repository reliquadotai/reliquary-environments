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
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from reliquary_competitive_code.judge.compare import outputs_match

MEMORY_BYTES = 2 << 30
OUTPUT_CAP_SLACK = 64 * 1024
WALL_FACTOR = 2.0
WALL_SLACK_S = 1.0
READ_SLACK = 1 << 20

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
    __test__ = False  # not a pytest class

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


def _reap(proc: subprocess.Popen) -> tuple[int, float]:
    """Wait for the child ourselves: its exit code and CPU as the kernel counted them."""
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    return proc.returncode, usage.ru_utime + usage.ru_stime


def _killpg(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_test(code: str, stdin: str, *, time_limit_s: float, output_cap: int) -> RunResult:
    """CPU time and exit status come from the kernel (wait4), never from what the
    child prints: the submission shares the child's stdout and could forge it."""
    request = json.dumps({"code": code, "stdin": stdin, "output_cap": output_cap}).encode("utf-8")
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
    chunks: list[bytes] = []
    keep = output_cap + READ_SLACK

    def read_out() -> None:
        kept = 0
        while True:
            block = proc.stdout.read(65536)
            if not block:
                return
            if kept < keep:
                chunks.append(block)
                kept += len(block)

    def write_in() -> None:
        try:
            proc.stdin.write(request)
            proc.stdin.close()
        except (OSError, ValueError):
            pass

    readers = [threading.Thread(target=read_out, daemon=True), threading.Thread(target=write_in, daemon=True)]
    for thread in readers:
        thread.start()
    wall_fired = threading.Event()

    def on_wall_clock() -> None:
        wall_fired.set()
        _killpg(proc)

    timer = threading.Timer(WALL_FACTOR * time_limit_s + WALL_SLACK_S, on_wall_clock)
    timer.start()
    try:
        exit_code, cpu = _reap(proc)
    finally:
        timer.cancel()
        _killpg(proc)  # anything the submission left behind
    for thread in readers:
        thread.join(timeout=5)
    proc.stdout.close()
    if wall_fired.is_set() or exit_code in (-signal.SIGXCPU, -signal.SIGKILL) or cpu > time_limit_s:
        return RunResult("timeout", "", cpu)
    if exit_code != 0:
        return RunResult("runtime_error", "", cpu)
    try:
        line = b"".join(chunks).decode("utf-8", errors="replace").rstrip().rsplit("\n", 1)[-1]
        result = json.loads(line)
        status, text = str(result["status"]), str(result["stdout"])
    except (ValueError, KeyError, TypeError, IndexError):
        return RunResult("runtime_error", "", cpu)
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
