"""Execute model-written Python against pinned test cases.

Reliquary core grades code through a gVisor-sandboxed grader service. A
standalone package cannot assume that service exists, so this ships the
protections that do not need it: a fresh process per case, CPU and address
space limits set in the child before exec, a wall-clock timeout above the CPU
limit, no inherited environment, and a process-group kill that reaches
forked descendants, not just the immediate child.

This is weaker than gVisor and does not claim containment. In particular it
does NOT block network access: a plain `subprocess` with a scrubbed
environment can still open sockets, so the child here can reach the network
if the host can. Enforcing `network = false` (the declaration this package's
`environment.toml` makes) is the embedding harness's job — the same way
`network_allow=[]` on `TaskData` is a declaration core's harness enforces,
not something this runner enforces itself. This module is defence in depth
for a package whose job is running someone else's generated code, not a
sandbox boundary.

The one-process-per-case rule is not a style choice. RLIMIT_CPU is cumulative
for the life of a process: a pooled worker accumulates CPU across evaluations
and is eventually SIGKILLed on an innocent later case. That cost 12% of code
submissions in production before it was found.
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
from collections.abc import Callable
from typing import Any


def _limits(cpu_seconds: int, memory_bytes: int) -> Callable[[], None]:
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        # Blocks ANY nonzero-size write, not just large ones: a correct
        # solution that writes a scratch file is scored the same as a wrong
        # one. It also blocks `multiprocessing.Queue`/`Lock`/`Semaphore` —
        # they create a /dev/shm-backed semaphore file and fail with
        # `OSError: [Errno 27] File too large` (EFBIG) at construction time,
        # before the submission's own logic runs at all. Kept anyway and
        # deliberately, not by oversight — these cases are pinned
        # stdin/stdout pairs, so a solution has no legitimate need to touch
        # the filesystem or use cross-process shared memory, and the
        # alternative (letting writes through) reopens the disk-filling
        # attack this closes.
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        # No RLIMIT_NPROC here. It was tried and removed: on Linux it is
        # scoped to the real UID, not to this process's subtree, so on a
        # box where that UID already owns hundreds of processes the child
        # hits EAGAIN on its *first* fork — a legitimate `subprocess`- or
        # `multiprocessing`-using submission is then scored False for
        # reasons that have nothing to do with its own behaviour, and the
        # same source's grade depends on unrelated host load. Fork-bomb
        # containment instead comes from `start_new_session=True` plus
        # `_kill_group` below, which reaches forked descendants directly.

    return apply


def _kill_group(proc: subprocess.Popen[str]) -> tuple[str, str]:
    """SIGKILL the child's entire process group, then reap it.

    `proc` was spawned with `start_new_session=True`, so its process group
    id equals its pid: killing that group reaches processes it forked, not
    only the immediate child. Killing just `proc` (what `Popen.kill()` or
    `subprocess.run`'s own timeout handling does) leaves a forked
    grandchild running past the wall-clock timeout — that gap is exactly
    what `RLIMIT_NPROC` was mistakenly relied on to cover.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        return proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.communicate()


def _spawn(
    source: str,
    stdin: str,
    *,
    cpu_seconds: int,
    memory_bytes: int,
    wall_seconds: float,
) -> subprocess.CompletedProcess[str] | None:
    """Run `source` once in a fresh, isolated subprocess; return the result,
    or `None` if the process could not be started at all.

    `preexec_fn` runs in the forked child before exec, which is where the
    hard limits above need to land — but it is documented to interact badly
    with multi-threaded parents (the fork only carries the calling thread,
    so a lock held by another thread at fork time deadlocks the child). This
    module spawns one short-lived child per call from what is expected to be
    single-threaded batch/test code, so that hazard does not apply here; a
    `posix_spawn`-based alternative (setting limits via `os.posix_spawn`'s
    file-actions, or a tiny wrapper executable) would avoid the fork
    entirely, but is more machinery than a single-purpose runner needs.

    Separated out from `_run_one` so tests can see the actual child a case
    ran in — its pid, or whether processes it forked survive — rather than
    only the pass/fail boolean `run_cases` returns.
    """
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", source],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_limits(cpu_seconds, memory_bytes),
            env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
            cwd="/",
            start_new_session=True,
        )
    except OSError:
        return None
    try:
        stdout, stderr = proc.communicate(input=stdin, timeout=wall_seconds)
    except (subprocess.TimeoutExpired, ValueError):
        stdout, stderr = _kill_group(proc)
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)


def _run_one(
    source: str,
    case: dict[str, Any],
    *,
    cpu_seconds: int,
    memory_bytes: int,
    wall_seconds: float,
) -> bool:
    """Run `source` once, in its own fresh subprocess, against one case."""
    completed = _spawn(
        source,
        str(case.get("input", "")),
        cpu_seconds=cpu_seconds,
        memory_bytes=memory_bytes,
        wall_seconds=wall_seconds,
    )
    if completed is None or completed.returncode != 0:
        return False
    return completed.stdout.strip() == str(case.get("expected_output", "")).strip()


def run_cases(
    source: str,
    cases: list[dict[str, Any]],
    *,
    cpu_seconds: int = 5,
    memory_bytes: int = 512 * 1024 * 1024,
    wall_seconds: float = 10.0,
) -> list[bool]:
    """One boolean per case, in order. One fresh process per case."""
    if not source.strip():
        return [False] * len(cases)
    return [
        _run_one(
            source,
            case,
            cpu_seconds=cpu_seconds,
            memory_bytes=memory_bytes,
            wall_seconds=wall_seconds,
        )
        for case in cases
    ]
