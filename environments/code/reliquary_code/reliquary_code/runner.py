"""Execute model-written Python against pinned test cases.

Reliquary core grades code through a gVisor-sandboxed grader service. A
standalone package cannot assume that service exists, so this ships the
protections that do not need it: a fresh process per case, CPU and address
space limits set in the child before exec, a wall-clock timeout above the CPU
limit, and no inherited environment.

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

import resource
import subprocess
import sys
from collections.abc import Callable
from typing import Any


def _limits(cpu_seconds: int, memory_bytes: int) -> Callable[[], None]:
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))

    return apply


def _run_one(
    source: str,
    case: dict[str, Any],
    *,
    cpu_seconds: int,
    memory_bytes: int,
    wall_seconds: float,
) -> bool:
    """Run `source` once, in its own fresh subprocess, against one case.

    `preexec_fn` runs in the forked child before exec, which is where the
    hard limits below need to land — but it is documented to interact badly
    with multi-threaded parents (the fork only carries the calling thread, so
    a lock held by another thread at fork time deadlocks the child). This
    module spawns one short-lived child per call from what is expected to be
    single-threaded batch/test code, so that hazard does not apply here; a
    `posix_spawn`-based alternative (setting limits via `os.posix_spawn`'s
    file-actions, or a tiny wrapper executable) would avoid the fork
    entirely, but is more machinery than a single-purpose runner needs.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", source],
            input=str(case.get("input", "")),
            capture_output=True,
            text=True,
            timeout=wall_seconds,
            preexec_fn=_limits(cpu_seconds, memory_bytes),
            env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
            cwd="/",
        )
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return False
    if completed.returncode != 0:
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
