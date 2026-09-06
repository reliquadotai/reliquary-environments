"""Execute model-written Python against pinned function-call cases.

The real OpenCodeInstruct corpus (`structured_cases`) does not carry
stdin/stdout pairs. Every case names an entrypoint and calls it directly,
e.g.::

    {"entry": {"kind": "function", "name": "is_balanced_brackets"},
     "args": ["()"], "kwargs": {}, "expected": true, "compare": "exact"}

A prior version of this module graded `case["input"]` / `case["expected_
output"]` instead — fields the real corpus never sets. Both `.get()` calls
silently returned `""`, so every case fed a child empty stdin and compared
its empty stdout to an empty string: every case passed, regardless of what
the submitted code did. This module now drives `cases.evaluate_call` (see
`cases.py` for provenance) the same way core's trusted grader server does:
call the requested entrypoint with the case's `args`/`kwargs`, then compare
the JSON-safe return value against the case's hidden `expected` value.
Stdin/stdout comparison is not supported — the corpus never uses it, so
keeping it would be an untested, unused code path pretending to be a
feature (see `_evaluate_one` / `run_cases` below for the explicit dispatch
this replaces the old `.get()`-defaulting with).

Reliquary core grades code through a gVisor-sandboxed grader service. A
standalone package cannot assume that service exists, so this ships the
protections that do not need it: a fresh process per case, CPU and address
space limits set in the child before exec, a wall-clock timeout above the
CPU limit, no inherited environment, and a process-group kill that reaches
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
submissions in production before it was found. Core's own grader server
keeps a warm worker pool (bounded by a per-lifetime CPU budget instead) and
re-execs the submission fresh for every case regardless — the same
fresh-namespace-per-case guarantee this module gets by spawning a fresh
process per case instead of a fresh namespace in a shared one.
"""

from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reliquary_code.cases import _outputs_match

# The sandboxed decision logic (`evaluate_call` and its helpers) is read as
# source text, not imported, because each case runs in a *fresh child
# process* started with `-I -S` (isolated mode: no cwd, no site-packages on
# sys.path — see `_spawn`). That child cannot `import reliquary_code`; it
# can only run what is handed to it as the `-c` script body. Splicing the
# module's own source in keeps this module the single source of truth for
# what runs in the child, instead of a second, hand-copied inline string
# drifting out of sync with `cases.py`.
_CASES_SOURCE = Path(__file__).with_name("cases.py").read_text(encoding="utf-8")

# New code, not ported from core: core's worker.py serves many requests off
# one long-lived process (`_serve_stdin`); this runner spawns one process
# per case instead (see module docstring), so it needs a driver that reads
# exactly one request from stdin, calls `evaluate_call` once, and writes
# exactly one JSON response to stdout before exiting.
_WORKER_DRIVER = """

if __name__ == "__main__":
    import json
    import sys

    request = json.loads(sys.stdin.read())
    output, status = evaluate_call(
        request.get("code", ""),
        request.get("entry", {}),
        request.get("args", []),
        request.get("kwargs", {}),
        0.0,
    )
    sys.stdout.write(json.dumps({"output": output, "status": status}))
    sys.stdout.flush()
"""

_WORKER_SCRIPT = _CASES_SOURCE + _WORKER_DRIVER


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
        # deliberately, not by oversight — a case calls one function with
        # in-memory arguments, so a solution has no legitimate need to touch
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

    Separated out from `_evaluate_one` so tests can see the actual child a
    case ran in — its pid, or whether processes it forked survive — rather
    than only the pass/fail boolean `run_cases` returns. Generic on
    purpose: `source` and `stdin` are arbitrary text, so this function
    itself knows nothing about cases, entrypoints, or JSON — only
    `_evaluate_one` below does.
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


def _evaluate_one(
    source: str,
    case: dict[str, Any],
    *,
    cpu_seconds: int,
    memory_bytes: int,
    wall_seconds: float,
) -> tuple[str, Any]:
    """Run one function-call case in its own fresh subprocess.

    Returns `(status, output)`, mirroring `cases.evaluate_call`'s own
    `(output, status)` contract (reordered here so a caller can pattern-
    match on status first). Any failure to get a well-formed response back
    — the process could not start, was killed (wall clock, `RLIMIT_CPU`,
    `RLIMIT_AS`), or wrote something that is not the expected JSON object —
    is folded into `"runtime_error"`, the same status `evaluate_call` itself
    returns for an in-process crash. The caller does not need to
    distinguish "the child never got to reply" from "the child replied that
    it crashed"; both mean the case did not run to a real answer.
    """
    request = {
        "code": source,
        "entry": case.get("entry", {}),
        "args": case.get("args", []),
        "kwargs": case.get("kwargs", {}),
    }
    completed = _spawn(
        _WORKER_SCRIPT,
        json.dumps(request),
        cpu_seconds=cpu_seconds,
        memory_bytes=memory_bytes,
        wall_seconds=wall_seconds,
    )
    if completed is None or completed.returncode != 0:
        return "runtime_error", None
    try:
        response = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError):
        return "runtime_error", None
    if not isinstance(response, dict):
        return "runtime_error", None
    return str(response.get("status", "runtime_error")), response.get("output")


def run_cases(
    source: str,
    cases: list[dict[str, Any]],
    *,
    cpu_seconds: int = 5,
    memory_bytes: int = 512 * 1024 * 1024,
    wall_seconds: float = 10.0,
) -> list[bool]:
    """One boolean per case, in order. One fresh process per case.

    Every case is a function/method call (see module docstring) — there is
    no stdin/stdout case format here, and dispatch is not a `.get()` that
    would silently accept one.

    A case that runs to a real answer is scored by comparing that answer to
    the case's hidden `expected` value (`cases._outputs_match`, ported from
    core's trusted grader server). A case that does not — the submission
    crashed, imported something forbidden, tampered with builtins, or timed
    out / was resource-killed — does not just fail *that* case. It aborts
    the whole list and every case is scored False, discarding any passes
    already recorded. This mirrors core's real `GraderServer._dispatch`: a
    genuine execution failure on any one case zeroes the entire submission's
    reward rather than pro-rating it, because a crash partway through means
    the trusted grader never got to see whether the remaining cases would
    have passed either — crediting them anyway would be inventing a result.
    The sole exception, also ported from core, is `"bad_output"` (the
    function returned something that cannot be turned into a JSON-safe
    primitive): that case alone is scored False and the rest still run,
    because the failure is specific to that one call's return value, not to
    the submission as a whole.
    """
    if not source.strip():
        return [False] * len(cases)
    results: list[bool] = []
    for case in cases:
        status, output = _evaluate_one(
            source,
            case,
            cpu_seconds=cpu_seconds,
            memory_bytes=memory_bytes,
            wall_seconds=wall_seconds,
        )
        if status == "ok":
            results.append(
                _outputs_match(output, case.get("expected"), case.get("compare", "exact"))
            )
            continue
        if status == "bad_output":
            results.append(False)
            continue
        return [False] * len(cases)
    return results
