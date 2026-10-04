"""The program's side of the sandbox: one submission, one stdin, one stdout.

Standard library only, and no import from this package: the local runner ships
this file's text into a fresh interpreter, and the core's gVisor worker is
meant to call the same `run` in its warm process, so the two cannot drift.

The import gate and the reduced builtins are rules of the task, not a security
boundary. Containment is the sandbox's job (gVisor in production, a limited
subprocess locally). What this file guarantees is that the rules are the same
everywhere: the same modules, the same `sys`, the same verdict for an exit.
"""

from __future__ import annotations

import builtins
import io
import sys
import threading
import time
import types

ALLOWED_IMPORT_ROOTS = frozenset({
    "abc", "array", "bisect", "collections", "copy", "dataclasses", "decimal",
    "enum", "fractions", "functools", "heapq", "itertools", "math", "operator",
    "random", "re", "statistics", "string", "sys", "threading", "typing",
})
DENIED_BUILTINS = frozenset({
    "breakpoint", "compile", "dir", "eval", "exec", "globals", "help",
    "locals", "open", "vars",
})
GATE_MESSAGE = "is not available in the grader sandbox"


class OutputLimitExceeded(BaseException):
    """Raised from a write past the cap. A BaseException, so that a
    submission's `except Exception` cannot swallow it and keep printing."""


class _CappedBytes(io.BytesIO):
    def __init__(self, cap: int) -> None:
        super().__init__()
        self._cap = cap

    def write(self, data) -> int:
        if self.tell() + len(data) > self._cap:
            raise OutputLimitExceeded()
        return super().write(data)


def _sys_shim(stdin, stdout) -> types.ModuleType:
    shim = types.ModuleType("sys")
    shim.stdin = stdin
    shim.stdout = stdout
    shim.stderr = io.StringIO()
    shim.argv = ["main.py"]
    shim.maxsize = sys.maxsize
    shim.version_info = sys.version_info
    shim.exit = sys.exit
    shim.setrecursionlimit = sys.setrecursionlimit
    shim.getrecursionlimit = sys.getrecursionlimit
    shim.set_int_max_str_digits = sys.set_int_max_str_digits
    shim.get_int_max_str_digits = sys.get_int_max_str_digits
    return shim


def _safe_builtins(shim: types.ModuleType) -> dict:
    real_import = builtins.__import__

    def gated_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name).split(".", 1)[0]
        if level != 0 or root not in ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"module {name!r} {GATE_MESSAGE}")
        if root == "sys":
            return shim
        return real_import(name, globals, locals, fromlist, level)

    safe = {k: v for k, v in builtins.__dict__.items() if k not in DENIED_BUILTINS}
    safe["__import__"] = gated_import
    # `exit` and `quit` come from `site`, which the local child runs without
    # (-S). Defined here so a script ending in `exit()` behaves the same in
    # every host.
    safe["exit"] = safe["quit"] = sys.exit
    return safe


def _join_threads() -> None:
    for thread in threading.enumerate():
        if thread is not threading.main_thread() and not thread.daemon:
            thread.join()


def run(code: str, stdin_text: str, output_cap: int) -> dict:
    """Run `code` as a script on `stdin_text`; return status, stdout, CPU time."""
    raw_out = _CappedBytes(output_cap)
    stdout = io.TextIOWrapper(raw_out, encoding="utf-8", write_through=True)
    stdin = io.TextIOWrapper(io.BytesIO(stdin_text.encode("utf-8")), encoding="utf-8")
    shim = _sys_shim(stdin, stdout)
    namespace = {"__name__": "__main__", "__builtins__": _safe_builtins(shim)}
    saved_stdin, saved_stdout = sys.stdin, sys.stdout
    saved_limit = sys.getrecursionlimit()
    sys.stdin, sys.stdout = stdin, stdout
    status = "ok"
    start = time.process_time()
    try:
        exec(compile(code, "<submission>", "exec"), namespace)
        _join_threads()
    except OutputLimitExceeded:
        status = "output_limit"
    except SystemExit as stop:
        if stop.code not in (None, 0):
            status = "runtime_error"
        else:
            _join_threads()
    except ImportError as error:
        status = "forbidden_import" if GATE_MESSAGE in str(error) else "runtime_error"
    except BaseException:
        status = "runtime_error"
    finally:
        cpu_seconds = time.process_time() - start
        try:
            stdout.flush()
        except OutputLimitExceeded:
            status = "output_limit"
        sys.stdin, sys.stdout = saved_stdin, saved_stdout
        sys.setrecursionlimit(saved_limit)
    text = raw_out.getvalue().decode("utf-8", errors="replace")
    return {
        "status": status,
        "stdout": text if status == "ok" else "",
        "cpu_seconds": cpu_seconds,
    }
