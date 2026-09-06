"""Deciding whether a completion answers a function-call case, copied from core.

Reliquary core grades OpenCodeInstruct by calling a named function (or
method) inside a gVisor-sandboxed worker and comparing its return value
against a hidden expected value — it does NOT compare stdin/stdout text.
This module ports the part of that decision that lives inside the worker:
which callable in the submitted source is the graded entry point, what
builtins it may see, and how its return value is turned into a JSON-safe
primitive. Copied rather than reimplemented on purpose: these functions
carry accumulated fixes (entry-point resolution by structure when the
requested name is absent, never selecting an imported callable, never
selecting a function that returns nothing) that a fresh implementation
would rediscover one production incident at a time. Keep the first section
below diffable against `reliquary/environment/grader/worker.py` at core
commit 10c2a4d9.

Not ported: `_serve_stdin` / `main`, worker.py's long-lived
JSON-lines-over-stdin loop that lets one warm worker process serve many
requests from the trusted grader server. This package's runner spawns one
fresh subprocess per case instead (see `runner.py`), so there is no
multi-request loop to keep warm — the driver that calls `evaluate_call`
once per subprocess lives in `runner.py`, not here.

The second section is NOT from worker.py. `evaluate_call` returns a raw
`(output, status)` pair; deciding whether that counts as a pass still
needs the hidden `expected` value and the `compare` mode, and that
comparison is deliberately kept out of the sandboxed worker in core too —
it lives in the trusted `GraderServer` in `reliquary/environment/grader/
server.py`, as classmethods on `GraderServer`. `_json_equal` is ported
from there (as a plain function; the `cls` parameter dropped) for the same
reason as the first section: a naive `output == expected` mishandles
`True == 1`, mismatched int/float representations of the same number, and
dict key order, all of which core's real grader already accounts for.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import faulthandler
import inspect
import io
import math
from typing import Any

# Armed before anything redirects sys.stderr, so a hard death (SIGSEGV,
# SIGABRT) still writes a Python traceback to the real fd 2, which the
# parent process (here: the runner's per-case subprocess) can capture.
# Without it such a worker dies silently and the crash is unattributable.
faulthandler.enable()


_CRITICAL_BUILTINS = {
    name: getattr(builtins, name)
    for name in ("__import__", "compile", "eval", "exec", "open", "input")
}

_ALLOWED_IMPORT_ROOTS = {
    "abc", "array", "bisect", "collections", "copy", "dataclasses", "decimal",
    "enum", "functools", "heapq", "itertools", "math", "operator", "re",
    "statistics", "string", "typing",
}

_DENIED_BUILTINS = {
    "breakpoint", "compile", "dir", "eval", "exec", "globals", "help", "input",
    "locals", "open", "vars",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".", 1)[0]
    if level != 0 or root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"module {name!r} is not available in the grader sandbox")
    return _CRITICAL_BUILTINS["__import__"](name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    safe = {
        name: value
        for name, value in builtins.__dict__.items()
        if name not in _DENIED_BUILTINS
    }
    safe["__import__"] = _safe_import
    return safe


def _critical_builtins_intact() -> bool:
    return all(
        getattr(builtins, name) is original
        for name, original in _CRITICAL_BUILTINS.items()
    )


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe primitive, or raise TypeError.

    This intentionally rejects arbitrary objects so custom ``__eq__`` /
    comparator tricks never reach trusted scoring.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError("dict key is not a string")
            out[k] = _json_safe(v)
        return out
    raise TypeError(f"unsupported output type: {type(value).__name__}")


def _user_defined_names(code: str) -> set[str]:
    """Top-level def/class names in the submitted source.

    Used to resolve the entry point by structure when the requested name is
    absent — and to never select an imported callable as the entry point.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _accepts_arity(fn: Any, nargs: int) -> bool:
    """True if *fn* can be called with *nargs* positional arguments."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return True
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required = sum(1 for p in positional if p.default is p.empty)
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
    upper = float("inf") if has_varargs else len(positional)
    return required <= nargs <= upper


def _defined_functions_in_order(code: str) -> list[str]:
    """Top-level function names, in source order."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _call_graph_roots(code: str, fn_names: set[str]) -> set[str]:
    """Function names not called from inside a *different* top-level function.

    These are the call-graph roots — the entry point of a "main + helpers"
    solution. Self-recursion does not disqualify a root.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(fn_names)
    called_by_others: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in fn_names
                and sub.func.id != node.name
            ):
                called_by_others.add(sub.func.id)
    return set(fn_names) - called_by_others


def _returns_a_value(code: str, name: str) -> bool:
    """True if top-level function *name* has a ``return <expr>`` (not bare/None).

    A print-only or None-returning function can never match a return-value case,
    so it is never the graded entry. Unparseable code is treated as returning a
    value so analysis failure never excludes a real solution.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            visitor = _ValueReturnVisitor()
            for stmt in node.body:
                visitor.visit(stmt)
            return visitor.has_value_return
    return True


class _ValueReturnVisitor(ast.NodeVisitor):
    """Find value returns without descending into nested definitions."""

    def __init__(self) -> None:
        self.has_value_return = False

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and not (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ):
            self.has_value_return = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None


def _resolve_function(ns: dict[str, Any], code: str, nargs: int) -> Any | None:
    """Resolve the entry function when the requested name is absent.

    The prompt asks for a behavior, not a name, so pick a single entry
    deterministically: the only arity match; else the only call-graph root
    (a function no *other* top-level function calls); else the last-defined
    arity match. Exactly one function is then run against the hidden cases —
    never several with "accept any pass" — so a wrong pick simply fails them.
    """
    order = [
        name for name in _defined_functions_in_order(code)
        if callable(ns.get(name)) and not isinstance(ns.get(name), type)
    ]
    candidates = [name for name in order if _accepts_arity(ns[name], nargs)]
    if not candidates:
        return None
    # Drop print-only / None-returning helpers when a value-returning one exists:
    # they can never match a return-value case, so they are never the entry.
    valued = [name for name in candidates if _returns_a_value(code, name)]
    if valued:
        candidates = valued
    if len(candidates) == 1:
        return ns[candidates[0]]
    roots = _call_graph_roots(code, set(order))
    root_candidates = [name for name in candidates if name in roots]
    if len(root_candidates) == 1:
        return ns[root_candidates[0]]
    pool = root_candidates or candidates
    return ns[pool[-1]]


def _resolve_class(ns: dict[str, Any], defined: set[str]) -> Any | None:
    """The submitted code's sole class, or None if ambiguous."""
    classes = [ns[name] for name in defined if isinstance(ns.get(name), type)]
    return classes[0] if len(classes) == 1 else None


def evaluate_call(
    code: str,
    entry: dict[str, Any],
    args: list[Any],
    kwargs: dict[str, Any],
    timeout_s: float,
) -> tuple[Any | None, str]:
    """Execute miner code and call the requested entrypoint.

    Returns ``(output, status)``. The server enforces wall-clock timeouts;
    ``timeout_s`` is accepted for protocol symmetry.
    """
    del timeout_s
    if not code or not code.strip():
        return None, "runtime_error"
    if not isinstance(entry, dict):
        return None, "bad_entry"
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        return None, "bad_request"

    ns: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "__name__": "<miner_code>",
    }
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            exec(compile(code, "<miner_code>", "exec"), ns)
        except ImportError as e:
            if "not available in the grader sandbox" in str(e):
                return None, "forbidden_import"
            return None, "runtime_error"
        except BaseException:
            return None, "runtime_error"
        if not _critical_builtins_intact():
            return None, "tampered"

        try:
            kind = entry.get("kind")
            if kind == "function":
                # Prompt specifies a behavior, not a name: accept the requested
                # name, else resolve the sole/only-arity-matching defined function.
                fn = ns.get(entry["name"])
                if not callable(fn):
                    fn = _resolve_function(ns, code, len(args))
                if not callable(fn):
                    return None, "runtime_error"
            elif kind == "method":
                cls = ns.get(entry["class_name"])
                if not isinstance(cls, type):
                    cls = _resolve_class(ns, _user_defined_names(code))
                if cls is None:
                    return None, "runtime_error"
                fn = getattr(cls(), entry["method"])
            else:
                return None, "bad_entry"
            output = fn(*args, **kwargs)
            if not _critical_builtins_intact():
                return None, "tampered"
            return _json_safe(output), "ok"
        except ImportError as e:
            if "not available in the grader sandbox" in str(e):
                return None, "forbidden_import"
            return None, "runtime_error"
        except TypeError as e:
            if "unsupported output type" in str(e) or "dict key" in str(e) or "non-finite" in str(e):
                return None, "bad_output"
            return None, "runtime_error"
        except BaseException:
            return None, "runtime_error"


# ---------------------------------------------------------------------------
# Trusted-side comparison, ported from `GraderServer` in
# `reliquary/environment/grader/server.py` at the same core commit — NOT
# from worker.py (see module docstring). `evaluate_call` above only returns
# what the submitted code produced; turning that into a pass/fail verdict
# against the case's hidden `expected` value is core's server's job, kept
# separate from the sandboxed worker on purpose. Ported as plain functions
# (the `cls` classmethod parameter dropped) since `GraderServer` itself is
# not a self-contained thing this package can import.
# ---------------------------------------------------------------------------


def _outputs_match(output: Any, expected: Any, compare: str) -> bool:
    if compare != "exact":
        return False
    return _json_equal(output, expected)


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) or isinstance(right, float):
            return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
        return left == right
    if left is None or right is None or isinstance(left, str) or isinstance(right, str):
        return type(left) is type(right) and left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return (
            set(left.keys()) == set(right.keys())
            and all(_json_equal(left[k], right[k]) for k in left)
        )
    return False
