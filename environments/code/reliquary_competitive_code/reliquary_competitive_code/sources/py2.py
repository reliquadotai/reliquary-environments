"""Python 2 references made runnable: 2to3 without `fix_input`, plus a literal
reader for Python 2's evaluating `input()`.

Stock 2to3 turns `input()` into `eval(input())`, and the judge denies `eval`;
measured on 100 v1 rejects, stock 2to3 curates 52, this conversion 82. The
helper reads ints, floats, quoted strings and comma tuples, what Python 2
programs feed `input()` in practice. Integer `/` is not rewritten (2to3 cannot
know the operand types): such a program fails its tests and is not kept.

`lib2to3` ships with Python up to 3.12, the version this package requires.
"""

from __future__ import annotations

import ast
import re
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from lib2to3.refactor import RefactoringTool, get_fixers_from_package

_TOOL: RefactoringTool | None = None
_HELPER = '''def _py2_input(prompt=""):
    s = input().strip()
    def one(t):
        t = t.strip()
        for cast in (int, float):
            try:
                return cast(t)
            except ValueError:
                pass
        if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\\"":
            return t[1:-1]
        return t
    if "," in s:
        parts = [p for p in s.strip("()[]").split(",") if p.strip()]
        return tuple(one(p) for p in parts)
    return one(s)
'''
_INPUT = re.compile(r"(?<![\w.])input\s*\(")
_PY2_ONLY = re.compile(
    r"\b(raw_input|xrange|iteritems|itervalues|iterkeys|has_key|unichr|basestring)\b"
    r"|^\s*print\s+[^\s(=]|\bunicode\(|\blong\(|sys\.maxint",
    re.MULTILINE,
)


def looks_python2(code: str) -> bool:
    """Python 3 cannot parse it, or it names what only Python 2 has."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Python 2 literals warn ("invalid decimal literal")
            ast.parse(code)
    except (SyntaxError, ValueError):
        return True
    return bool(_PY2_ONLY.search(code))


def _tool() -> RefactoringTool:
    global _TOOL
    if _TOOL is None:
        fixers = [f for f in get_fixers_from_package("lib2to3.fixes") if not f.endswith(".fix_input")]
        _TOOL = RefactoringTool(fixers)
    return _TOOL


def convert(code: str) -> str | None:
    """The Python 3 version of `code`, or None when 2to3 cannot parse it."""
    uses_input = bool(_INPUT.search(code))
    source = _INPUT.sub("_py2_input(", code) if uses_input else code
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = str(_tool().refactor_string(source.rstrip() + "\n", "<reference>"))
    except Exception:  # lib2to3 raises its own parse errors, and RecursionError
        return None
    if uses_input:
        lines = out.splitlines(keepends=True)
        k = 0
        while k < len(lines) and lines[k].startswith("from __future__"):
            k += 1
        out = "".join(lines[:k]) + _HELPER + "".join(lines[k:])
    return out


def upgrade(references: tuple[str, ...]) -> tuple[tuple[str, ...], frozenset[str]]:
    """Each Python 2 reference converted, in the same order, and the set of
    those converted. A Python 3 reference is kept as it is (2to3 would rewrite
    `print(x)` as `print((x))`); one 2to3 cannot parse is kept and fails."""
    out, converted = [], set()
    for code in references:
        new = convert(code) if looks_python2(code) else None
        if new is not None and new.strip() != code.strip():
            out.append(new)
            converted.add(new)
        else:
            out.append(code)
    return tuple(dict.fromkeys(out)), frozenset(converted)
