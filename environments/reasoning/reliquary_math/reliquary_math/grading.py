"""Answer equality for OpenMathInstruct-2, copied from Reliquary core.

Copied rather than reimplemented on purpose: these functions carry fixes for
surface-form false negatives that took a measured 28% down to 0.9%, and a
fresh implementation would rediscover every one of them. Keep them
diffable against
`reliquary/environment/openmathinstruct.py` at core commit 10c2a4d9.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional

# ---------------------------------------------------------------------------
# Protocol-derived constants, pinned.
#
# Core computes these two flags from a live, validator-side protocol profile
# (`reliquary.constants.RAW_COMPLETION_PROMPTS` / `MATH_ANSWER_FORMAT`,
# themselves read off `ACTIVE_PROTOCOL_PROFILE`). This package has no
# equivalent of that profile and must not import `reliquary.*`, so both are
# pinned here to the current v4 default: chat-template prompts (not raw
# completions) and boxed-only answers (no legacy trailing-number fallback).
# If core's live default ever changes, this package will not follow it
# automatically — see task-5-report.md.
# ---------------------------------------------------------------------------

_RAW_COMPLETION_PROMPTS = False
_MATH_ANSWER_FORMAT = "boxed"

# ---------------------------------------------------------------------------
# Answer extraction — reuse the balanced-brace parser from MATH env
# ---------------------------------------------------------------------------

def _last_boxed_only_string(text: str) -> Optional[str]:
    """Return the last \\boxed{...} / \\fbox{...} substring or None.

    Balanced-brace walk to handle nested expressions correctly.
    """
    idx = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if idx < 0:
        return None
    try:
        open_idx = text.index("{", idx)
    except ValueError:
        return None
    depth = 0
    for j in range(open_idx, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[idx : j + 1]
    return None


def _strip_boxed_wrapper(s: str) -> str:
    for prefix in (r"\boxed{", r"\fbox{"):
        if s.startswith(prefix) and s.endswith("}"):
            return s[len(prefix) : -1]
    return s


# ---------------------------------------------------------------------------
# Answer normalization — handles both LaTeX and plain numeric forms.
# OpenMathInstruct expected_answer is typically plain (e.g. "45", "3/4",
# "-2", "\\frac{1}{2}") so we accept both shapes.
# ---------------------------------------------------------------------------

_TEXT_RE = re.compile(r"\\text\{([^}]*)\}")
_MBOX_RE = re.compile(r"\\mbox\{([^}]*)\}")


def _normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # v4 raw-completion answers come wrapped in inline/display LaTeX ("\(x\)" /
    # "\[x\]"); the delimiters never carry meaning. Gated so v3 grading (which
    # never saw these — the chat-template model boxes) stays byte-identical.
    # Core reads this from a live, validator-side protocol profile
    # (`reliquary.constants.RAW_COMPLETION_PROMPTS`); this package has no
    # such profile and keeps no dependency on core, so it is pinned to the
    # v4 default (chat-template prompts) as a module-level constant below.
    if _RAW_COMPLETION_PROMPTS:
        for delim in (r"\(", r"\)", r"\[", r"\]"):
            s = s.replace(delim, "")
    # Drop LaTeX spacing macros first
    for macro in (r"\!", r"\,", r"\ ", r"\;", r"\:"):
        s = s.replace(macro, "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    # Strip unit decorations so a value with units matches the bare value
    # ("119^\circ" == "119"). These never carry numeric meaning for grading.
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "", s)
    s = (s.replace(r"\circ", "").replace(r"\degree", "")
          .replace(r"\%", "").replace("%", ""))
    s = _TEXT_RE.sub(r"\1", s)
    s = _MBOX_RE.sub(r"\1", s)
    s = s.replace(r"\$", "").replace("$", "")
    # Plain-number normalization (OMI specific): "3.0" -> "3", "+5" -> "5"
    s = s.strip().rstrip(".").strip()
    s = re.sub(r"\s+", "", s)
    # Strip leading + on integers
    if len(s) > 1 and s[0] == "+":
        s = s[1:]
    # "3.0" -> "3" only when there's nothing else
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".", 1)[0]
    return s


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:/\d+)?")


def _as_number(s: str) -> Optional[Fraction]:
    """Parse a plain integer / decimal / simple fraction as an exact Fraction.

    Returns None for anything else (LaTeX expressions, words, empty), so the
    caller can fall back to string comparison. Exact rationals avoid float
    rounding, so "82.50" == "82.5" and "1/2" == "0.5".
    """
    if not _NUMERIC_RE.fullmatch(s):
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _latex_to_pyexpr(s: str) -> Optional[str]:
    """Rewrite the common LaTeX math macros (\\frac, \\sqrt, \\cdot, ...) into a
    plain Python/sympy expression. Returns None if any unknown ``\\macro``
    survives, so the caller falls back to string comparison rather than guess.
    """
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\sqrt\[([^\[\]{}]+)\]\{([^{}]+)\}", r"((\2)**(1.0/(\1)))", s)
        s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt((\1))", s)
        s = re.sub(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = (s.replace(r"\cdot", "*").replace(r"\times", "*")
          .replace(r"\div", "/").replace(r"\pi", "pi"))
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    return None if "\\" in s else s


def _expr_is_safe(expr, _depth: int = 0, allow_symbols: bool = False) -> bool:
    """Structural whitelist guarding ``evalf`` against adversarial payloads.

    Only numbers, named constants (pi, E), ``+ - * /`` and powers with a *small
    numeric* exponent (which also covers roots and ``1/x``) are allowed. Power
    towers (nested/symbolic exponents), huge exponents, factorials, other
    functions and free symbols are rejected — anything whose ``evalf`` could
    blow up on a miner-controlled boxed answer. When ``allow_symbols`` is set,
    free symbols are also treated as safe (used by the symbolic-equality path,
    which restricts inputs to single-letter variables via ``_expr_str_is_safe``
    first).
    """
    import sympy

    if _depth > 40:
        return False
    if expr.is_Number or isinstance(expr, sympy.NumberSymbol):
        return True
    if allow_symbols and expr.is_Symbol:
        return True
    if isinstance(expr, (sympy.Add, sympy.Mul)):
        return all(_expr_is_safe(a, _depth + 1, allow_symbols) for a in expr.args)
    if isinstance(expr, sympy.Pow):
        base, exp = expr.as_base_exp()
        if not exp.is_Number:
            return False  # nested / symbolic exponent (e.g. a power tower)
        try:
            if abs(float(exp)) > 10:
                return False  # huge exponent
        except (TypeError, ValueError, OverflowError):
            return False
        return _expr_is_safe(base, _depth + 1, allow_symbols)
    return False  # factorial, other functions, disallowed symbols, ...


def _expr_str_is_safe(s: str, allow_symbols: bool = False) -> bool:
    """Token whitelist applied BEFORE ``parse_expr``.

    ``parse_expr`` eagerly evaluates some constructs during parsing (notably
    ``n!`` -> ``factorial(n)``), so a huge factorial would hang before the
    structural guard ever runs. Allow only digits, ``+ - * / ( ) .``, spaces and
    the tokens ``sqrt`` / ``pi``; anything else (``!``, function names like
    ``exp``/``gamma``/``factorial``, stray symbols) is rejected up front. With
    ``allow_symbols`` set, isolated single-letter variables are also allowed
    (multi-letter identifiers, function names and calls still rejected).
    """
    # Strip the allowed alpha tokens even when digit-adjacent ("2sqrt(5)") but
    # not when part of a longer identifier ("sqrtx"); whatever survives must be
    # numbers / operators only.
    cleaned = re.sub(r"(?<![A-Za-z])(?:sqrt|pi)(?![A-Za-z])", "", s)
    if allow_symbols:
        # drop lone single-letter variables (not part of a longer identifier and
        # not a function call `f(`), leaving numbers / operators for the fullmatch.
        cleaned = re.sub(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z(])", "", cleaned)
    return re.fullmatch(r"[0-9.+\-*/() \t]*", cleaned) is not None


def _latex_value_equal(candidate: str, gt: str) -> bool:
    """Numeric value-equality for LaTeX answers, antlr-free and bounded.

    The candidate is a miner-controlled boxed payload, so two whitelists bound
    compute before any heavy evaluation: ``_expr_str_is_safe`` rejects tokens
    ``parse_expr`` would eager-evaluate (factorials, function names), and
    ``_expr_is_safe`` rejects power towers / huge exponents before ``evalf``.
    The length cap alone does not bound compute (``9^9^9^9`` is 7 chars but
    astronomically large). Returns False on any parse/eval error or disallowed
    structure.
    """
    if not candidate or not gt or len(candidate) > 100 or len(gt) > 100:
        return False
    ec, eg = _latex_to_pyexpr(candidate), _latex_to_pyexpr(gt)
    if ec is None or eg is None:
        return False
    if not (_expr_str_is_safe(ec) and _expr_str_is_safe(eg)):
        return False  # factorial / function name parse_expr would eager-evaluate
    try:
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
        tr = standard_transformations + (implicit_multiplication_application,)
        xc = parse_expr(ec, transformations=tr, evaluate=False)
        xg = parse_expr(eg, transformations=tr, evaluate=False)
        if not (_expr_is_safe(xc) and _expr_is_safe(xg)):
            return False
        vc = complex(xc.evalf())
        vg = complex(xg.evalf())
    except Exception:
        return False
    return abs(vc - vg) <= 1e-9 * (1 + abs(vg))


def _expand_term_bound(expr, cap: int) -> int:
    """Cheap upper bound on the monomial count ``sympy.expand(expr)`` would
    produce, short-circuiting at ``cap + 1``. Walks the parsed tree WITHOUT
    expanding, so it bounds ``expand``'s combinatorial cost — which the exponent
    cap in ``_expr_is_safe`` does not: ``(a+b+...+n)**10`` stays within the char
    and exponent limits yet expands to ``C(terms+9, 10)`` monomials. Deterministic.
    """
    import sympy
    from math import comb

    if expr.is_Atom:
        return 1
    if isinstance(expr, sympy.Add):
        total = 0
        for arg in expr.args:
            total += _expand_term_bound(arg, cap)
            if total > cap:
                return cap + 1
        return total
    if isinstance(expr, sympy.Mul):
        prod = 1
        for arg in expr.args:
            prod *= _expand_term_bound(arg, cap)
            if prod > cap:
                return cap + 1
        return prod
    if isinstance(expr, sympy.Pow):
        base, exp = expr.as_base_exp()
        if not exp.is_Number:
            return 1
        try:
            f = float(exp)
        except (TypeError, ValueError, OverflowError):
            return 1
        if f < 0 or f != int(f):
            return 1  # only positive integer-VALUED exponents expand multinomially
        k = int(f)
        b = _expand_term_bound(base, cap)
        if b <= 1:
            return 1
        terms = comb(b + k - 1, k)
        return terms if terms <= cap else cap + 1
    return 1


def _latex_symbolic_equal(candidate: str, gt: str) -> bool:
    """True iff candidate and gt both parse to ALGEBRAIC expressions (each with
    at least one free symbol) whose expanded difference is identically zero.

    Closes algebraic reorderings the numeric path misses (2-b == -b+2,
    (x+1)^2 == x^2+2x+1). Purely-numeric answers have no free symbols and never
    enter here, so the numeric 1e-9 path still governs numbers and no rounding
    false-positive can leak in. Bounded and deterministic; never raises.
    """
    if not candidate or not gt or len(candidate) > 100 or len(gt) > 100:
        return False
    ec, eg = _latex_to_pyexpr(candidate), _latex_to_pyexpr(gt)
    if ec is None or eg is None:
        return False
    if not (_expr_str_is_safe(ec, allow_symbols=True)
            and _expr_str_is_safe(eg, allow_symbols=True)):
        return False
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
        tr = standard_transformations + (implicit_multiplication_application,)
        xc = parse_expr(ec, transformations=tr, evaluate=False)
        xg = parse_expr(eg, transformations=tr, evaluate=False)
        if not (xc.free_symbols and xg.free_symbols):
            return False  # numbers stay on the numeric path
        if not (_expr_is_safe(xc, allow_symbols=True)
                and _expr_is_safe(xg, allow_symbols=True)):
            return False
        cap = 2000
        if _expand_term_bound(xc, cap) > cap or _expand_term_bound(xg, cap) > cap:
            return False
        return sympy.expand(xc - xg).is_zero is True
    except Exception:
        return False


def _split_structure(s: str) -> Optional[tuple[str, list[str]]]:
    """Return ``(signature, ordered elements)`` for a matrix/vector/tuple, else None.

    The signature includes matrix dimensions or the opening delimiter
    (``(`` / ``[``). Two answers only compare element-wise when their signatures
    match, so a row vector never matches a column vector and an open ``(a,b)``
    interval/point never matches a closed ``[a,b]`` one. Scalars return None.
    """
    m = re.search(r"\\begin\{[bp]?matrix\}(.*?)\\end\{[bp]?matrix\}", s, re.S)
    if m:
        rows = [
            [c.strip() for c in row.split("&") if c.strip()]
            for row in m.group(1).split(r"\\")
        ]
        rows = [row for row in rows if row]
        if not rows:
            return None
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            return None
        return f"matrix:{len(rows)}x{width}", [c for row in rows for c in row]
    if len(s) >= 2 and s[0] in "([" and s[-1] in ")]" and "," in s:
        return s[0], [x.strip() for x in s[1:-1].split(",")]
    return None


def _answers_equal(candidate: str, gt: str) -> bool:
    """Compare by value when both sides are numbers, else by normalized string.

    The prompt asks for a value, so any surface form of that value is correct
    (trailing zeros, fraction vs decimal, \\frac{5721}{5} == 1144.2). Same-shape
    structured answers (matrix/vector/tuple) compare element-wise by value;
    scalars fall back to LaTeX value-equality, algebraic equivalence, then exact
    normalized-string match.
    """
    cs, gs = _split_structure(candidate), _split_structure(gt)
    if cs is not None and gs is not None and cs[0] == gs[0]:
        return len(cs[1]) == len(gs[1]) and all(
            _answers_equal(_normalize_answer(a), _normalize_answer(b))
            for a, b in zip(cs[1], gs[1])
        )
    c, g = _as_number(candidate), _as_number(gt)
    if c is not None and g is not None:
        return c == g
    if candidate == gt:
        return True
    if _latex_value_equal(candidate, gt):
        return True
    return _latex_symbolic_equal(candidate, gt)


def _compute_omi_reward(problem: dict, completion: str) -> float:
    """Score an OMI completion.

    1.0 if the last \\boxed{...} matches expected_answer by value (numbers) or
    by normalized string (expressions). 0.0 otherwise. Never raises.
    """
    try:
        boxed = _last_boxed_only_string(completion)
        if boxed is not None:
            candidate = _normalize_answer(_strip_boxed_wrapper(boxed))
        else:
            # v4 requires the format requested by the canonical prompt. The
            # boxed span is the only reward-bearing region the validator's
            # answer-integrity proof can authenticate, so every unboxed form
            # scores zero. Older profiles retain their paid trailing-number
            # behavior byte-for-byte through the explicit profile contract.
            # Core reads this from the live protocol profile
            # (`reliquary.constants.MATH_ANSWER_FORMAT`); pinned here to the
            # v4 default for the same reason as `_RAW_COMPLETION_PROMPTS`
            # above.
            if _MATH_ANSWER_FORMAT == "boxed":
                return 0.0
            # Legacy fallback: trailing number / fraction at end of text.
            tail = completion.strip().split("\n")[-1].strip()
            m = re.match(r"^([\-\+]?\d+(?:\.\d+)?(?:/\d+)?)", tail)
            if m is None:
                return 0.0
            candidate = _normalize_answer(m.group(1))
        # Core's own `get_problem` renames the corpus column
        # `expected_answer` to `ground_truth` before this function ever
        # sees it; this package's `corpus.get_problem` (grading.py's
        # sibling module) keeps the original OMI column name, so this
        # reads `expected_answer` instead of `ground_truth`.
        gt = _normalize_answer(problem.get("expected_answer", ""))
        if gt == "":
            return 0.0
        return 1.0 if _answers_equal(candidate, gt) else 0.0
    except Exception:
        return 0.0


# `_answers_equal` itself expects pre-normalized inputs — every core caller
# (`_compute_omi_reward` below, and core's own test suite via a local
# `equal(a, b) = _answers_equal(_normalize_answer(a), _normalize_answer(b))`
# helper in tests/unit/test_math_grader_surface_form.py) normalizes both
# sides first. A bare `answers_equal = _answers_equal` alias would silently
# drop that step for any caller outside this module, so the public entry
# point normalizes explicitly instead of aliasing directly.
def answers_equal(candidate: str, gt: str) -> bool:
    return _answers_equal(_normalize_answer(candidate), _normalize_answer(gt))


compute_reward = _compute_omi_reward
