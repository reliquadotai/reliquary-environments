"""Whether the boxed answer is the right number, and what that is worth.

One reward and no partial credit: the last `\\boxed{}` span states one number
within two percent of the reference, or the answer scores zero.

Unlike the integer corpus of `reliquary-dapo-math`, a science answer is a
measurement. `109`, `108.7` and `1.09 \\times 10^{2}` are the same answer to a
drag problem whose inputs carry three significant figures, and an exact
comparison would grade the rounding rather than the physics. So the number is
compared with a relative tolerance, and everything around it is normalised on
purpose and nothing more: the notation a number is written in (thousands
separators, scientific notation, a simple fraction) and the unit written after
it. The unit is read past rather than converted; the prompt states the unit the
reference is in whenever it has one.

What is refused is a box that states anything other than one number — two
values, a range, a symbol — because a grader that read the first number of a
hedge would pay for the hedge.
"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction

GRADER_VERSION = "boxed-number-rel2pct-v1"
SPEC_SCHEMA = "reliquary/science-verifier/v1"

# Within this share of the reference, or the answer is a different number.
# Two percent, measured: on 200 eval problems answered by Qwen3.8-27B, 9 of
# the 200 landed 1-2% from the reference, and those were the rounding of a
# constant or an intermediate (g = 9.8 against 9.81) rather than a different
# derivation; past 2% the misses were different answers.
RELATIVE_TOLERANCE = 0.02
# A reference of zero has no relative scale; this is the slack it gets instead.
ZERO_TOLERANCE = 1e-9

ANSWER_INSTRUCTION = "\n\nPut your final answer, a single number, within \\boxed{}."

_MINUS = str.maketrans({"\u2212": "-", "\u2013": "-", "\u00d7": "\\times "})
_SUPERSCRIPT = re.compile(r"[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
_SUPERSCRIPT_DIGITS = str.maketrans("⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹", "-+0123456789")

# Wrappers that carry no value: math delimiters and display commands.
_DELIMITERS = re.compile(
    r"\\\(|\\\)|\\\[|\\\]|\$|\\displaystyle|\\left|\\right|\\!|\\,|\\;|\\:|\\ |~"
)
# A relation whose right-hand side is the answer: `v = 5`, `E \approx 2.1`.
_RELATION = re.compile(r"=|\\approx|\\simeq|\\sim|\u2248|\\equiv")
# What may stand left of it: a name, optionally Greek, primed or wrapped in a
# function of one letter (`v`, `\Delta H`, `T'`, `P(A)`), once subscripts —
# which may hold digits, as in `v_2` — are removed.
_SUBSCRIPT = re.compile(r"_\s*(?:\{[^{}]*\}|\w)")
_VARIABLE = re.compile(r"^(?:\\?[A-Za-z]+\s*)+'*(?:\([A-Za-z ]*\))?$")

_DIGITS = r"(?:\d{1,3}(?:(?:,|\{,\})\d{3})+|\d+)(?:\.\d*)?|\.\d+"
_EXPONENT = r"\{?\s*[+-]?\s*\d+\s*\}?"
_NUMBER = re.compile(
    rf"""
    (?P<sign>[+-])?\s*
    (?:
        \\[dt]?frac\{{\s*(?P<fn>{_DIGITS})\s*\}}\{{\s*(?P<fd>{_DIGITS})\s*\}}
      | 10\s*\^\s*(?P<bare>{_EXPONENT})
      | (?P<mantissa>{_DIGITS})
        (?:
            \s*[eE]\s*(?P<e>[+-]?\d+)
          | \s*(?:\\times|\\cdot|\*|x)\s*10\s*\^\s*(?P<p>{_EXPONENT})
          | \s*/\s*(?P<den>{_DIGITS})
        )?
    )
    """,
    re.VERBOSE,
)
# Unit markup to drop before checking what follows the number.
_UNIT_MARKUP = re.compile(
    r"\\(?:text|mathrm|textrm|mbox|operatorname|rm)\s*\{((?:[^{}]|\{[^{}]*\})*)\}"
)
# The commands a unit is spelled with, as the characters they print. Any other
# command left after the markup is a symbol: `5\pi` is not 5.
_UNIT_COMMANDS = {
    "\\mu": "µ", "\\micro": "µ", "\\Omega": "Ω", "\\ohm": "Ω",
    "\\AA": "Å", "\\angstrom": "Å", "\\cdot": "·",
}
# A unit's own exponent: `m^2`, `s^{-1}`, `s⁻¹`.
_UNIT_EXPONENT = re.compile(r"\^\s*\{?\s*[+-]?\s*\d+\s*\}?")
_UNIT_SEPARATORS = re.compile(r"[\s/·*()\-]+")
_SI_PREFIXES = ("", "Y", "Z", "E", "P", "T", "G", "M", "k", "h", "da", "d", "c", "m", "µ", "μ", "u", "n", "p", "f")
_SI_BASES = (
    "m", "g", "s", "A", "K", "mol", "cd", "Hz", "N", "Pa", "J", "W", "C", "V",
    "F", "Ω", "S", "Wb", "T", "H", "L", "l", "eV", "Da", "Bq", "Gy", "Sv",
    "rad", "sr", "lm", "lx", "M", "bar", "cal", "Ci", "bit", "byte",
    "Torr", "Wh", "VA", "var", "FLOPS", "bps",
)
# Symbols a unit is written with. A token outside this set must be a plain
# lowercase word (`molecules`, `years`, `percent`), so that a symbol standing
# for a quantity (`k`, `RT`, `mv`, `k_B T`) never passes for a unit.
_UNIT_TOKENS = frozenset(
    {prefix + base for prefix in _SI_PREFIXES for base in _SI_BASES}
    | {
        "%", "°", "°C", "°F", "°R", "ºR", "Å", "min", "h", "hr", "hrs", "d", "yr",
        "yrs", "y", "atm", "mmHg", "Hg", "psi", "psia", "psig", "ksi", "lb", "lbf",
        "lbm", "lbs", "ft", "in", "mi", "yd", "oz", "gal", "Cal", "BTU", "Btu",
        "ppm", "ppb", "AU", "au", "pc", "kpc", "Mpc", "ly", "c", "erg", "dyn",
        "dyne", "dynes", "G", "Gs", "rem", "mrem", "rpm", "amu", "u", "D", "cP",
        "P", "St", "bbl", "scf", "STB", "acre", "acres", "ha", "mph", "kph",
        "knots", "hp", "kWh", "MWh", "kVA", "kvar", "Mbps", "kbps", "KB", "MB",
        "GB", "TB", "MFLOPS", "GFLOPS", "eq", "Eq", "mEq", "N", "BM", "esu", "Ry",
        "mb", "nb", "pb", "mR",
    }
) - {"kT", "kB", "RT"}
# A lowercase word is a unit (`molecules`, `degrees`) unless it is a function.
_NOT_UNITS = frozenset({"sin", "cos", "tan", "log", "exp", "ln", "and", "or", "to"})
_UNIT_FORBIDDEN = re.compile(r"[+=<>_{}\\|!'≤≥⇌→,;:±]")


def last_boxed_span(text: str) -> str | None:
    """The last `\\boxed{...}` or `\\fbox{...}` substring, or None.

    A balanced-brace walk rather than a regex, because the span it has to close
    is the one the model opened: `\\boxed{\\frac{1}{2}}` closes at the third
    brace, and a lazy pattern would stop at the first.
    """
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return None
    opening = text.index("{", start)
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : position]
    # An unclosed brace is an answer the model never finished writing, which is
    # what a completion cut at the token budget looks like.
    return None


def _plain(digits: str) -> str:
    return digits.replace("{,}", "").replace(",", "")


# Past any physical magnitude and well inside a double. The bound is what
# keeps `10^{999999999}` from costing the grader an exact power of ten.
MAX_EXPONENT = 300


def _exponent(raw: str) -> int:
    exponent = int(re.sub(r"[\s{}+]", "", raw))
    if abs(exponent) > MAX_EXPONENT:
        raise ValueError(f"exponent {exponent} is beyond the bound")
    return exponent


def unit_text(rest: str) -> str | None:
    """What follows a number, as a unit, or None if it is not one.

    Empty is a unit (a bare number). Otherwise every token must be a unit
    symbol, with an SI prefix or not, or a plain lowercase word: a digit
    outside an exponent is a second value, an operator or a subscript is an
    expression, and a capitalised name or a lone letter is a symbol.
    """
    unit = _UNIT_MARKUP.sub(r" \1 ", rest)
    unit = re.sub(r"\^\s*\\circ|\\circ|\\degree", "°", unit)
    unit = unit.replace("\\%", "%")
    for command, printed in _UNIT_COMMANDS.items():
        unit = re.sub(re.escape(command) + r"(?![A-Za-z])", printed, unit)
    # A prefix written apart from its unit, `\mu m`, is still one unit.
    unit = re.sub(r"([µμ])\s+(?=[A-Za-zΩ])", r"\1", unit)
    unit = " ".join(unit.split())
    if not unit:
        return ""
    if unit[0] in "-/":
        # `2-butyne` and `24-hour` are names with a number in them.
        return None
    bare = _UNIT_EXPONENT.sub(" ", unit)
    if re.search(r"\d", bare) or _UNIT_FORBIDDEN.search(bare):
        return None
    tokens = [token.rstrip(".") for token in _UNIT_SEPARATORS.split(bare) if token]
    if not tokens or len(tokens) > 5:
        return None
    for token in tokens:
        if token in _UNIT_TOKENS or token == "per":
            continue
        if token.isascii() and token.isalpha() and token.islower() and len(token) >= 3:
            # A word has a vowel; `mnv` is a product of symbols.
            if token not in _NOT_UNITS and re.search(r"[aeiouy]", token):
                continue
        return None
    return unit


def split_number(text: str, *, relation: bool = True) -> tuple[float, str] | None:
    """The one number `text` states and the unit text after it, or None.

    With `relation`, one relation whose left-hand side names a variable is read
    past (`v_2 = 5 m/s` states 5); an equation, or two relations, is not one
    number. What follows the number may be a unit and nothing else: any further
    digit outside a unit's own exponent means a second value.
    """
    text = _DELIMITERS.sub(" ", text.translate(_MINUS)).strip()
    # `10⁻⁴` and `s⁻¹` are `10^{-4}` and `s^{-1}`, written in another script.
    text = _SUPERSCRIPT.sub(
        lambda found: "^{" + found.group().translate(_SUPERSCRIPT_DIGITS) + "}", text
    )
    sides = _RELATION.split(text)
    if len(sides) > 2 or (len(sides) == 2 and not relation):
        return None
    if len(sides) == 2:
        name = _SUBSCRIPT.sub("", sides[0]).strip()
        if name and not _VARIABLE.match(name):
            return None
        text = sides[1].strip()
    match = _NUMBER.match(text)
    if match is None:
        return None
    try:
        if match["fn"] is not None:
            denominator = Fraction(_plain(match["fd"]))
            if denominator == 0:
                return None
            value = Fraction(_plain(match["fn"])) / denominator
        elif match["bare"] is not None:
            value = Fraction(10) ** _exponent(match["bare"])
        else:
            value = Fraction(_plain(match["mantissa"]))
            if match["e"] is not None:
                value *= Fraction(10) ** _exponent(match["e"])
            elif match["p"] is not None:
                value *= Fraction(10) ** _exponent(match["p"])
            elif match["den"] is not None:
                denominator = Fraction(_plain(match["den"]))
                if denominator == 0:
                    return None
                value /= denominator
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if match["sign"] == "-":
        value = -value

    rest = text[match.end() :]
    if re.match(r"\s*\^(?!\s*\\circ)", rest):
        # A power on the number itself: `5^2` is 25, and `10^{-3}` was read
        # above. Neither is the number before the caret.
        return None
    unit = unit_text(rest)
    if unit is None:
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if not math.isfinite(number):
        return None
    return number, unit


def parse_number(text: str) -> float | None:
    """The one number `text` states, or None."""
    parsed = split_number(text)
    return None if parsed is None else parsed[0]


def close_enough(given: float, answer: float) -> bool:
    if answer == 0.0:
        return abs(given) <= ZERO_TOLERANCE
    return abs(given - answer) <= RELATIVE_TOLERANCE * abs(answer)


def verifier_spec(answer: float) -> str:
    """Everything grading needs, as one string the task can carry."""
    return json.dumps(
        {"schema": SPEC_SCHEMA, "grader_version": GRADER_VERSION, "answer": answer},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def grade(spec: str, completion: str | None) -> float:
    """1.0 if the last boxed span states the number `spec` holds, 0.0 otherwise."""
    try:
        parsed = json.loads(spec)
    except (TypeError, ValueError):
        return 0.0
    if not isinstance(parsed, dict):
        return 0.0
    if parsed.get("schema") != SPEC_SCHEMA:
        return 0.0
    if parsed.get("grader_version") != GRADER_VERSION:
        return 0.0
    answer = parsed.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, (int, float)):
        return 0.0
    if not math.isfinite(answer):
        return 0.0

    if not isinstance(completion, str):
        return 0.0
    span = last_boxed_span(completion)
    if span is None:
        # No box, no answer: a number picked out of the prose would reward
        # whichever number the reasoning happened to end on.
        return 0.0
    given = parse_number(span)
    return 1.0 if given is not None and close_enough(given, float(answer)) else 0.0


def reference_completion(answer: float) -> str:
    """The shortest completion that scores 1.0."""
    return f"\\boxed{{{answer!r}}}"


__all__ = [
    "ANSWER_INSTRUCTION",
    "GRADER_VERSION",
    "RELATIVE_TOLERANCE",
    "SPEC_SCHEMA",
    "close_enough",
    "grade",
    "last_boxed_span",
    "parse_number",
    "reference_completion",
    "split_number",
    "verifier_spec",
]
