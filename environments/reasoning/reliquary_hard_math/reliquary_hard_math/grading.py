"""Whether the boxed answer is the reference, and what that is worth.

One reward and no partial credit: the last `\\boxed{}` span of the answer
states the same mathematical object as the reference, or the answer scores
zero.

Unlike `reliquary-dapo-math`, whose upstream rewrote every answer into an
integer, the references here are what the problems ask for: 58% integers, the
rest fractions, radicals, multiples of pi, simple expressions in the problem's
own letters, ordered tuples, lists of solutions and ratios. A string comparison
would grade the spelling — `\\frac{\\sqrt{2}}{2}` against `\\dfrac{1}{\\sqrt2}` —
so both sides are parsed and compared by value. The design is about what the
comparison refuses, because a corpus filter that pays a wrong answer ships it
as training data:

* **No guessing.** The grammar is a small, closed subset of LaTeX: numbers,
  `+ - * /`, `\\frac`, `\\sqrt` (with an index), `\\pi`, powers, single-letter
  names with a subscript, brackets. Anything outside it — a function, a word,
  `\\infty`, an inequality — is not parsed and scores zero. Spellings whose
  reading is ambiguous are refused rather than read one way: a mixed number
  (`2\\frac{1}{2}`), digits after an implicit product (`(a)5`), an unbraced
  multi-digit exponent (`2^10`, which LaTeX prints as 2¹0).
* **Exact where it can be.** A rational value (`5`, `0.5`, `\\frac{10}{4}`,
  `2^{2011}`, `\\sqrt{16}`) is computed as an exact fraction and compared
  exactly, so `5` never equals `5/2` and `0.333` never equals `1/3`, at any
  magnitude.
* **Numeric where it must be, at a precision no rounding reaches.** A value
  with a surd or pi is evaluated in decimal arithmetic at 60 significant digits
  or more (more for large magnitudes) and compared to 40. An answer rounded to
  any number of digits a model would write is far outside that; two distinct
  closed forms agreeing to 40 digits do not occur in a competition answer.
* **Letters are compared as identities.** An answer in the problem's letters
  (`n+2`, `\\frac{a}{b+c}`) is evaluated at three fixed points per letter and
  must agree at every point where both sides are defined, and at least two;
  the two sides must use the same letters.
* **Structure is not normalised away.** A tuple compares element by element
  with the same brackets (`(0,1)` is not `[0,1]`); a list of tuples, which is
  how a problem's solution set is written, compares as a multiset; a ratio
  compares part by part (`1:2` is not `2:4`).
* **Bounded.** No `eval`, no sympy, no code: a hand-written parser over at most
  `MAX_ANSWER_CHARS` characters and `MAX_NODES` nodes, integer exponents up to
  `MAX_EXPONENT`, exact values up to `MAX_EXACT_DIGITS` digits and decimal
  precision up to `MAX_PRECISION`. Everything is standard-library arithmetic,
  so the verdict is the same on every machine.

Where the answer is read from: the text after the last `</think>` when the
completion carries one, nothing at all when it opens `<think>` without closing
it, and the whole completion otherwise. The last of these is what a direct
completion, a harness that strips the reasoning, and a chat template that opens
the block in the prompt all look like — which means a completion cut at its
budget inside a block the template opened is read whole. See README.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import (
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from fractions import Fraction

GRADER_VERSION = "boxed-expression-v1"
SPEC_SCHEMA = "reliquary/hard-math-verifier/v1"

ANSWER_INSTRUCTION = "\n\nPut your final answer within \\boxed{}."

# Bounds. Every one of them is far above what a reference needs (the longest is
# 60 characters) and far below what would cost the grader real time.
MAX_ANSWER_CHARS = 400
MAX_NODES = 400
MAX_EXPONENT = 100_000
MAX_EXACT_DIGITS = 100_000
BASE_PRECISION = 60
# Agreement required on the numeric path, in significant digits.
AGREEMENT_DIGITS = 40
MAX_PRECISION = 3_000
MAX_ELEMENTS = 64
# Where letters are evaluated, fixed so the verdict is reproducible. Three
# kinds of point, because the letters of a competition answer live in different
# places: a probability in (0, 1), a length past 1, an index that has to be an
# integer for `(-1)^n` to exist. A point where either side is undefined is
# skipped; every other point must agree, and at least `MIN_POINTS` must.
POINT_KINDS = ("small",) * 2 + ("unit",) * 3 + ("real",) * 3 + ("integer",) * 4
MIN_POINTS = 2


class _Refused(Exception):
    """The text is not an answer this grader reads."""


class _NotExact(Exception):
    """The value is not a rational this grader computes exactly."""


class _Undefined(Exception):
    """The value does not exist at this point (a negative radicand, 1/0)."""


# --------------------------------------------------------------------------
# Where the answer is
# --------------------------------------------------------------------------


def answer_region(completion: str) -> str:
    """The part of a completion an answer is read from."""
    if "</think>" in completion:
        return completion.rsplit("</think>", 1)[1]
    if "<think>" in completion:
        # Opened and never closed: all of it is reasoning.
        return ""
    return completion


def last_boxed_span(text: str) -> str | None:
    """The content of the last `\\boxed{...}` or `\\fbox{...}`, or None.

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


# --------------------------------------------------------------------------
# Normalisation: spellings that carry no value
# --------------------------------------------------------------------------

_UNICODE = (
    ("\u2212", "-"), ("\u2013", "-"), ("\u00d7", "\\cdot "), ("\u22c5", "\\cdot "),
    ("\u00b7", "\\cdot "), ("\u00f7", "/"), ("\u03c0", "\\pi "), ("\u221a", "\\sqrt"),
)
_DROPPED = (
    "\\displaystyle", "\\textstyle", "\\left", "\\right", "\\bigl", "\\bigr",
    "\\Bigl", "\\Bigr", "\\big", "\\Big", "\\,", "\\;", "\\:", "\\!", "\\ ", "~",
    "$", "\\(", "\\)", "\\[", "\\]",
)
_DEGREES = re.compile(r"\^\s*\{\s*\\circ\s*\}|\^\s*\\circ|\\circ|\\degree|°|º")
# A unit written after the value: `5 \text{ cm}`. Only a trailing block, and
# only letters in it — `5 \text{ or } 7` keeps its block and fails to parse.
_TRAILING_UNIT = re.compile(
    r"\\(?:text|textrm|mathrm|mbox)\s*\{\s*[A-Za-z][A-Za-z ]*\}\s*(?:\^\s*\{?\d\}?)?$"
)
_EMPTY_TEXT = re.compile(r"\\(?:text|textrm|mathrm|mbox)\s*\{\s*\}")
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
# Digit groups set apart by spacing, which is how `2\,449\,999` arrives once
# the spacing command is gone.
_SPACED_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(?: \d{3})+(?:\.\d+)?$")
_NAMED_VALUE = re.compile(r"^[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?(?:\([^()]*\))?=([^=]+)$")
_CONTROL_WORD = re.compile(r"\\[A-Za-z]+")


def normalise(text: str) -> str:
    """The answer with every spelling that carries no value removed."""
    for character, replacement in _UNICODE:
        text = text.replace(character, replacement)
    text = _DEGREES.sub("", text)
    for token in _DROPPED:
        text = text.replace(token, " ")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("{,}", ",").replace("\\%", "").replace("%", "")
    text = _EMPTY_TEXT.sub(" ", text)
    text = text.strip().rstrip(".").strip()
    stripped = _TRAILING_UNIT.sub("", text).strip()
    if stripped:
        text = stripped
    text = " ".join(text.split())
    if _THOUSANDS.match(text.replace(" ", "")) or _SPACED_THOUSANDS.match(text):
        text = text.replace(" ", "").replace(",", "")
    named = _NAMED_VALUE.match(text.replace(" ", ""))
    if named is not None:
        # `x = 5` names the answer it gives. Only a lone name on the left: an
        # equation with an expression on each side is not a value.
        text = text.split("=", 1)[1].strip()
    return text


# --------------------------------------------------------------------------
# LaTeX arguments: `\frac23`, `\sqrt3`, `x^2` become braced
# --------------------------------------------------------------------------


def _group_end(text: str, start: int, opening: str, closing: str) -> int:
    """Index just past the group opened at `start`."""
    depth = 0
    for position in range(start, len(text)):
        if text[position] == opening:
            depth += 1
        elif text[position] == closing:
            depth -= 1
            if depth == 0:
                return position + 1
    raise _Refused("unbalanced group")


def _argument(text: str, position: int, *, exponent: bool) -> tuple[str, int]:
    """One LaTeX argument starting at `position`, braced, and where it ends."""
    while position < len(text) and text[position] == " ":
        position += 1
    if position >= len(text):
        raise _Refused("missing argument")
    character = text[position]
    if character == "{":
        end = _group_end(text, position, "{", "}")
        return "{" + _brace_arguments(text[position + 1 : end - 1]) + "}", end
    if character == "(":
        # Not LaTeX, but how a power or a root is written in plain text, and it
        # has only one reading.
        end = _group_end(text, position, "(", ")")
        return "{" + _brace_arguments(text[position : end]) + "}", end
    if character == "\\":
        word = _CONTROL_WORD.match(text, position)
        if word is None:
            raise _Refused("bad control sequence")
        if word.group() in ("\\frac", "\\sqrt"):
            # `\sqrt\frac{3}{5}`: the argument is the whole fraction.
            parts = [word.group()]
            end = word.end()
            for _ in range(2 if word.group() == "\\frac" else 1):
                braced, end = _argument(text, end, exponent=False)
                parts.append(braced)
            return "{" + "".join(parts) + "}", end
        return "{" + word.group() + "}", word.end()
    if character.isdigit():
        if exponent and position + 1 < len(text) and text[position + 1].isdigit():
            # LaTeX prints `2^10` as 2¹0; plain text means 2¹⁰. Neither is
            # guessed.
            raise _Refused("unbraced multi-digit exponent")
        return "{" + character + "}", position + 1
    if character.isalpha():
        return "{" + character + "}", position + 1
    raise _Refused(f"bad argument {character!r}")


_ROOT_OR_FRACTION = re.compile(r"\\(frac|sqrt)(?![A-Za-z])")


def _brace_arguments(text: str) -> str:
    output: list[str] = []
    position = 0
    while position < len(text):
        command = _ROOT_OR_FRACTION.match(text, position)
        if command is not None:
            output.append(command.group())
            position = command.end()
            if command.group(1) == "sqrt":
                while position < len(text) and text[position] == " ":
                    position += 1
                if position < len(text) and text[position] == "[":
                    end = _group_end(text, position, "[", "]")
                    output.append("[" + _brace_arguments(text[position + 1 : end - 1]) + "]")
                    position = end
                arguments = 1
            else:
                arguments = 2
            for _ in range(arguments):
                braced, position = _argument(text, position, exponent=False)
                output.append(braced)
            continue
        if text[position] == "^":
            braced, position = _argument(text, position + 1, exponent=True)
            output.append("^" + braced)
            continue
        output.append(text[position])
        position += 1
    return "".join(output)


# --------------------------------------------------------------------------
# Tokens and the expression grammar
# --------------------------------------------------------------------------

_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<number>\d+(?:\.\d+)?|\.\d+)
  | (?P<name>[A-Za-z](?:_(?:\{[^{}]+\}|[A-Za-z0-9]))?)
  | (?P<command>\\[A-Za-z]+)
  | (?P<symbol>[-+*/^(){}\[\]])
    """,
    re.VERBOSE,
)
_COMMANDS = {"\\frac", "\\sqrt", "\\pi", "\\cdot", "\\times", "\\div"}


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    text: str


def _tokens(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    letters = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            raise _Refused(f"unexpected {text[position]!r}")
        position = match.end()
        kind = match.lastgroup
        assert kind is not None
        if kind == "space":
            letters = 0
            continue
        value = match.group()
        if kind == "command" and value not in _COMMANDS:
            raise _Refused(f"unsupported {value}")
        if kind == "name":
            letters += 1
            if letters > 2 and "_" not in value:
                # Three letters in a row are a word or a function — `sin`,
                # `and` — not a product a reference is written as.
                raise _Refused("a word")
            # A subscripted name is one opaque symbol, `F_{n+1}` as much as
            # `a_x`: spacing inside it is dropped and `a_{x}` is `a_x`.
            value = value.replace(" ", "")
            if re.fullmatch(r"[A-Za-z]_\{[A-Za-z0-9]\}", value):
                value = value[:2] + value[3]
        else:
            letters = 0
        tokens.append(_Token(kind, value))
    return tokens


# AST nodes are tuples: ("num", Fraction) ("pi",) ("var", name)
# ("add"|"sub"|"mul"|"div"|"pow", a, b) ("neg", a) ("root", a, index)
Node = tuple


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.position = 0
        self.nodes = 0

    def _node(self, *parts: object) -> Node:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise _Refused("too large")
        return tuple(parts)

    def peek(self) -> _Token | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self, text: str | None = None) -> _Token:
        token = self.peek()
        if token is None or (text is not None and token.text != text):
            raise _Refused(f"expected {text!r}")
        self.position += 1
        return token

    def parse(self) -> Node:
        node = self.expression()
        if self.peek() is not None:
            raise _Refused(f"trailing {self.peek().text!r}")
        return node

    def expression(self) -> Node:
        node = self.term()
        while (token := self.peek()) is not None and token.text in ("+", "-"):
            self.take()
            node = self._node("add" if token.text == "+" else "sub", node, self.term())
        return node

    def _starts_implicit_factor(self, token: _Token) -> bool:
        return token.kind == "name" or token.text in ("\\frac", "\\sqrt", "\\pi", "(", "{")

    def term(self) -> Node:
        node = self.factor()
        while (token := self.peek()) is not None:
            if token.text in ("*", "/", "\\cdot", "\\times", "\\div"):
                self.take()
                kind = "div" if token.text in ("/", "\\div") else "mul"
                node = self._node(kind, node, self.factor())
            elif token.kind == "number":
                # `2 3`, `(a)5`, `x^2 3`: a number never joins a product
                # silently. The one exception is a power of it after anything
                # but a number — `n2^{n-1}`, `\sqrt{3}2^{1/3}` — which has one
                # reading; `24 4^{n}` still has two.
                following = self.tokens[self.position + 1] if self.position + 1 < len(self.tokens) else None
                if node[0] == "num" or following is None or following.text != "^":
                    raise _Refused("digits after an implicit product")
                node = self._node("mul", node, self.factor())
            elif self._starts_implicit_factor(token):
                if (
                    node[0] == "num"
                    and token.text == "\\frac"
                    and self._numeric_fraction_ahead()
                ):
                    # `2\frac{1}{2}` is 2.5 to a person and 1 to a parser.
                    raise _Refused("mixed number")
                node = self._node("mul", node, self.factor())
            else:
                break
        return node

    def _numeric_fraction_ahead(self) -> bool:
        window = [token.text for token in self.tokens[self.position : self.position + 7]]
        return (
            len(window) == 7
            and window[1] == "{" and window[3] == "}" and window[4] == "{" and window[6] == "}"
            and re.fullmatch(r"\d+", window[2]) is not None
            and re.fullmatch(r"\d+", window[5]) is not None
        )

    def factor(self) -> Node:
        token = self.peek()
        if token is not None and token.text in ("+", "-"):
            self.take()
            operand = self.factor()
            return operand if token.text == "+" else self._node("neg", operand)
        return self.power()

    def power(self) -> Node:
        base = self.primary()
        token = self.peek()
        if token is not None and token.text == "^":
            self.take()
            exponent = self.braced()
            following = self.peek()
            if following is not None and following.text == "^":
                raise _Refused("double superscript")
            return self._node("pow", base, exponent)
        return base

    def braced(self) -> Node:
        self.take("{")
        node = self.expression()
        self.take("}")
        return node

    def primary(self) -> Node:
        token = self.peek()
        if token is None:
            raise _Refused("missing operand")
        if token.kind == "number":
            self.take()
            return self._node("num", Fraction(token.text))
        if token.kind == "name":
            self.take()
            return self._node("var", token.text)
        if token.text == "\\pi":
            self.take()
            return self._node("pi")
        if token.text == "(":
            self.take()
            node = self.expression()
            self.take(")")
            return node
        if token.text == "{":
            return self.braced()
        if token.text == "\\frac":
            self.take()
            numerator = self.braced()
            denominator = self.braced()
            return self._node("div", numerator, denominator)
        if token.text == "\\sqrt":
            self.take()
            index: Node = ("num", Fraction(2))
            if self.peek() is not None and self.peek().text == "[":
                self.take()
                index = self.expression()
                self.take("]")
            return self._node("root", self.braced(), index)
        raise _Refused(f"unexpected {token.text!r}")


def parse_expression(text: str) -> Node:
    """The expression tree of one scalar answer, or `_Refused`."""
    if not text or len(text) > MAX_ANSWER_CHARS:
        raise _Refused("empty or oversized")
    return _Parser(_tokens(_brace_arguments(text))).parse()


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------


def _digits(value: Fraction) -> int:
    """Decimal digits of numerator and denominator, from their bit lengths.

    Not `len(str(...))`: converting an integer past 4,300 digits to text is
    refused by Python itself, and is quadratic before that.
    """
    bits = abs(value.numerator).bit_length() + value.denominator.bit_length()
    return int(bits * 0.30103) + 2


def _exact_root(value: Fraction, index: int) -> Fraction:
    if index < 1 or index > MAX_EXPONENT:
        raise _NotExact
    if value < 0:
        if index % 2 == 0:
            raise _Undefined
        return -_exact_root(-value, index)

    def integer_root(n: int) -> int:
        if n < 2:
            return n
        guess = round(n ** (1.0 / index)) if n.bit_length() < 1000 else 1 << (n.bit_length() // index)
        # Newton's method from above, then settle.
        x = max(guess, 1)
        while True:
            y = ((index - 1) * x + n // x ** (index - 1)) // index
            if y >= x:
                break
            x = y
        while x**index > n:
            x -= 1
        while (x + 1) ** index <= n:
            x += 1
        return x

    numerator = integer_root(value.numerator)
    denominator = integer_root(value.denominator)
    if numerator**index != value.numerator or denominator**index != value.denominator:
        raise _NotExact
    return Fraction(numerator, denominator)


def _exact_power(base: Fraction, exponent: Fraction) -> Fraction:
    if abs(exponent.numerator) > MAX_EXPONENT or exponent.denominator > MAX_EXPONENT:
        raise _NotExact
    if base == 0:
        if exponent <= 0:
            raise _Undefined
        return Fraction(0)
    if _digits(base) * abs(exponent.numerator) > MAX_EXACT_DIGITS:
        raise _NotExact
    root = _exact_root(base, exponent.denominator) if exponent.denominator != 1 else base
    return root ** exponent.numerator


def exact_value(node: Node) -> Fraction:
    """The node's value as an exact rational, or `_NotExact`/`_Undefined`."""
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind in ("pi", "var"):
        raise _NotExact
    if kind == "neg":
        return -exact_value(node[1])
    if kind == "root":
        index = exact_value(node[2])
        if index.denominator != 1:
            raise _NotExact
        return _exact_root(exact_value(node[1]), int(index))
    left, right = exact_value(node[1]), exact_value(node[2])
    if kind == "add":
        value = left + right
    elif kind == "sub":
        value = left - right
    elif kind == "mul":
        value = left * right
    elif kind == "div":
        if right == 0:
            raise _Undefined
        value = left / right
    else:
        value = _exact_power(left, right)
    if _digits(value) > MAX_EXACT_DIGITS:
        raise _NotExact
    return value


_PI_CACHE: dict[int, Decimal] = {}


def _pi(precision: int) -> Decimal:
    """Pi to `precision` digits, by the `decimal` documentation's recipe."""
    if precision not in _PI_CACHE:
        with localcontext() as context:
            context.prec = precision + 5
            three = Decimal(3)
            last, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
            while s != last:
                last = s
                n, na = n + na, na + 8
                d, da = d + da, da + 32
                t = (t * n) / d
                s += t
        with localcontext() as context:
            context.prec = precision
            _PI_CACHE[precision] = +s
    return _PI_CACHE[precision]


def _point_value(name: str, point: int) -> Decimal:
    """The value letter `name` takes at `point`, drawn from a fixed hash.

    `small` points fall in [0.01, 0.2), `unit` points in [0.05, 0.95), `real`
    points in [1.25, 4.25) and `integer` points on 5..13 — past 4, where `2^n`
    and `n^2` still meet.
    """
    digest = hashlib.sha256(f"{GRADER_VERSION}:{name}:{point}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big")
    kind = POINT_KINDS[point]
    if kind == "integer":
        return Decimal(5 + draw % 9)
    fraction = Decimal(draw) / Decimal(1 << 64)
    if kind == "small":
        return fraction * Decimal("0.19") + Decimal("0.01")
    if kind == "unit":
        return fraction * Decimal("0.9") + Decimal("0.05")
    return fraction * 3 + Decimal("1.25")


def _decimal_power(base: Decimal, exponent_node: Node, environment: dict[str, Decimal]) -> Decimal:
    try:
        exponent = exact_value(exponent_node)
    except _NotExact:
        exponent = None
    if exponent is not None:
        if abs(exponent.numerator) > MAX_EXPONENT:
            raise _Undefined
        if exponent.denominator > MAX_EXPONENT:
            # `2^{2^{-2022}}`: a root too deep to take exactly, of a positive
            # base only.
            if base <= 0:
                raise _Undefined
            return base ** (Decimal(exponent.numerator) / Decimal(exponent.denominator))
        if exponent.denominator == 1:
            if base == 0 and exponent <= 0:
                raise _Undefined
            return base ** int(exponent)
        if base < 0:
            if exponent.denominator % 2 == 0:
                raise _Undefined
            magnitude = (-base) ** (Decimal(exponent.numerator) / Decimal(exponent.denominator))
            return magnitude if exponent.numerator % 2 == 0 else -magnitude
        if base == 0:
            if exponent <= 0:
                raise _Undefined
            return Decimal(0)
        return base ** (Decimal(exponent.numerator) / Decimal(exponent.denominator))
    power = decimal_value(exponent_node, environment)
    if abs(power) > MAX_EXPONENT:
        raise _Undefined
    if power == power.to_integral_value():
        # An exponent in letters that lands on an integer at this point:
        # `(-1)^n` at an integer point exists, at any other it does not.
        if base == 0 and power <= 0:
            raise _Undefined
        return base ** int(power)
    if base <= 0:
        raise _Undefined
    return base**power


def decimal_value(node: Node, environment: dict[str, Decimal]) -> Decimal:
    """The node's value in the current decimal context."""
    kind = node[0]
    if kind == "num":
        return Decimal(node[1].numerator) / Decimal(node[1].denominator)
    if kind == "pi":
        return environment["\\pi"]
    if kind == "var":
        return environment[node[1]]
    if kind == "neg":
        return -decimal_value(node[1], environment)
    if kind == "root":
        radicand = decimal_value(node[1], environment)
        try:
            index = exact_value(node[2])
        except _NotExact:
            raise _Undefined from None
        if index.denominator != 1 or index < 1 or index > MAX_EXPONENT:
            raise _Undefined
        index = int(index)
        if index == 2:
            if radicand < 0:
                raise _Undefined
            return radicand.sqrt()
        if radicand < 0:
            if index % 2 == 0:
                raise _Undefined
            return -((-radicand) ** (Decimal(1) / Decimal(index)))
        if radicand == 0:
            return Decimal(0)
        return radicand ** (Decimal(1) / Decimal(index))
    left = decimal_value(node[1], environment)
    if kind == "pow":
        return _decimal_power(left, node[2], environment)
    right = decimal_value(node[2], environment)
    if kind == "add":
        return left + right
    if kind == "sub":
        return left - right
    if kind == "mul":
        return left * right
    if right == 0:
        raise _Undefined
    return left / right


def _names(node: Node) -> frozenset[str]:
    if node[0] == "var":
        return frozenset((node[1],))
    names: frozenset[str] = frozenset()
    for part in node[1:]:
        if isinstance(part, tuple):
            names |= _names(part)
    return names


def _evaluate(node: Node, precision: int, point: int, names: frozenset[str]) -> Decimal:
    context = Context(prec=precision, Emax=10**6, Emin=-(10**6))
    with localcontext(context):
        environment = {name: +_point_value(name, point) for name in names}
        environment["\\pi"] = _pi(precision)
        try:
            return +decimal_value(node, environment)
        except (InvalidOperation, DivisionByZero, Overflow, ZeroDivisionError):
            raise _Undefined from None


def _agree(left: Decimal, right: Decimal, precision: int) -> bool:
    """Agreement to all but the last `precision - AGREEMENT_DIGITS` digits.

    Relative below a magnitude of one; above it the precision grows with the
    magnitude, so the tolerance stays at 10^-40 in absolute terms and a value
    with 600 digits is still told apart from its neighbour.
    """
    with localcontext(Context(prec=precision, Emax=10**6, Emin=-(10**6))):
        scale = max(abs(left), abs(right))
        if scale == 0:
            return True
        slack = precision - (BASE_PRECISION - AGREEMENT_DIGITS)
        return abs(left - right) <= scale * Decimal(10) ** -slack


def _precision_for(values: list[Decimal]) -> int:
    magnitude = max((value.adjusted() for value in values if value != 0), default=0)
    return BASE_PRECISION + max(0, magnitude)


def values_equal(candidate: Node, reference: Node) -> bool:
    """Whether two expression trees state the same value."""
    names = _names(candidate)
    if names != _names(reference):
        return False
    if not names:
        try:
            return exact_value(candidate) == exact_value(reference)
        except (_NotExact, _Undefined):
            pass
    agreed = 0
    for point in range(len(POINT_KINDS) if names else 1):
        try:
            first = [_evaluate(node, BASE_PRECISION, point, names) for node in (candidate, reference)]
            precision = _precision_for(first)
            if precision > MAX_PRECISION:
                continue
            if precision > BASE_PRECISION:
                first = [_evaluate(node, precision, point, names) for node in (candidate, reference)]
        except _Undefined:
            continue
        if not _agree(first[0], first[1], precision):
            return False
        agreed += 1
    return agreed >= (MIN_POINTS if names else 1)


# --------------------------------------------------------------------------
# Structure: scalars, tuples, lists of tuples, ratios
# --------------------------------------------------------------------------



def _split_top(text: str, separator: str) -> list[str]:
    """`text` split on `separator` wherever it is outside every bracket."""
    parts: list[str] = []
    depth = 0
    start = 0
    for position, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
            if depth < 0:
                raise _Refused("unbalanced")
        elif character == separator and depth == 0:
            parts.append(text[start:position])
            start = position + 1
    if depth != 0:
        raise _Refused("unbalanced")
    parts.append(text[start:])
    return [part.strip() for part in parts]


def _closes_at_end(text: str) -> bool:
    """Whether the bracket opening `text` is the one closing it."""
    depth = 0
    for position, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
            if depth == 0:
                return position == len(text) - 1
    return False


@dataclass(frozen=True, slots=True)
class Answer:
    """A parsed answer: its shape, its brackets, and its parts."""

    shape: str  # "scalar" | "tuple" | "list" | "ratio"
    brackets: str
    parts: tuple


def _tuple(text: str) -> Answer | None:
    if len(text) >= 2 and text[0] in "([" and text[-1] in ")]" and _closes_at_end(text):
        inner = _split_top(text[1:-1], ",")
        if len(inner) >= 2:
            if len(inner) > MAX_ELEMENTS:
                raise _Refused("too many elements")
            return Answer("tuple", text[0] + text[-1], tuple(parse_expression(part) for part in inner))
    return None


def parse_answer(text: str) -> Answer:
    """The structure an answer states, or `_Refused`."""
    text = normalise(text)
    if not text or len(text) > MAX_ANSWER_CHARS:
        raise _Refused("empty or oversized")
    if text.startswith("\\{") and text.endswith("\\}"):
        # A solution set written as a set: its members are what is compared.
        text = text[2:-2].strip()
    if "\\{" in text or "\\}" in text:
        raise _Refused("a set")
    items = _split_top(text, ",")
    if len(items) > 1:
        if len(items) > MAX_ELEMENTS:
            raise _Refused("too many elements")
        members = [_tuple(item) for item in items]
        if any(member is None for member in members):
            # `1, 2, 3` says neither whether order matters nor what is listed.
            raise _Refused("a bare list")
        return Answer("list", "", tuple(members))
    single = _tuple(text)
    if single is not None:
        return single
    ratio = _split_top(text, ":")
    if len(ratio) > 1:
        return Answer("ratio", ":", tuple(parse_expression(part) for part in ratio))
    return Answer("scalar", "", (parse_expression(text),))


def _ordered_equal(candidate: tuple, reference: tuple) -> bool:
    return len(candidate) == len(reference) and all(
        values_equal(left, right) for left, right in zip(candidate, reference, strict=True)
    )


def answers_equal(candidate: Answer, reference: Answer) -> bool:
    """Whether two parsed answers state the same object."""
    if candidate.shape != reference.shape or candidate.brackets != reference.brackets:
        return False
    if candidate.shape != "list":
        return _ordered_equal(candidate.parts, reference.parts)
    if len(candidate.parts) != len(reference.parts):
        return False
    # A solution set is a multiset of tuples: each member is matched to one
    # reference member it equals, and none is used twice.
    unmatched = list(reference.parts)
    for member in candidate.parts:
        for position, other in enumerate(unmatched):
            if answers_equal(member, other):
                del unmatched[position]
                break
        else:
            return False
    return True


def equivalent(candidate: str, reference: str) -> bool:
    """Whether answer text `candidate` states what `reference` states."""
    try:
        return answers_equal(parse_answer(candidate), parse_answer(reference))
    except (_Refused, _NotExact, _Undefined, RecursionError, ValueError,
            ArithmeticError, MemoryError):
        return False


def gradable(reference: str) -> bool:
    """Whether a reference is one this grader reads and recognises as itself."""
    try:
        parse_answer(reference)
    except (_Refused, RecursionError, ValueError, ArithmeticError):
        return False
    return grade(verifier_spec(reference), reference_completion(reference)) == 1.0


# --------------------------------------------------------------------------
# The reward
# --------------------------------------------------------------------------


def verifier_spec(answer: str) -> str:
    """Everything grading needs, as one string the task can carry."""
    return json.dumps(
        {"schema": SPEC_SCHEMA, "grader_version": GRADER_VERSION, "answer": answer},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def grade(spec: str, completion: str | None) -> float:
    """1.0 if the last boxed span of the answer states the reference, else 0.0."""
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
    if not isinstance(answer, str) or not answer:
        # A spec with no answer to compare against would pay for having been
        # stripped of the thing being asked.
        return 0.0
    if not isinstance(completion, str):
        return 0.0
    span = last_boxed_span(answer_region(completion))
    if span is None:
        # No box, no answer: a value picked out of the prose would reward
        # whichever value the reasoning happened to end on.
        return 0.0
    return 1.0 if equivalent(span, answer) else 0.0


def reference_completion(answer: str) -> str:
    """The shortest completion that scores 1.0."""
    return f"\\boxed{{{answer}}}"


def wrong_answer(answer: str) -> str:
    """An answer of the reference's own shape that does not equal it.

    The first scalar is moved by one; everything else is kept. It is the
    cheapest wrong answer a grader that read only the shape would pay.
    """
    text = normalise(answer)
    if text.startswith("\\{") and text.endswith("\\}"):
        text = text[2:-2].strip()
    items = _split_top(text, ",")
    first = items[0]
    if len(first) >= 2 and first[0] in "([" and _closes_at_end(first):
        inner = _split_top(first[1:-1], ",")
        if len(inner) >= 2:
            inner[0] = f"({inner[0]})+1"
            items[0] = first[0] + ",".join(inner) + first[-1]
            return ", ".join(items)
    ratio = _split_top(first, ":")
    ratio[0] = f"({ratio[0]})+1"
    return ":".join(ratio)


__all__ = [
    "ANSWER_INSTRUCTION",
    "GRADER_VERSION",
    "SPEC_SCHEMA",
    "Answer",
    "answer_region",
    "answers_equal",
    "equivalent",
    "gradable",
    "grade",
    "last_boxed_span",
    "normalise",
    "parse_answer",
    "parse_expression",
    "reference_completion",
    "values_equal",
    "verifier_spec",
    "wrong_answer",
]
