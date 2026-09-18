"""Whether the boxed answer is the right integer, and what that is worth.

One reward and no partial credit: the last `\\boxed{}` span parses to the
ground-truth integer, or the answer scores zero. A run of correct algebra that
ends on the wrong number is a wrong answer to a competition problem.

The point of this corpus is that both sides of that comparison are integers.
Upstream transformed every answer into one — a problem whose answer was
`\\frac{a\\sqrt b}{c}` asks for `a + b + c` instead — so the comparison is
exact rather than a normalisation contest. What a maths grader usually spends
its life on, deciding whether `0.5`, `1/2` and `\\frac{1}{2}` are the same
answer, has no purchase here, and nothing beyond digit grouping is normalised
on purpose: every further rule is a rule that can make two different values
equal.
"""

from __future__ import annotations

import json
import re

GRADER_VERSION = "boxed-integer-v1"
SPEC_SCHEMA = "reliquary/dapo-math-verifier/v1"

# Appended to every problem, in place of DAPO's own `Answer:` contract. The
# repository's maths environment asks for the same thing in the same words, so
# a policy trained against one is not asked to learn a second convention.
ANSWER_INSTRUCTION = "\n\nPut your final answer within \\boxed{}."

# LaTeX spacing, which carries no value: `12\,345` and `12 345` are `12345`.
_SPACING = re.compile(r"\\[,!;:\s]|\s|~")
# A full digit-group thousands separator and nothing else: `29,400` but not
# `1,234,5`, whose last group is short, nor `1,2`, which is a pair.
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+$")
_INTEGER = re.compile(r"^[+-]?\d+$")


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
                return text[start : position + 1]
    # An unclosed brace is an answer the model never finished writing, which is
    # what a completion cut at the token budget looks like.
    return None


def parse_integer(span: str) -> int | None:
    """The integer a boxed span states, or None if it states something else.

    Thousands separators are the one spelling accepted beyond the digits, in
    either the plain or the LaTeX form, because a model writes `1{,}000` as
    readily as `1000` and the separator is typography rather than value.
    """
    for wrapper in ("\\boxed{", "\\fbox{"):
        if span.startswith(wrapper) and span.endswith("}"):
            span = span[len(wrapper) : -1]
            break
    span = span.strip().strip("$")
    span = _SPACING.sub("", span.replace("{,}", ","))
    if _THOUSANDS.match(span):
        span = span.replace(",", "")
    if not _INTEGER.match(span):
        return None
    return int(span)


def verifier_spec(answer: int) -> str:
    """Everything grading needs, as one string the task can carry."""
    return json.dumps(
        {"schema": SPEC_SCHEMA, "grader_version": GRADER_VERSION, "answer": answer},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def grade(spec: str, completion: str | None) -> float:
    """1.0 if the last boxed span is the answer `spec` states, 0.0 otherwise."""
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
    if not isinstance(answer, int) or isinstance(answer, bool):
        # A spec with no answer to compare against would pay for having been
        # stripped of the thing being asked.
        return 0.0

    if not isinstance(completion, str):
        return 0.0
    span = last_boxed_span(completion)
    if span is None:
        # No box, no answer. The instruction asks for one in the prompt, and a
        # number picked out of the prose would reward whichever number the
        # reasoning happened to end on.
        return 0.0
    given = parse_integer(span)
    return 1.0 if given is not None and given == answer else 0.0


def reference_completion(answer: int) -> str:
    """The shortest completion that scores 1.0."""
    return f"\\boxed{{{answer}}}"


__all__ = [
    "ANSWER_INSTRUCTION",
    "GRADER_VERSION",
    "SPEC_SCHEMA",
    "grade",
    "last_boxed_span",
    "parse_integer",
    "reference_completion",
    "verifier_spec",
]
