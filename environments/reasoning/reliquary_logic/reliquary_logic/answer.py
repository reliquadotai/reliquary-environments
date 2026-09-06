"""Bounded deterministic JSON answer extraction.

Vendored verbatim from reliquadotai/reliquary at f4a6a23ad883,
`reliquary/environment/structured_output.py`, with imports rewritten
for this standalone package.

The bounds are the environment's contract, not defensive coding: no
floats, no non-finite constants, no duplicate keys, bounded depth and
size. A wrong answer cannot be spelled right.
"""

from __future__ import annotations

import json
import re
from typing import Any


MAX_COMPLETION_BYTES = 16 * 1024
MAX_CONTAINER_ITEMS = 64
MAX_DEPTH = 8
MAX_INTEGER_MAGNITUDE = (1 << 53) - 1

# Any language tag, not just `json`. A tag the opener does not recognise
# leaves that block's closing fence to be read as an opener, which pairs it
# with the *answer* block's opening fence and yields an empty body.
_JSON_FENCE = re.compile(
    r"```[A-Za-z0-9_.+-]*[ \t]*\r?\n(?P<body>.*?)\r?\n```",
    flags=re.IGNORECASE | re.DOTALL,
)


class StructuredOutputError(ValueError):
    """A model answer is not in the bounded canonical JSON channel."""


def _reject_float(value: str) -> Any:
    raise StructuredOutputError(f"floating-point values are not allowed: {value}")


def _reject_constant(value: str) -> Any:
    raise StructuredOutputError(f"non-finite values are not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StructuredOutputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_json_value(value: Any, *, depth: int = 0) -> int:
    if depth > MAX_DEPTH:
        raise StructuredOutputError("JSON answer is nested too deeply")
    if value is None or isinstance(value, (str, bool)):
        return 0
    if isinstance(value, int):
        if abs(value) > MAX_INTEGER_MAGNITUDE:
            raise StructuredOutputError("JSON integer exceeds the safe bound")
        return 0
    if isinstance(value, float):
        raise StructuredOutputError("floating-point values are not allowed")
    if isinstance(value, list):
        count = len(value)
        for item in value:
            count += _validate_json_value(item, depth=depth + 1)
            if count > MAX_CONTAINER_ITEMS:
                raise StructuredOutputError("JSON answer contains too many items")
        return count
    if isinstance(value, dict):
        count = len(value)
        for key, item in value.items():
            if not isinstance(key, str):
                raise StructuredOutputError("JSON object keys must be strings")
            count += _validate_json_value(item, depth=depth + 1)
            if count > MAX_CONTAINER_ITEMS:
                raise StructuredOutputError("JSON answer contains too many items")
        return count
    raise StructuredOutputError(f"unsupported JSON value: {type(value).__name__}")


def _brace_spans(text: str) -> list[str]:
    """Brace-balanced top-level spans, quote- and escape-aware."""
    spans: list[str] = []
    depth = start = 0
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start:index + 1])
    return spans


def extract_json_answer(completion: str | None) -> dict[str, Any]:
    """Extract the final fenced JSON object, or one bare whole object.

    Explanatory reasoning may precede a final fenced answer. If no JSON fence
    exists, the entire trimmed completion must be a JSON object. Duplicate
    keys, floats, non-finite constants, excessive depth/size, and trailing
    content fail closed.
    """

    if not isinstance(completion, str):
        raise StructuredOutputError("completion must be a string")
    if len(completion.encode("utf-8")) > MAX_COMPLETION_BYTES:
        raise StructuredOutputError("completion exceeds the byte limit")

    matches = list(_JSON_FENCE.finditer(completion))
    candidates = [m.group("body") for m in matches] or [completion.strip()]

    # Last *valid* block, not merely the last block. A model that reasons
    # before answering writes code fences of its own, and an odd count of
    # them shifts the pairing so the answer block is consumed as an earlier
    # block's terminator. Measured on reliquarylogic under the step-by-step
    # template: 7.4% of unparsed completions — 2.2% of all rollouts — end in
    # a well-formed answer this recovers. Nothing is loosened: whichever
    # block is chosen still passes every check below, and the answer is
    # still compared against the reference, so a wrong answer gains nothing
    # by being readable.
    failure: Exception | None = None
    value = None
    for payload in reversed(candidates):
        if not payload:
            continue
        try:
            parsed = json.loads(
                payload,
                object_pairs_hook=_unique_object,
                parse_float=_reject_float,
                parse_constant=_reject_constant,
            )
        except StructuredOutputError as exc:
            failure = failure or exc
            continue
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            failure = failure or StructuredOutputError("invalid JSON answer")
            del exc
            continue
        if isinstance(parsed, dict):
            value = parsed
            break
        failure = failure or StructuredOutputError("JSON answer must be an object")
    if value is None and matches:
        # An odd number of fence markers — a code block the model opened in
        # its reasoning and never closed — shifts the pairing so the answer
        # never appears as a body of its own. Fall back to brace-balanced
        # spans, and only here: where fences pair correctly the behaviour
        # above is unchanged.
        for payload in reversed(_brace_spans(completion)):
            try:
                parsed = json.loads(
                    payload,
                    object_pairs_hook=_unique_object,
                    parse_float=_reject_float,
                    parse_constant=_reject_constant,
                )
            except (StructuredOutputError, json.JSONDecodeError,
                    TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                value = parsed
                break
    if value is None:
        # An empty candidate is no longer grounds to stop early: a stray
        # closing fence produces one, and the answer sits after it.
        if not any(candidates):
            raise StructuredOutputError("JSON answer is empty")
        raise failure or StructuredOutputError("invalid JSON answer")
    if not isinstance(value, dict):
        raise StructuredOutputError("JSON answer must be an object")
    _validate_json_value(value)
    return value


def canonical_json(value: Any) -> str:
    """Canonical compact representation after applying the same bounds."""

    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


__all__ = [
    "MAX_COMPLETION_BYTES",
    "StructuredOutputError",
    "canonical_json",
    "extract_json_answer",
]
