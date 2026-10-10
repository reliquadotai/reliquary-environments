"""Function calls: the calls an answer makes, compared with the calls expected.

Two call dialects are read. The Qwen3.5 chat template's, which the teacher and
the student both speak:

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Paris
    </parameter>
    </function>
    </tool_call>

and the Hermes JSON form older Qwen templates used,
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`. Arguments written as
text are typed from the tool's JSON schema, as the template wrote them.

Grading, `tool_calls`:

- without `first_of_many`, the answer makes exactly the expected calls, in any
  order (xLAM's parallel calls are unordered);
- with `first_of_many`, the expected call is one of several the reference made
  at once and was kept alone; the answer passes if any of its calls matches it;
- a call matches when the function is the same, every expected argument is
  present and equal, and every extra argument the answer adds is one the schema
  gives a default for, set to that default.

Values compare after normalisation: numbers by value (2 == 2.0), strings without
case or surrounding or repeated whitespace, lists element-wise, objects
key-wise. Strings longer than `FREE_TEXT` characters in the reference are free
prose (an email body, a message to a customer): the answer only has to supply a
non-empty string there, since no two writers produce the same paragraph.

`no_tool_call`: the reference answered in prose; the answer must make no call
and say something.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

FREE_TEXT = 60
CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"
_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_FUNCTION = re.compile(r"\A\s*<function=([^>\n]+)>\s*\n?(.*?)</function>\s*\Z", re.S)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


class Malformed(ValueError):
    """The answer opened a call it did not write in either dialect."""


def _schema(tools: list[dict[str, Any]] | None, name: str) -> Mapping[str, Any]:
    for tool in tools or []:
        if tool.get("name") == name:
            parameters = tool.get("parameters") or {}
            properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
            return properties if isinstance(properties, Mapping) else {}
    return {}


def _coerce(value: str, schema: Any) -> Any:
    declared = schema.get("type") if isinstance(schema, Mapping) else None
    types = declared if isinstance(declared, list) else [declared]
    if declared is None:
        if value[:1] in ("{", "["):
            try:
                parsed = json.loads(value)
            except ValueError:
                return value
            if isinstance(parsed, (dict, list)):
                return parsed
        return value
    if "string" not in types and value in ("None", "null"):
        return None
    for kind in types:
        if kind == "boolean" and value in ("True", "true", "False", "false"):
            return value in ("True", "true")
        if kind == "integer":
            try:
                return int(value)
            except ValueError:
                pass
        if kind == "number":
            try:
                return float(value)
            except ValueError:
                pass
        if kind in ("object", "array"):
            try:
                parsed = json.loads(value)
            except ValueError:
                continue
            if isinstance(parsed, dict if kind == "object" else list):
                return parsed
    return value


def parse_calls(answer: str, tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Every call the answer makes, in order; raises Malformed on a broken one."""
    calls = []
    if answer.count(CALL_OPEN) != answer.count(CALL_CLOSE):
        raise Malformed("unbalanced tool_call tags")
    for block in _BLOCK.findall(answer):
        match = _FUNCTION.match(block)
        if match:
            name = match.group(1).strip()
            schema = _schema(tools, name)
            arguments: dict[str, Any] = {}
            for key, value in _PARAMETER.findall(match.group(2)):
                key = key.strip()
                if key in arguments:
                    raise Malformed(f"parameter {key!r} given twice")
                arguments[key] = _coerce(value, schema.get(key))
            calls.append({"name": name, "arguments": arguments})
            continue
        try:
            payload = json.loads(block)
        except ValueError as error:
            raise Malformed("tool_call block in neither dialect") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            raise Malformed("tool_call JSON without a name")
        arguments = payload.get("arguments", payload.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as error:
                raise Malformed("arguments are not JSON") from error
        if not isinstance(arguments, dict):
            raise Malformed("arguments are not an object")
        calls.append({"name": payload["name"], "arguments": arguments})
    if not calls and "<function=" in answer:
        raise Malformed("function markup outside a tool_call block")
    return calls


def _norm(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _norm(v) for k, v in value.items()}
    return value


def _equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and len(expected) > FREE_TEXT:
        return isinstance(actual, str) and bool(actual.strip())
    if isinstance(expected, str) and not isinstance(actual, str) and isinstance(actual, (int, float)):
        # A number the reference wrote as text ("5") against the same number.
        try:
            return float(expected) == float(actual)
        except ValueError:
            return False
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        return set(expected) == set(actual) and all(_equal(expected[k], actual[k]) for k in expected)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(expected) == len(actual) and all(_equal(e, a) for e, a in zip(expected, actual))
    return _norm(expected) == _norm(actual)


def matches(expected: Mapping[str, Any], actual: Mapping[str, Any], tools: list[dict[str, Any]] | None) -> bool:
    if expected["name"] != actual["name"]:
        return False
    want = expected.get("arguments") or {}
    got = actual.get("arguments") or {}
    for key, value in want.items():
        if key not in got or not _equal(value, got[key]):
            return False
    schema = _schema(tools, expected["name"])
    for key in set(got) - set(want):
        spec = schema.get(key)
        if not isinstance(spec, Mapping) or "default" not in spec or not _equal(spec["default"], got[key]):
            return False
    return True


def grade(check: dict[str, Any], answer: str, tools: list[dict[str, Any]] | None) -> float:
    try:
        calls = parse_calls(answer, tools)
    except Malformed:
        return 0.0
    if check["type"] == "no_tool_call":
        return 1.0 if not calls and answer.strip() else 0.0
    expected = check["calls"]
    if not calls:
        return 0.0
    if check.get("first_of_many"):
        return 1.0 if any(matches(expected[0], call, tools) for call in calls) else 0.0
    if len(calls) != len(expected):
        return 0.0
    remaining = list(calls)
    for want in expected:
        hit = next((i for i, call in enumerate(remaining) if matches(want, call, tools)), None)
        if hit is None:
            return 0.0
        remaining.pop(hit)
    return 1.0


def render_call(call: Mapping[str, Any]) -> str:
    """A call in the Qwen3.5 dialect, the way the chat template writes one."""
    text = f"{CALL_OPEN}\n<function={call['name']}>\n"
    for key, value in (call.get("arguments") or {}).items():
        shown = json.dumps(value, ensure_ascii=False) if isinstance(value, (Mapping, list, tuple)) else str(value)
        text += f"<parameter={key}>\n{shown}\n</parameter>\n"
    return text + f"</function>\n{CALL_CLOSE}"


__all__ = ["FREE_TEXT", "Malformed", "grade", "matches", "parse_calls", "render_call"]
