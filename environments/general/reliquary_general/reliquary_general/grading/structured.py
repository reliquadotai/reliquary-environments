"""Structured outputs: parse the answer in the requested format, validate the schema.

The rule is NVIDIA NeMo Gym's `structured_outputs` resource server (Apache-2.0,
commit 016bb6b8), restated here on the standard library, PyYAML and jsonschema:

- the schema is made strict before validation — every declared property becomes
  required and no undeclared property is allowed, recursively;
- XML is read the way `xmltodict` reads it (children become keys, repeated
  children a list, attributes `@name`, mixed text `#text`, an empty element
  `None`), then leaf strings are coerced to the schema's scalar types and the
  `{"item": [...]}` wrapper an XML list carries is unwrapped at array positions;
- validation is JSON Schema 2020-12 without format assertions, the default of
  the OpenAPI 3.1 validator upstream uses.

Two departures. YAML aliases are refused: they let a few lines stand for an
exponentially large document, and JSON-shaped data never needs them. And one
toward leniency: an answer that is exactly one fenced code block
is read from inside the fence. Upstream parses the raw text, so a model that
fences its JSON scores zero there for formatting the way most chat interfaces
expect; the fence carries no information the schema could check.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Any

import jsonschema
import yaml

FORMATS = ("json", "yaml", "xml")
_FENCE = re.compile(r"\A```[A-Za-z0-9_+-]*[ \t]*\n(?P<body>.*?)\n?```\Z", re.S)


def unfence(text: str) -> str:
    match = _FENCE.match(text.strip())
    return match.group("body") if match else text


def strictify(schema: Any) -> None:
    if isinstance(schema, dict):
        if "properties" in schema:
            schema["required"] = list(schema["properties"])
            schema["additionalProperties"] = False
        for value in schema.values():
            strictify(value)
    elif isinstance(schema, list):
        for value in schema:
            strictify(value)


def _element(node: ET.Element) -> Any:
    children = list(node)
    text = (node.text or "").strip()
    if not children and not node.attrib:
        return text if text else None
    out: dict[str, Any] = {f"@{k}": v for k, v in node.attrib.items()}
    for child in children:
        value = _element(child)
        if child.tag in out:
            if not isinstance(out[child.tag], list):
                out[child.tag] = [out[child.tag]]
            out[child.tag].append(value)
        else:
            out[child.tag] = value
    if text:
        out["#text"] = text
    return out


def parse_xml(text: str) -> Any:
    root = ET.fromstring(text.strip())
    return {root.tag: _element(root)}


def coerce_xml(data: Any, schema: Any) -> Any:
    if not isinstance(schema, dict) or "type" not in schema:
        return data
    kind = schema["type"]
    if kind == "object" and isinstance(data, dict):
        properties = schema.get("properties", {})
        return {k: coerce_xml(v, properties[k]) if k in properties else v for k, v in data.items()}
    if kind == "array":
        items = schema.get("items", {})
        if isinstance(data, dict) and len(data) == 1:
            data = next(iter(data.values()))
        if not isinstance(data, list):
            data = [data] if data is not None else []
        return [coerce_xml(item, items) for item in data]
    if data is None and kind == "string":
        return ""
    if isinstance(data, str):
        try:
            if kind == "integer":
                return int(data)
            if kind == "number":
                return float(data)
            if kind == "boolean":
                lower = data.lower()
                if lower in ("true", "1"):
                    return True
                if lower in ("false", "0"):
                    return False
        except ValueError:
            pass
    return data


class _NoAliasLoader(yaml.SafeLoader):
    """SafeLoader that refuses aliases.

    An alias shares a node, so a few lines of anchors can stand for a document
    of billions of nodes that validation would then walk. JSON has no aliases,
    and no schema here needs one.
    """

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise yaml.YAMLError("aliases are not accepted")
        return super().compose_node(parent, index)


def parse(fmt: str, text: str) -> Any:
    if fmt == "json":
        return json.loads(text)
    if fmt == "yaml":
        return yaml.load(text, Loader=_NoAliasLoader)  # noqa: S506 - a SafeLoader
    if fmt == "xml":
        return parse_xml(text)
    raise ValueError(f"unknown format {fmt!r}")


def schema_of(check: dict[str, Any]) -> dict[str, Any]:
    schema = json.loads(check["schema"])
    strictify(schema)
    return schema


def builds(check: dict[str, Any]) -> bool:
    try:
        if check["format"] not in FORMATS:
            return False
        jsonschema.Draft202012Validator.check_schema(schema_of(check))
    except (KeyError, TypeError, ValueError, jsonschema.SchemaError):
        return False
    return True


def grade(check: dict[str, Any], answer: str) -> float:
    if not answer.strip():
        return 0.0
    schema = schema_of(check)
    try:
        data = parse(check["format"], unfence(answer))
    except Exception:
        return 0.0
    if check["format"] == "xml":
        data = coerce_xml(data, schema)
    validator = jsonschema.Draft202012Validator(schema)
    try:
        return 0.0 if next(validator.iter_errors(data), None) is not None else 1.0
    except Exception:
        return 0.0


__all__ = ["FORMATS", "builds", "grade", "parse", "strictify", "unfence"]
