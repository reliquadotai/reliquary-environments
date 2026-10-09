"""Write the packaged goldens and the artifact manifest that pins them.

Run after any change to the package's files: the goldens freeze one train task
per block, with a completion its grader must pass and one it must fail (or, for
an ungraded block, the fact that grading refuses it), and `artifact.json` pins
the digest of every file the wheel ships, `environment.toml` included through
`source_manifest_sha256`.

    uv run python scripts/write_goldens.py

The passing completions are built, not invented: the expected call rendered in
the template's dialect, a minimal instance of the schema, an answer that meets a
constraint the row asks for. The row each golden uses is the first in its
segment whose check a builder below can satisfy, so the choice is reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from reliquary_general.corpus import BLOCKS, segments
from reliquary_general.environment import ENVIRONMENT, GeneralPromptsEnvironment, _build_task
from reliquary_general.grading import Ungraded, grade
from reliquary_general.grading.structured import schema_of
from reliquary_general.grading.tools import render_call

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "reliquary_general"
SCAN = 400

# One-constraint rows whose constraint a fixed answer can meet.
IFEVAL_ANSWERS = {
    "punctuation:no_comma": "Here is a short answer that never needs a single comma.",
    "detectable_format:title": "<<A short answer>>\n\nHere is the answer in one paragraph.",
    "startend:quotation": '"Here is the whole answer, wrapped in double quotes."',
    "detectable_format:json_format": '```json\n{"answer": "here it is"}\n```',
}


def instance(schema: Any) -> Any:
    """A minimal value of a (strictified) schema, or raise if it needs more."""
    if not isinstance(schema, dict):
        raise ValueError("schema")
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    if kind == "object":
        return {k: instance(v) for k, v in (schema.get("properties") or {}).items()}
    if kind == "array":
        return [instance(schema.get("items", {})) for _ in range(schema.get("minItems", 0))]
    if kind == "string":
        if any(k in schema for k in ("pattern", "format")):
            raise ValueError("string with pattern or format")
        return "x" * max(1, schema.get("minLength", 1))
    if kind == "integer":
        return int(schema.get("minimum", schema.get("exclusiveMinimum", -1) + 1))
    if kind == "number":
        return float(schema.get("minimum", 0.0))
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    raise ValueError(f"type {kind!r}")


def _as_xml(tag: str, value: Any) -> str:
    if isinstance(value, dict):
        return f"<{tag}>" + "".join(_as_xml(k, v) for k, v in value.items()) + f"</{tag}>"
    if isinstance(value, list):
        return f"<{tag}>" + "".join(_as_xml("item", v) for v in value) + f"</{tag}>"
    if value is True:
        return f"<{tag}>true</{tag}>"
    return f"<{tag}>{'' if value is None else value}</{tag}>"


def passing(task: dict[str, Any]) -> str | None:
    check = task["private"]["check"]
    kind = check["type"]
    if kind == "tool_calls" and not check.get("first_of_many"):
        return "I'll look that up.\n\n" + "\n".join(render_call(c) for c in check["calls"])
    if kind == "tool_calls":
        return render_call(check["calls"][0])
    if kind == "no_tool_call":
        return "None of the available functions can do this, so here is a direct answer instead."
    if kind == "identity":
        return "I'm Teutonic, an AI assistant. I don't have any details about who built or trained me."
    if kind == "clarify":
        if check["expect"] == "ask":
            cue = "text" if check["slot"] == "content" else "language"
            return f"Happy to help. Which {cue} should I work with? Please share it."
        return "Here is the result you asked for."
    if kind == "ifeval":
        ids = [c["id"] for c in check["constraints"]]
        return IFEVAL_ANSWERS.get(ids[0]) if len(ids) == 1 else None
    if kind == "structured":
        try:
            value = instance(schema_of(check))
        except ValueError:
            return None
        if check["format"] == "json":
            return json.dumps(value)
        if check["format"] == "xml":
            (root, body), = value.items()
            return _as_xml(root, body)
        return None
    return None


def failing(task: dict[str, Any]) -> str:
    kind = task["private"]["check"]["type"]
    if kind in ("tool_calls",):
        return "I can't help with that."
    if kind == "no_tool_call":
        return render_call({"name": "not_a_tool", "arguments": {}})
    if kind == "identity":
        return "I'm Qwen, a large language model created by Alibaba Cloud."
    if kind == "clarify":
        if task["private"]["check"]["expect"] == "ask":
            return "Sure, here is the result: " + "words " * 400
        return "Could you please provide the text you'd like me to work on?"
    return ""


def golden(block: str) -> dict[str, Any]:
    environment = GeneralPromptsEnvironment("train")
    for segment in segments("train"):
        if segment.block != block:
            continue
        for index in range(segment.start, min(segment.stop, segment.start + SCAN)):
            task = _build_task(index, "train")
            entry = {
                "block": block,
                "index": index,
                "mode": task["metadata"]["mode"],
                "prompt_sha256": hashlib.sha256(
                    json.dumps(task["metadata"]["messages"], sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "segment": task["metadata"]["segment"],
                "split": "train",
                "task_id": task["id"],
            }
            if task["private"]["check"] is None:
                try:
                    environment.grade(index, "anything")
                except Ungraded:
                    return {**entry, "graded": False}
                raise AssertionError(f"{block}: an ungraded row scored")
            good = passing(task)
            if good is None:
                continue
            bad = failing(task)
            assert environment.grade(index, good)["reward"] == 1.0, (block, index, good)
            assert environment.grade(index, bad)["reward"] == 0.0, (block, index, bad)
            return {
                **entry,
                "graded": True,
                "check_type": task["private"]["check"]["type"],
                "completion": good,
                "thinking_completion": "Let me work out the reply.\n</think>\n\n" + good,
                "wrong_completion": bad,
                "state_digest": environment.grade(index, good)["state_digest"],
            }
    raise AssertionError(f"no golden candidate in {block}")


def main() -> None:
    goldens = [golden(block) for block in BLOCKS]
    (PACKAGE / "goldens").mkdir(exist_ok=True)
    (PACKAGE / "goldens" / "reference.jsonl").write_text(
        "".join(json.dumps(g, sort_keys=True, ensure_ascii=True) + "\n" for g in goldens)
    )
    files = sorted(
        path
        for path in PACKAGE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.name != "artifact.json"
    )
    manifest = {
        "schema": "reliquary/environment-artifact/v1",
        "environment": ENVIRONMENT,
        "contract": "reliquary/checked-answer/v1",
        "distribution": {"name": "reliquary-general", "version": "0.1.0a1"},
        "entrypoints": {
            "taskset": "reliquary_general:GeneralTaskset",
            "replay": "reliquary_general:GeneralPromptsEnvironment",
        },
        "source_manifest_sha256": hashlib.sha256((ROOT / "environment.toml").read_bytes()).hexdigest(),
        "files": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
    }
    (PACKAGE / "artifact.json").write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"{len(goldens)} goldens, {len(files)} files pinned")


if __name__ == "__main__":
    main()
