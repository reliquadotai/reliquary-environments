"""Programmatic graders, one per check type, and the dispatch between them.

Nothing here calls a model. A block whose quality only a judge could assess —
open chat, multi-turn chat, safety — carries no check, and `grade` refuses it
rather than inventing a score: a corpus job over those rows is declared with
`filter: null` and keeps every completion.
"""

from __future__ import annotations

from typing import Any

from reliquary_general.grading import clarify, identity, ifeval, structured, tools
from reliquary_general.grading.answer import answer_text

GRADER_VERSION = "general-graders-v1"
CHECK_TYPES = ("ifeval", "structured", "identity", "clarify", "tool_calls", "no_tool_call")


class Ungraded(ValueError):
    """The task has no programmatic grader; its job must not declare a filter."""


def builds(check: dict[str, Any] | None, prompt: str) -> bool:
    """Whether a check can grade anything at all (the loader refuses one that cannot)."""
    if check is None:
        return True
    kind = check.get("type")
    if kind == "ifeval":
        return ifeval.builds(check, prompt)
    if kind == "structured":
        return structured.builds(check)
    if kind == "identity":
        return True
    if kind == "clarify":
        return check.get("expect") in ("ask", "answer") and check.get("slot") in clarify.SLOT_CUES
    if kind == "tool_calls":
        return bool(check.get("calls")) and all(isinstance(c.get("name"), str) for c in check["calls"])
    if kind == "no_tool_call":
        return True
    return False


def grade(
    check: dict[str, Any] | None,
    completion: str | None,
    *,
    prompt: str = "",
    tool_specs: list[dict[str, Any]] | None = None,
) -> float:
    """1.0 or 0.0 for one completion; raises Ungraded when there is no check."""
    if check is None:
        raise Ungraded("this task has no programmatic grader")
    answer = answer_text(completion)
    kind = check["type"]
    if kind == "ifeval":
        return ifeval.grade(check, prompt, answer)
    if kind == "structured":
        return structured.grade(check, answer)
    if kind == "identity":
        return identity.grade(check, prompt, answer)
    if kind == "clarify":
        return clarify.grade(check, answer)
    if kind in ("tool_calls", "no_tool_call"):
        return tools.grade(check, answer, tool_specs)
    raise ValueError(f"unknown check type {kind!r}")


__all__ = ["CHECK_TYPES", "GRADER_VERSION", "Ungraded", "answer_text", "builds", "grade"]
