"""Synchronous, JSON-shaped ABI used by Reliquary corpus jobs, replay and tests.

A task is a conversation to continue: `metadata.messages` is the whole prompt —
a single user turn, a frozen multi-turn history, or a tool-use transcript —
ending on the turn the policy answers, with `metadata.tools` the functions it
may call. `prompt` repeats the last user message as a plain string for callers
that can only render one turn; for any row whose `messages` hold more than that
one user turn, rendering `prompt` alone is a different task, and a caller has
to render `messages` (and `tools`) instead.

`metadata.system` is the generation-time system prompt — the Teutonic identity
— and `metadata.system_scope` says what to do with it: `generation-only`, i.e.
send it to the teacher, strip it from the training example. A system message
inside `messages` belongs to the task (a role-play persona, a support agent's
policy) and stays at training. When both are present, the generation system
goes first, then a blank line, then the task's own.

`metadata.mode` and `metadata.renderer` say which thinking mode the row was
drawn for; a corpus job over a segment names that renderer.
`metadata.single_turn` is true when `messages` is one user message and there
are no tools: then `prompt` is the whole task, and a one-turn renderer (with or
without the generation system) serves it faithfully. A segment's single-turn
rows come first (`single_turn_count` in `segments()`).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from reliquary_general.corpus import SPLITS, Row, locate, row, segments
from reliquary_general.corpus import length as split_length
from reliquary_general.grading import GRADER_VERSION, Ungraded, grade

ENVIRONMENT = "reliquary_general_v1"
TASK_FAMILY = "general_v1"
GENERATION_SYSTEM = (
    "You are Teutonic, a helpful AI assistant. When asked about your identity, say that you are Teutonic. "
    "You have no information about who created or trained you, so do not attribute yourself to any company, "
    "research lab or other AI model."
)
SYSTEM_SCOPE = "generation-only"


def _identity(found: Row) -> str:
    """Named after the prompt and its mode, not its position."""
    return hashlib.sha256(
        f"{ENVIRONMENT}:{GRADER_VERSION}:{found.key}:{found.mode}".encode("utf-8")
    ).hexdigest()[:16]


def _last_user(found: Row) -> str:
    for message in reversed(found.messages):
        if message["role"] == "user" and message["content"].strip():
            return message["content"]
    raise RuntimeError(f"row {found.key} has no user turn")


def _build_task(index: int, split: str) -> dict[str, Any]:
    found = row(int(index), split)
    segment, _ = locate(int(index), split)
    return {
        "id": _identity(found),
        "prompt": _last_user(found),
        "metadata": {
            "task_family": TASK_FAMILY,
            "block": found.block,
            "mode": found.mode,
            "renderer": segment.renderer,
            "segment": segment.name,
            "max_new_tokens": segment.max_new_tokens,
            "messages": [dict(m) for m in found.messages],
            "tools": [dict(t) for t in found.tools] if found.tools else None,
            "single_turn": found.single_turn,
            "system": GENERATION_SYSTEM,
            "system_scope": SYSTEM_SCOPE,
            "graded": found.check is not None,
            "check_type": found.check["type"] if found.check else None,
            "grader_version": GRADER_VERSION,
            "corpus_row": found.key,
            "source": found.source,
            "split": split,
        },
        "private": {"check": found.check},
    }


class GeneralPromptsEnvironment:
    """Synchronous, JSON-shaped ABI used by Reliquary replay and local tests."""

    name = ENVIRONMENT
    max_turns = 1
    validator_authoritative_reward = True

    def __init__(self, split: str = "train") -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split

    def __len__(self) -> int:
        return split_length(self.split)

    def segments(self) -> list[dict[str, Any]]:
        """The (block, mode) runs of this split, as a job declaration reads them."""
        return [
            {
                "segment": s.name,
                "block": s.block,
                "mode": s.mode,
                "renderer": s.renderer,
                "prompt_start": s.start,
                "prompt_count": s.count,
                "graded": s.graded,
                "single_turn_count": s.single_turn,
                "max_new_tokens": s.max_new_tokens,
            }
            for s in segments(self.split)
        ]

    def task(self, index: int) -> dict[str, Any]:
        task = _build_task(index, self.split)
        return {key: task[key] for key in ("id", "prompt", "metadata")}

    def grade(self, index: int, completion: str) -> dict[str, Any]:
        """1.0 or 0.0; raises `Ungraded` for a row of a block with no grader."""
        task = _build_task(index, self.split)
        metadata = task["metadata"]
        reward = grade(
            task["private"]["check"],
            completion,
            prompt=task["prompt"],
            tool_specs=metadata["tools"],
        )
        return {
            "reward": reward,
            "success": reward >= 1.0,
            "state_digest": hashlib.sha256(
                json.dumps(
                    {"id": task["id"], "success": reward >= 1.0},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            ).hexdigest(),
        }

    def replay(self, index: int, completion: str) -> dict[str, Any]:
        return {"reward": self.grade(index, completion)}


__all__ = [
    "ENVIRONMENT",
    "GENERATION_SYSTEM",
    "SYSTEM_SCOPE",
    "TASK_FAMILY",
    "GeneralPromptsEnvironment",
    "Ungraded",
]
