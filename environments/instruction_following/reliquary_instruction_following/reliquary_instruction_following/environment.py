"""Synchronous, JSON-shaped ABI used by Reliquary replay and local tests.

The task is a writing request and its constraints; the answer is the reply,
exactly as it arrives. Nothing is stripped and nothing is wrapped, because a
wrapper would be text too — "your last word must be contest" is a claim about
the reply, and an environment that added "Answer:" to it would be grading its
own prose.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from reliquary_instruction_following.corpus import (
    CONSTRAINT_COUNTS,
    SPLITS,
    Row,
    rows,
)
from reliquary_instruction_following.grading import (
    CHECKER_VERSION,
    grade,
    verifier_spec,
)

ENVIRONMENT = "reliquary_instruction_following_v1"
TASK_FAMILY = "instruction_following_v1"


def _identity(row: Row) -> str:
    """A task's name follows its row, not its position.

    Selecting a difficulty band or retuning the split shares renumbers the
    index space. Naming a task after the row it grades means none of that
    renames a task that has already been paid for.
    """
    return hashlib.sha256(
        f"{ENVIRONMENT}:{CHECKER_VERSION}:{row.key}".encode("utf-8")
    ).hexdigest()[:16]


def _build_task(
    index: int, split: str, constraint_counts: tuple[int, ...] | None = None
) -> dict[str, Any]:
    pool = rows(split, constraint_counts)
    row = pool[int(index) % len(pool)]
    return {
        "id": _identity(row),
        "prompt": row.prompt,
        "metadata": {
            "task_family": TASK_FAMILY,
            "corpus_row": row.key,
            "instruction_ids": list(
                constraint.instruction_id for constraint in row.constraints
            ),
            "families": list(row.families),
            # The difficulty dial. One constraint is usually met by accident,
            # three rarely are; a pool where the policy always or never
            # succeeds teaches nothing, so a caller selects the band.
            "constraints": len(row.constraints),
            "checker_version": CHECKER_VERSION,
            "split": split,
        },
        "private": {"verifier_spec": verifier_spec(row.prompt, row.constraints)},
    }


class InstructionFollowingEnvironment:
    """Synchronous, JSON-shaped ABI used by Reliquary replay and local tests.

    There is no `reference_completion` here, and its absence is the point: a
    text that satisfies the constraints is the whole task, nothing in this
    package can write one, and a fabricated one would make the environment
    look verified when it is not. The packaged goldens carry hand-written
    answers instead, which is what freezes the checker.
    """

    name = ENVIRONMENT
    max_turns = 1
    validator_authoritative_reward = True

    def __init__(
        self,
        split: str = "train",
        constraint_counts: tuple[int, ...] | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if constraint_counts is not None:
            constraint_counts = tuple(sorted(set(constraint_counts)))
            if not set(constraint_counts) <= set(CONSTRAINT_COUNTS):
                raise ValueError(
                    f"constraint counts must be drawn from {CONSTRAINT_COUNTS}"
                )
        self.split = split
        self.constraint_counts = constraint_counts

    def __len__(self) -> int:
        return len(rows(self.split, self.constraint_counts))

    def task(self, index: int) -> dict[str, Any]:
        task = _build_task(index, self.split, self.constraint_counts)
        return {key: task[key] for key in ("id", "prompt", "metadata")}

    def grade(self, index: int, completion: str) -> dict[str, Any]:
        task = _build_task(index, self.split, self.constraint_counts)
        reward = grade(task["private"]["verifier_spec"], completion)
        return {
            "reward": reward,
            "success": reward >= 1.0,
            # The digest carries the verdict, not the answer: an answer is
            # free text and two of them can differ while following the same
            # instructions equally well.
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
        """Grade the same way a validator would, from the answer alone."""
        return {"reward": self.grade(index, completion)}


__all__ = ["ENVIRONMENT", "TASK_FAMILY", "InstructionFollowingEnvironment"]
