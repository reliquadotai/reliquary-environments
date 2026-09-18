"""Synchronous, JSON-shaped ABI used by Reliquary replay and local tests.

The task is a competition maths problem and the answer contract this
repository asks for; the answer is whatever the last `\\boxed{}` span holds.
Nothing else in the completion is read, and the reasoning that precedes it is
neither graded nor required to be present.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from reliquary_dapo_math.corpus import SPLITS, Problem, problems
from reliquary_dapo_math.grading import (
    ANSWER_INSTRUCTION,
    GRADER_VERSION,
    grade,
    reference_completion,
    verifier_spec,
)

ENVIRONMENT = "reliquary_dapo_math_v1"
TASK_FAMILY = "dapo_math_v1"


def _identity(problem: Problem) -> str:
    """A task's name follows its problem, not its position.

    Retuning the split shares renumbers the index space. Naming a task after
    the problem it grades means none of that renames a task that has already
    been paid for — and it is the same reason the corpus is deduplicated: one
    problem, one name, one cooldown.
    """
    return hashlib.sha256(
        f"{ENVIRONMENT}:{GRADER_VERSION}:{problem.key}".encode("utf-8")
    ).hexdigest()[:16]


def _build_task(index: int, split: str) -> dict[str, Any]:
    pool = problems(split)
    problem = pool[int(index) % len(pool)]
    return {
        "id": _identity(problem),
        "prompt": problem.problem + ANSWER_INSTRUCTION,
        "metadata": {
            "task_family": TASK_FAMILY,
            "corpus_problem": problem.key,
            "grader_version": GRADER_VERSION,
            "split": split,
        },
        "private": {
            "verifier_spec": verifier_spec(problem.answer),
            "reference_answer": problem.answer,
        },
    }


class DapoMathEnvironment:
    """Synchronous, JSON-shaped ABI used by Reliquary replay and local tests."""

    name = ENVIRONMENT
    max_turns = 1
    validator_authoritative_reward = True

    def __init__(self, split: str = "train") -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split

    def __len__(self) -> int:
        return len(problems(self.split))

    def task(self, index: int) -> dict[str, Any]:
        task = _build_task(index, self.split)
        return {key: task[key] for key in ("id", "prompt", "metadata")}

    def grade(self, index: int, completion: str) -> dict[str, Any]:
        task = _build_task(index, self.split)
        reward = grade(task["private"]["verifier_spec"], completion)
        return {
            "reward": reward,
            "success": reward >= 1.0,
            # The digest carries the verdict, not the answer: two correct
            # completions differ in every token before the box.
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

    def reference_completion(self, index: int) -> str:
        """The answer that scores 1.0, in the shape the task asks for.

        It is a box and nothing else. A worked solution would be a claim about
        how the problem should be solved, and this environment grades only
        where the reasoning arrived.
        """
        task = _build_task(index, self.split)
        return reference_completion(task["private"]["reference_answer"])


__all__ = ["ENVIRONMENT", "TASK_FAMILY", "DapoMathEnvironment"]
