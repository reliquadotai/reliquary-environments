"""OpenMathInstruct-2 as one Verifiers taskset and one replay environment.

Two surfaces, as this repository requires: `MathTaskset` for Verifiers and
prime-rl, and `MathEnvironment` for synchronous replay.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from reliquary_math.corpus import corpus_length, get_problem
from reliquary_math.grading import compute_reward

ENVIRONMENT = "reliquary_math_v1"
SPLITS = ("train",)

# Copied from the v5 protocol profile's math template. Core renders this from
# the active profile; a standalone package has no active profile, so it is a
# constructor argument with this default.
DEFAULT_PROMPT = (
    "Solve the following problem step by step.\n\n"
    "{problem}\n\n"
    "Put your final answer within \\boxed{{}}."
)


class MathEnvironment:
    """Synchronous, JSON-shaped ABI used by replay and local tests."""

    name = ENVIRONMENT
    max_turns = 1
    validator_authoritative_reward = True

    def __init__(
        self,
        split: str = "train",
        prompt_template: str = DEFAULT_PROMPT,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split
        self.prompt_template = prompt_template

    def __len__(self) -> int:
        return corpus_length()

    def _row(self, index: int) -> dict[str, Any]:
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} outside the pinned corpus")
        return get_problem(index)

    def task(self, index: int) -> dict[str, Any]:
        row = self._row(index)
        prompt = self.prompt_template.format(problem=row["problem"])
        return {
            "id": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "prompt": prompt,
            "metadata": {"index": int(index), "split": self.split},
        }

    def grade(self, index: int, completion: str) -> dict[str, Any]:
        row = self._row(index)
        reward = compute_reward(row, completion)
        return {
            "reward": reward,
            "success": reward >= 1.0,
            "state_digest": hashlib.sha256(
                json.dumps(
                    {"index": int(index), "success": reward >= 1.0},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }

    def replay(self, index: int, completion: str) -> dict[str, Any]:
        return {"reward": self.grade(index, completion)}

    def reference_completion(self, index: int) -> str:
        """The answer that scores 1.0, in the shape the task asks for."""
        row = self._row(index)
        return f"\\boxed{{{row['expected_answer']}}}"
