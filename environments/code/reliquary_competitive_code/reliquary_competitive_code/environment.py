"""Synchronous, JSON-shaped ABI used by Reliquary replay and local tests.

The task is a competitive-programming statement and the contract below; the
submission is the last fenced Python block; the reward is 1.0 when it passes
every hidden test of the problem and 0.0 otherwise.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from reliquary_competitive_code.corpus import Corpus, pinned_corpus
from reliquary_competitive_code.extraction import extract_program
from reliquary_competitive_code.judge import judge
from reliquary_competitive_code.layout import SPLITS

ENVIRONMENT = "reliquary_competitive_code_v1"
TASK_FAMILY = "competitive_programming_stdio_v1"
JUDGE_VERSION = "stdio-tokens-v1"
INSTRUCTION = (
    "\n\nWrite a complete Python 3 program that reads the input from standard "
    "input and prints the answer to standard output. Use input() or sys.stdin; "
    "the os and io modules and file access are not available. Put the whole "
    "program in a single ```python code block at the end of your answer."
)


def render_prompt(statement: str) -> str:
    return statement.strip() + INSTRUCTION


class CompetitiveCodeEnvironment:
    name = ENVIRONMENT
    max_turns = 1
    validator_authoritative_reward = True

    def __init__(self, split: str = "train", corpus: Corpus | None = None) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split
        self._corpus = corpus if corpus is not None else pinned_corpus()

    def __len__(self) -> int:
        return len(self._corpus.problems(self.split))

    def _problem(self, index: int):
        problems = self._corpus.problems(self.split)
        if not problems:
            raise ValueError(f"the {self.split} split is empty")
        return int(index) % len(problems), problems[int(index) % len(problems)]

    def _identity(self, problem_id: str) -> str:
        return hashlib.sha256(f"{ENVIRONMENT}:{JUDGE_VERSION}:{problem_id}".encode()).hexdigest()[:16]

    def task(self, index: int) -> dict[str, Any]:
        _, problem = self._problem(index)
        return {
            "id": self._identity(problem.problem_id),
            "prompt": render_prompt(problem.statement),
            "metadata": {
                "task_family": TASK_FAMILY,
                "problem_id": problem.problem_id,
                "judge_version": JUDGE_VERSION,
                "time_limit_s": problem.time_limit_s,
                "split": self.split,
            },
        }

    def grade(self, index: int, completion: str) -> dict[str, Any]:
        position, problem = self._problem(index)
        tests = self._corpus.tests(self.split, position)
        if not tests:
            raise RuntimeError(f"problem {problem.problem_id} has no tests")
        verdict = judge(
            extract_program(completion or ""),
            tests,
            time_limit_s=problem.time_limit_s,
        )
        reward = 1.0 if verdict.passed else 0.0
        return {
            "reward": reward,
            "success": verdict.passed,
            "status": verdict.status,
            # The digest carries the verdict, not the program: two correct
            # programs differ in every character.
            "state_digest": hashlib.sha256(
                json.dumps(
                    {"id": self._identity(problem.problem_id), "success": verdict.passed},
                    sort_keys=True, separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }

    def replay(self, index: int, completion: str) -> dict[str, Any]:
        return {"reward": self.grade(index, completion)}

    def reference_completion(self, index: int) -> str:
        _, problem = self._problem(index)
        return f"```python\n{self._corpus.reference(problem.problem_id)}\n```"


__all__ = ["ENVIRONMENT", "INSTRUCTION", "JUDGE_VERSION", "TASK_FAMILY", "CompetitiveCodeEnvironment", "render_prompt"]
