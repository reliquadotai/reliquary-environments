"""Procedural logic puzzles with exact, typed answers.

Twelve families of generated puzzle — boolean evaluation, bracket balancing,
constraint grids, spatial and temporal reasoning — each with a checker that
compares typed JSON rather than free text. A wrong answer cannot be spelled
right: `52` has one spelling, so the strictness of the envelope is what
removes the grader ambiguity that free-form marking suffers from.

Single turn, no tools, no state. The model reads a puzzle and answers with
one fenced JSON object.

Two surfaces, as this repository requires: `LogicTaskset` for Verifiers and
prime-rl, and `LogicEnvironment` for synchronous JSON replay.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterator
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field

from reliquary_logic.answer import (
    StructuredOutputError,
    canonical_json,
    extract_json_answer,
)
from reliquary_logic.tasks import (
    CHECKER_VERSION,
    GENERATOR_VERSION,
    TASK_FAMILY,
    VIRTUAL_LENGTH,
    GeneratedLogicTask,
    check_answer,
    generate_logic_task,
    verifier_spec,
)

ENVIRONMENT = "reliquary_logic_v2"
SPLITS = ("train", "eval", "qualification")
# Splits interleave rather than share an index space, so no task can appear
# in two of them. The family is drawn from a hash of the index rather than
# cycling with it, so taking every third index leaves the family mix intact —
# `test_splits_are_disjoint_and_unbiased` measures that rather than assuming.
TASK_COUNT = VIRTUAL_LENGTH // len(SPLITS)

REASONING_PROMPT = (
    "Solve the following problem step by step.\n\n"
    "{problem}\n\n"
    "After your reasoning, give the final answer in the last fenced JSON "
    "code block."
)


def _normalize(index: int, split: str) -> int:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return (int(index) % TASK_COUNT) * len(SPLITS) + SPLITS.index(split)


def _render(puzzle: str, family: str, reasoning: frozenset[str] | None) -> str:
    """Wrap the puzzle where reasoning is wanted, serve it bare where not.

    Measured on Qwen3-4B-Base, asking for step-by-step reasoning is worth
    +10.9 points of in-band groups on numbrix and +9.4 on
    dyck_language_errors, costs math_path 14.1, and changes nothing on the
    six families that answer correctly in ten tokens — where it multiplies
    generated tokens by fifteen. `None` means every family, which is the
    default because the band measures the signal available now and says
    nothing about whether reinforcing reasoning broadly pays later.
    """
    if reasoning is not None and family not in reasoning:
        return puzzle
    return REASONING_PROMPT.format(problem=puzzle)


def _build_task(
    index: int, split: str, reasoning: frozenset[str] | None = None
) -> dict[str, Any]:
    position = _normalize(index, split)
    task: GeneratedLogicTask = generate_logic_task(position)
    identity = hashlib.sha256(
        f"{ENVIRONMENT}:{GENERATOR_VERSION}:{split}:{index}".encode("ascii")
    ).hexdigest()[:16]
    return {
        "id": identity,
        "prompt": _render(task.prompt, task.family, reasoning),
        "metadata": {
            "family": task.family,
            "task_family": TASK_FAMILY,
            "operation_id": task.operation_id,
            "difficulty": task.difficulty,
            "generator_version": GENERATOR_VERSION,
            "checker_version": CHECKER_VERSION,
            "split": split,
        },
        "private": {
            "verifier_spec": verifier_spec(task),
            "reference_answer": task.expected,
        },
    }


def _grade(spec: str, completion: str) -> float:
    """Reward for one completion: the checker's verdict, or zero."""
    try:
        parsed = json.loads(spec)
        if not isinstance(parsed, dict):
            return 0.0
        if parsed.get("schema") != "reliquary/logic-verifier/v1":
            return 0.0
        if parsed.get("checker_version") != CHECKER_VERSION:
            return 0.0
        answer = extract_json_answer(completion)
        if set(answer) != {"result"}:
            return 0.0
        return float(check_answer(parsed, answer["result"]))
    except (StructuredOutputError, ValueError, TypeError, RecursionError):
        return 0.0


class LogicEnvironment:
    """Synchronous, JSON-shaped ABI used by Reliquary replay and local tests."""

    name = ENVIRONMENT
    max_turns = 1
    validator_authoritative_reward = True

    def __init__(
        self,
        split: str = "train",
        reasoning_families: tuple[str, ...] | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split
        self.reasoning = (
            None if reasoning_families is None else frozenset(reasoning_families)
        )

    def __len__(self) -> int:
        return TASK_COUNT

    def task(self, index: int) -> dict[str, Any]:
        task = _build_task(index, self.split, self.reasoning)
        return {key: task[key] for key in ("id", "prompt", "metadata")}

    def grade(self, index: int, completion: str) -> dict[str, Any]:
        task = _build_task(index, self.split, self.reasoning)
        reward = _grade(task["private"]["verifier_spec"], completion)
        # `canonical_json` refuses floats — the same bound it puts on a model
        # answer — so the digest carries the verdict, not the score.
        return {
            "reward": reward,
            "success": reward >= 1.0,
            "state_digest": hashlib.sha256(
                canonical_json(
                    {"id": task["id"], "success": reward >= 1.0}
                ).encode("utf-8")
            ).hexdigest(),
        }

    def replay(self, index: int, completion: str) -> dict[str, Any]:
        """Grade the same way a validator would, from the answer alone."""
        return {"reward": self.grade(index, completion)}

    def reference_completion(self, index: int) -> str:
        """The answer that scores 1.0, in the shape the task asks for."""
        task = _build_task(index, self.split, self.reasoning)
        answer = canonical_json({"result": task["private"]["reference_answer"]})
        return f"```json\n{answer}\n```"


class LogicData(vf.TaskData):
    task_id: str
    family: str
    operation_id: str
    difficulty: int
    generator_version: str
    split: Literal["train", "eval", "qualification"]
    verifier_spec: str = Field(repr=False)


class LogicTaskConfig(vf.TaskConfig):
    pass


class LogicTask(vf.Task[LogicData, vf.State, LogicTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.task_id

    @vf.reward(weight=1.0)
    async def verified_answer(self, trace: vf.Trace) -> float:
        return _grade(self.data.verifier_spec, trace.last_reply or "")

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The reference answer scores 1.0 and a perturbed one does not.

        Both halves matter: a checker that accepted anything would pass the
        first on its own.
        """
        del runtime
        index = self.data.idx or 0
        environment = LogicEnvironment(self.data.split)
        good = _grade(
            self.data.verifier_spec, environment.reference_completion(index)
        )
        # A checker that accepted anything would pass the first half alone,
        # so a well-formed wrong answer has to score zero as well.
        bad = _grade(
            self.data.verifier_spec,
            '```json\n{"result": "__not_the_answer__"}\n```',
        )
        return good == 1.0 and bad == 0.0


class LogicConfig(vf.TasksetConfig):
    split: Literal["train", "eval", "qualification"] = "train"
    task: LogicTaskConfig = LogicTaskConfig()
    reasoning_families: tuple[str, ...] | None = None


class LogicTaskset(vf.Taskset[LogicTask, LogicConfig]):
    INFINITE = True

    def load(self) -> Iterator[LogicTask]:
        environment = LogicEnvironment(
            self.config.split, self.config.reasoning_families
        )
        for index in itertools.count():
            task = _build_task(index, self.config.split, environment.reasoning)
            metadata = task["metadata"]
            yield LogicTask(
                LogicData(
                    idx=index,
                    prompt=task["prompt"],
                    network_allow=[],
                    task_id=task["id"],
                    family=metadata["family"],
                    operation_id=metadata["operation_id"],
                    difficulty=metadata["difficulty"],
                    generator_version=metadata["generator_version"],
                    split=self.config.split,
                    verifier_spec=task["private"]["verifier_spec"],
                ),
                self.config.task,
            )


__all__ = ["LogicEnvironment", "LogicTaskset"]
