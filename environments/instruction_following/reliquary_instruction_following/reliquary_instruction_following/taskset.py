"""Verifiable instruction following as a native Verifiers taskset.

A writing request arrives carrying one to three formal constraints, and the
reply scores 1 only if every one of them checks out. What makes the reward
exact is that the constraints are machine-checkable claims about the text —
which word it starts with, how many paragraphs it has, which words it never
uses — rather than an opinion about how well it wrote.

Single turn, no tools, no state. The reply is the answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from reliquary_instruction_following.corpus import rows
from reliquary_instruction_following.environment import (
    ENVIRONMENT,
    InstructionFollowingEnvironment,
    _build_task,
)
from reliquary_instruction_following.grading import builds, grade


class InstructionFollowingData(vf.TaskData):
    task_id: str
    corpus_row: str
    instruction_ids: tuple[str, ...]
    # The difficulty dial, on the task rather than in the config, so a caller
    # can band a mixed run after the fact as well as select one before it.
    constraints: int
    split: Literal["train", "eval", "qualification"]
    verifier_spec: str = Field(repr=False)


class InstructionFollowingTaskConfig(vf.TaskConfig):
    pass


class InstructionFollowingTask(
    vf.Task[InstructionFollowingData, vf.State, InstructionFollowingTaskConfig]
):
    @property
    def key(self) -> str:
        return self.data.task_id

    @vf.reward(weight=1.0)
    async def followed_instructions(self, trace: vf.Trace) -> float:
        return grade(self.data.verifier_spec, trace.last_reply or "")

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Every constraint builds, and an empty answer earns nothing.

        Its sibling environments check a gold answer here. This one has none to
        check: writing a text that satisfies the constraints is the task, so
        what can be verified without one is that every verifier resolves and
        builds, and that the checker fails closed on an answer that says
        nothing.
        """
        del runtime
        return builds(self.data.verifier_spec) and (
            grade(self.data.verifier_spec, "") == 0.0
        )


class InstructionFollowingConfig(vf.TasksetConfig):
    split: Literal["train", "eval", "qualification"] = "train"
    task: InstructionFollowingTaskConfig = InstructionFollowingTaskConfig()
    constraint_counts: tuple[int, ...] | None = None


class InstructionFollowingTaskset(
    vf.Taskset[InstructionFollowingTask, InstructionFollowingConfig]
):
    def load(self) -> Iterator[InstructionFollowingTask]:
        environment = InstructionFollowingEnvironment(
            self.config.split, self.config.constraint_counts
        )
        for index in range(len(environment)):
            task = _build_task(
                index, self.config.split, environment.constraint_counts
            )
            metadata = task["metadata"]
            yield InstructionFollowingTask(
                InstructionFollowingData(
                    idx=index,
                    prompt=task["prompt"],
                    network_allow=[],
                    task_id=task["id"],
                    corpus_row=metadata["corpus_row"],
                    instruction_ids=tuple(metadata["instruction_ids"]),
                    constraints=metadata["constraints"],
                    split=self.config.split,
                    verifier_spec=task["private"]["verifier_spec"],
                ),
                self.config.task,
            )


__all__ = ["InstructionFollowingTaskset"]
