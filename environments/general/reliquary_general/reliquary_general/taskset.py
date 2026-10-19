"""The graded, tool-free blocks as a native Verifiers taskset.

This package is a prompt corpus for distillation first; the Verifiers surface
exposes the part of it a rollout can be scored on without a judge and without
executable tools: verifiable instructions, structured outputs, identity and
clarification. Chat, multi-turn chat and safety have no programmatic grader,
and the tool blocks carry static schemas whose calls a Verifiers harness would
try to execute; both stay corpus-only.

The prompt is the row's messages, and the Teutonic identity prompt is the
system prompt, as at generation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field

from reliquary_general.corpus import segments
from reliquary_general.environment import GENERATION_SYSTEM, _build_task
from reliquary_general.grading import builds, grade

TASKSET_BLOCKS = ("ifeval", "structured", "identity", "clarification")


class GeneralData(vf.TaskData):
    task_id: str
    block: str
    mode: str
    split: Literal["train", "eval", "qualification"]
    check: dict[str, Any] = Field(repr=False)
    last_user: str = Field(repr=False)


class GeneralTaskConfig(vf.TaskConfig):
    pass


class GeneralTask(vf.Task[GeneralData, vf.State, GeneralTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.task_id

    @vf.reward(weight=1.0)
    async def passes_check(self, trace: vf.Trace) -> float:
        return grade(self.data.check, trace.last_reply or "", prompt=self.data.last_user)

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The check builds, and an empty answer earns nothing.

        There is no gold answer to check: writing one is the task.
        """
        del runtime
        return builds(self.data.check, self.data.last_user) and (
            grade(self.data.check, "", prompt=self.data.last_user) == 0.0
        )


class GeneralConfig(vf.TasksetConfig):
    split: Literal["train", "eval", "qualification"] = "train"
    blocks: tuple[str, ...] = TASKSET_BLOCKS
    task: GeneralTaskConfig = GeneralTaskConfig()


class GeneralTaskset(vf.Taskset[GeneralTask, GeneralConfig]):
    def load(self) -> Iterator[GeneralTask]:
        unknown = set(self.config.blocks) - set(TASKSET_BLOCKS)
        if unknown:
            raise ValueError(f"blocks {sorted(unknown)} are corpus-only")
        for segment in segments(self.config.split):
            if segment.block not in self.config.blocks:
                continue
            for index in range(segment.start, segment.stop):
                task = _build_task(index, self.config.split)
                metadata = task["metadata"]
                yield GeneralTask(
                    GeneralData(
                        idx=index,
                        prompt=metadata["messages"],
                        system_prompt=GENERATION_SYSTEM,
                        network_allow=[],
                        task_id=task["id"],
                        block=metadata["block"],
                        mode=metadata["mode"],
                        split=self.config.split,
                        check=task["private"]["check"],
                        last_user=task["prompt"],
                    ),
                    self.config.task,
                )


__all__ = ["TASKSET_BLOCKS", "GeneralTaskset"]
