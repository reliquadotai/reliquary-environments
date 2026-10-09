"""Competitive programming as a native Verifiers taskset. Single turn, no tools."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf

from reliquary_competitive_code.environment import CompetitiveCodeEnvironment


class CompetitiveCodeData(vf.TaskData):
    task_id: str
    problem_id: str
    judge_version: str
    time_limit_s: float
    split: Literal["train", "eval", "qualification"]


class CompetitiveCodeTaskConfig(vf.TaskConfig):
    pass


class CompetitiveCodeTask(vf.Task[CompetitiveCodeData, vf.State, CompetitiveCodeTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.task_id

    def _environment(self) -> CompetitiveCodeEnvironment:
        return CompetitiveCodeEnvironment(self.data.split)

    @vf.reward(weight=1.0)
    async def passes_all_tests(self, trace: vf.Trace) -> float:
        graded = await asyncio.to_thread(self._environment().grade, self.data.idx, trace.last_reply or "")
        return graded["reward"]

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The reference passes and a program printing nothing useful does not."""
        del runtime
        environment = self._environment()
        good = await asyncio.to_thread(environment.grade, self.data.idx, environment.reference_completion(self.data.idx))
        bad = await asyncio.to_thread(environment.grade, self.data.idx, "```python\nprint('reliquary-wrong')\n```")
        return good["reward"] == 1.0 and bad["reward"] == 0.0


class CompetitiveCodeConfig(vf.TasksetConfig):
    split: Literal["train", "eval", "qualification"] = "train"
    task: CompetitiveCodeTaskConfig = CompetitiveCodeTaskConfig()


class CompetitiveCodeTaskset(vf.Taskset[CompetitiveCodeTask, CompetitiveCodeConfig]):
    def load(self) -> Iterator[CompetitiveCodeTask]:
        environment = CompetitiveCodeEnvironment(self.config.split)
        for index in range(len(environment)):
            task = environment.task(index)
            metadata = task["metadata"]
            yield CompetitiveCodeTask(
                CompetitiveCodeData(
                    idx=index,
                    prompt=task["prompt"],
                    network_allow=[],
                    task_id=task["id"],
                    problem_id=metadata["problem_id"],
                    judge_version=metadata["judge_version"],
                    time_limit_s=metadata["time_limit_s"],
                    split=self.config.split,
                ),
                self.config.task,
            )


__all__ = ["CompetitiveCodeTaskset"]
