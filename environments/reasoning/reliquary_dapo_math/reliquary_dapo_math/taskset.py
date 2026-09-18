"""Competition maths with integer answers as a native Verifiers taskset.

A problem arrives from `DAPO-Math-17k`, the policy reasons for as long as it
needs, and the reward is whether the last `\\boxed{}` span holds the right
integer. Single turn, no tools, no state.

The budget matters more here than in any sibling: `environment.toml` declares
what the task needs, and a run that grants less measures a difficulty the
corpus does not have.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from reliquary_dapo_math.environment import (
    DapoMathEnvironment,
    _build_task,
)
from reliquary_dapo_math.grading import grade, reference_completion


class DapoMathData(vf.TaskData):
    task_id: str
    corpus_problem: str
    grader_version: str
    split: Literal["train", "eval", "qualification"]
    verifier_spec: str = Field(repr=False)


class DapoMathTaskConfig(vf.TaskConfig):
    pass


class DapoMathTask(vf.Task[DapoMathData, vf.State, DapoMathTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.task_id

    @vf.reward(weight=1.0)
    async def exact_answer(self, trace: vf.Trace) -> float:
        return grade(self.data.verifier_spec, trace.last_reply or "")

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The stated answer scores 1.0 and a neighbouring integer does not.

        Both halves matter: a grader that read any box as correct would pass
        the first on its own, and `answer + 1` is the cheapest wrong answer
        that is still an integer in a box.
        """
        del runtime
        answer = json.loads(self.data.verifier_spec)["answer"]
        good = grade(self.data.verifier_spec, reference_completion(answer))
        bad = grade(self.data.verifier_spec, reference_completion(answer + 1))
        return good == 1.0 and bad == 0.0


class DapoMathConfig(vf.TasksetConfig):
    split: Literal["train", "eval", "qualification"] = "train"
    task: DapoMathTaskConfig = DapoMathTaskConfig()


class DapoMathTaskset(vf.Taskset[DapoMathTask, DapoMathConfig]):
    def load(self) -> Iterator[DapoMathTask]:
        environment = DapoMathEnvironment(self.config.split)
        for index in range(len(environment)):
            task = _build_task(index, self.config.split)
            metadata = task["metadata"]
            yield DapoMathTask(
                DapoMathData(
                    idx=index,
                    prompt=task["prompt"],
                    network_allow=[],
                    task_id=task["id"],
                    corpus_problem=metadata["corpus_problem"],
                    grader_version=metadata["grader_version"],
                    split=self.config.split,
                    verifier_spec=task["private"]["verifier_spec"],
                ),
                self.config.task,
            )


__all__ = ["DapoMathTaskset"]
