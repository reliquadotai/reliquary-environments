"""Science problems with numeric answers as a native Verifiers taskset.

A problem arrives from the `science` subset of `INTELLECT-3-RL`, the policy
reasons for as long as it needs, and the reward is whether the last `\\boxed{}`
span states the reference number to within one percent. Single turn, no tools,
no state.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from reliquary_science.environment import ScienceEnvironment, _build_task
from reliquary_science.grading import grade, reference_completion


class ScienceData(vf.TaskData):
    task_id: str
    corpus_problem: str
    domain: str
    grader_version: str
    split: Literal["train", "eval", "qualification"]
    verifier_spec: str = Field(repr=False)


class ScienceTaskConfig(vf.TaskConfig):
    pass


class ScienceTask(vf.Task[ScienceData, vf.State, ScienceTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.task_id

    @vf.reward(weight=1.0)
    async def numeric_answer(self, trace: vf.Trace) -> float:
        return grade(self.data.verifier_spec, trace.last_reply or "")

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The stated answer scores 1.0 and one five percent away does not.

        Both halves matter: a grader that read any box as correct would pass
        the first alone, and a value just outside the tolerance is the
        cheapest wrong answer that is still a number in a box.
        """
        del runtime
        answer = json.loads(self.data.verifier_spec)["answer"]
        wrong = answer * 1.05 if answer else 1.0
        good = grade(self.data.verifier_spec, reference_completion(answer))
        bad = grade(self.data.verifier_spec, reference_completion(wrong))
        return good == 1.0 and bad == 0.0


class ScienceConfig(vf.TasksetConfig):
    split: Literal["train", "eval", "qualification"] = "train"
    task: ScienceTaskConfig = ScienceTaskConfig()


class ScienceTaskset(vf.Taskset[ScienceTask, ScienceConfig]):
    def load(self) -> Iterator[ScienceTask]:
        environment = ScienceEnvironment(self.config.split)
        for index in range(len(environment)):
            task = _build_task(index, self.config.split)
            metadata = task["metadata"]
            yield ScienceTask(
                ScienceData(
                    idx=index,
                    prompt=task["prompt"],
                    network_allow=[],
                    task_id=task["id"],
                    corpus_problem=metadata["corpus_problem"],
                    domain=metadata["domain"],
                    grader_version=metadata["grader_version"],
                    split=self.config.split,
                    verifier_spec=task["private"]["verifier_spec"],
                ),
                self.config.task,
            )


__all__ = ["ScienceTaskset"]
