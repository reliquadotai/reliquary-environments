"""Olympiad maths with closed-form answers as a native Verifiers taskset.

A problem arrives from the AoPS subset of `nvidia/Nemotron-Math-v2`, the policy
reasons for as long as it needs, and the reward is whether the last
`\\boxed{}` span states the reference. Single turn, no tools, no state.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from reliquary_hard_math.environment import HardMathEnvironment, _build_task
from reliquary_hard_math.grading import grade, reference_completion, wrong_answer


class HardMathData(vf.TaskData):
    task_id: str
    corpus_problem: str
    grader_version: str
    tier: str
    split: Literal["train", "eval", "qualification"]
    verifier_spec: str = Field(repr=False)


class HardMathTaskConfig(vf.TaskConfig):
    pass


class HardMathTask(vf.Task[HardMathData, vf.State, HardMathTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.task_id

    @vf.reward(weight=1.0)
    async def equivalent_answer(self, trace: vf.Trace) -> float:
        return grade(self.data.verifier_spec, trace.last_reply or "")

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The reference scores 1.0 and a neighbour of its own shape does not.

        Both halves matter: a grader that read any box as correct would pass
        the first on its own.
        """
        del runtime
        answer = json.loads(self.data.verifier_spec)["answer"]
        good = grade(self.data.verifier_spec, reference_completion(answer))
        bad = grade(self.data.verifier_spec, reference_completion(wrong_answer(answer)))
        return good == 1.0 and bad == 0.0


class HardMathConfig(vf.TasksetConfig):
    split: Literal["train", "eval", "qualification"] = "train"
    task: HardMathTaskConfig = HardMathTaskConfig()


class HardMathTaskset(vf.Taskset[HardMathTask, HardMathConfig]):
    def load(self) -> Iterator[HardMathTask]:
        environment = HardMathEnvironment(self.config.split)
        for index in range(len(environment)):
            task = _build_task(index, self.config.split)
            metadata = task["metadata"]
            yield HardMathTask(
                HardMathData(
                    idx=index,
                    prompt=task["prompt"],
                    network_allow=[],
                    task_id=task["id"],
                    corpus_problem=metadata["corpus_problem"],
                    grader_version=metadata["grader_version"],
                    tier=metadata["tier"],
                    split=self.config.split,
                    verifier_spec=task["private"]["verifier_spec"],
                ),
                self.config.task,
            )


__all__ = ["HardMathTaskset"]
