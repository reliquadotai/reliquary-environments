"""verifiers' Harbor env, grading separate-verifier tasks as `TerminalTask`s.

`HarborEnv.finalize` grades every `train` task in a fresh box through a plain
`HarborTask` it builds itself, so a task subclass's grading never ran there.
This env builds a `TerminalTask` instead -- the same verifier box, the same
`_grade` -- so the grading box is recorded for cleanup and its per-test
results are kept. Starting it also reaps containers that dead processes of
this package left behind (`containers.reap`).
"""

from __future__ import annotations

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor.env import HarborEnv
from verifiers.v1.tasksets.harbor.taskset import HarborTask, verifier_box_data

from reliquary_terminal import containers
from reliquary_terminal.grading import TerminalTask


class TerminalEnv(HarborEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        containers.reap()

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        # HarborEnv.finalize, with the grader built as a TerminalTask.
        if not isinstance(task, HarborTask) or task.data.verifier is None:
            return
        solution = episode.traces[0]
        if not solution.ok:
            return
        grader = TerminalTask(verifier_box_data(task.data))
        scores = await self._grade(self._verifier_config(task), grader, solution)
        items = scores.items() if isinstance(scores, dict) else [("solved", scores)]
        for name, value in items:
            solution.record_reward(name, value)


__all__ = ["TerminalEnv"]
