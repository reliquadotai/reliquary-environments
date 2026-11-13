"""Terminal-Bench 2.1 as a verifiers taskset, evaluation only, graded offline.

Three splits, all held out (never trained on):

* `eval`: the tasks gated with the agent's box online through controlled egress
  (`gate.VERIFIED`), the split a gateway with egress serves and a Catalyst eval set freezes.
* `eval_offline`: the tasks gated with both boxes offline (`gate.VERIFIED_OFFLINE`), for a
  gateway without egress.
* `eval_all`: all 89 tasks, for the gold check.

A gated split refuses to load until the gold check recorded it; `gate.EXCLUDED` is never
served.

The tasks are vendored from the pinned Harbor dataset (`source`), so the gateway, the
validators and the miners load the same rows offline. Task order is the task id's,
sorted; `task_at(i)` is the `i`-th task `load()` yields.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf

from reliquary_terminal_bench import gate, source
from reliquary_terminal_bench.task import TerminalBenchTask, task_data

SPLITS = ("eval", "eval_offline", "eval_all")
GATES = {"eval": "VERIFIED", "eval_offline": "VERIFIED_OFFLINE"}


class TerminalBenchConfig(vf.TasksetConfig):
    # No default: a source that forgets the split fails loudly.
    split: Literal["eval", "eval_offline", "eval_all"] | None = None
    tasks: list[str] | None = None
    """Optional subset of task ids (None: the whole split). Indices stay the split's."""


@functools.cache
def split_ids(split: str) -> tuple[str, ...]:
    """The task ids of `split`, in order."""
    every = source.task_ids()
    if split == "eval_all":
        return every
    if split not in GATES:
        raise ValueError(f"reliquary-terminal-bench: unknown split {split!r} ({', '.join(SPLITS)})")
    verified = getattr(gate, GATES[split])
    if verified is None:
        raise ValueError(
            f"reliquary-terminal-bench: the {split} split is not gated yet: run "
            "scripts/gold_check.py on a gVisor host and record the verified tasks (gate.py); "
            f"eval_all serves all {len(every)} tasks meanwhile")
    unknown = sorted((verified | set(gate.EXCLUDED)) - set(every))
    if unknown:
        raise ValueError(f"reliquary-terminal-bench: gate names unknown tasks: {unknown}")
    return tuple(t for t in every if t in verified and t not in gate.EXCLUDED)


def task_for(task_id: str, idx: int, split: str, task_config=None) -> TerminalBenchTask:
    return TerminalBenchTask(task_data(task_id, idx, split), task_config)


class TerminalBenchTaskset(vf.Taskset[TerminalBenchTask, TerminalBenchConfig]):
    def _split(self) -> str:
        if self.config.split is None:
            raise ValueError(
                'reliquary-terminal-bench: --taskset.split is required ("eval" for the '
                'tasks gated with egress, "eval_offline" offline, "eval_all" for all 89)')
        return self.config.split

    def _ids(self) -> tuple[str, ...]:
        return split_ids(self._split())

    def __len__(self) -> int:
        return len(self._ids())

    def task_at(self, index: int) -> TerminalBenchTask:
        """Task `index` of the split, built alone: the index-th task `load` yields."""
        ids = self._ids()
        if type(index) is not int or not 0 <= index < len(ids):
            raise IndexError(index)
        return task_for(ids[index], index, self._split(), self.config.task)

    def load(self) -> Iterator[TerminalBenchTask]:
        wanted = None if self.config.tasks is None else set(self.config.tasks)
        for index, task_id in enumerate(self._ids()):
            if wanted is None or task_id in wanted:
                yield task_for(task_id, index, self._split(), self.config.task)


__all__ = ["GATES", "SPLITS", "TerminalBenchConfig", "TerminalBenchTaskset", "split_ids", "task_for"]
