"""DeepSWE v1.1 (Datacurve) as a verifiers taskset, evaluation only.

Two splits, both held out (never trained on):

* `eval`: the gated tasks (`gate.VERIFIED` less `gate.EXCLUDED`), the split a gateway
  serves and a Catalyst eval set freezes. It refuses to load until the gold check ran.
* `eval_all`: all 113 tasks, for the gold check and conformance runs.

The tasks are vendored at upstream's pinned commit (`source`), so the gateway, the
validators and the miners load the same rows offline. Task order is the task id's
(upstream's directory name), sorted.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf

from reliquary_deepswe import gate, source
from reliquary_deepswe.task import DeepSWETask, task_data

SPLITS = ("eval", "eval_all")


class DeepSWEConfig(vf.TasksetConfig):
    # No default, as reliquary-swe's: a source that forgets the split fails loudly.
    split: Literal["eval", "eval_all"] | None = None
    tasks: list[str] | None = None
    """Optional subset of task ids (None: the whole split). Indices stay the split's."""


@functools.cache
def split_ids(split: str) -> tuple[str, ...]:
    """The task ids of `split`, in order."""
    every = source.task_ids()
    if split == "eval_all":
        return every
    if split != "eval":
        raise ValueError(f"reliquary-deepswe: unknown split {split!r} (eval, eval_all)")
    if gate.VERIFIED is None:
        raise ValueError(
            "reliquary-deepswe: the eval split is not gated yet: run scripts/gold_check.py "
            "on a gVisor host and record the verified tasks (gate.py); eval_all serves all "
            f"{len(every)} tasks meanwhile")
    unknown = sorted(gate.VERIFIED - set(every))
    if unknown:
        raise ValueError(f"reliquary-deepswe: gate.VERIFIED names unknown tasks: {unknown}")
    return tuple(t for t in every if t in gate.VERIFIED and t not in gate.EXCLUDED)


def task_for(task_id: str, idx: int, split: str, task_config=None) -> DeepSWETask:
    toml = source.task_toml(task_id)
    return DeepSWETask(task_data(task_id, idx, split, toml, source.instruction(task_id)),
                       task_config)


class DeepSWETaskset(vf.Taskset[DeepSWETask, DeepSWEConfig]):
    def _split(self) -> str:
        if self.config.split is None:
            raise ValueError(
                'reliquary-deepswe: --taskset.split is required ("eval" for the gated '
                'tasks, "eval_all" for all 113)')
        return self.config.split

    def _ids(self) -> tuple[str, ...]:
        return split_ids(self._split())

    def __len__(self) -> int:
        return len(self._ids())

    def task_at(self, index: int) -> DeepSWETask:
        """Task `index` of the split, built alone: the index-th task `load` yields."""
        ids = self._ids()
        if type(index) is not int or not 0 <= index < len(ids):
            raise IndexError(index)
        return task_for(ids[index], index, self._split(), self.config.task)

    def load(self) -> Iterator[DeepSWETask]:
        wanted = None if self.config.tasks is None else set(self.config.tasks)
        for index, task_id in enumerate(self._ids()):
            if wanted is None or task_id in wanted:
                yield task_for(task_id, index, self._split(), self.config.task)


__all__ = ["SPLITS", "DeepSWEConfig", "DeepSWETaskset", "split_ids", "task_for"]
