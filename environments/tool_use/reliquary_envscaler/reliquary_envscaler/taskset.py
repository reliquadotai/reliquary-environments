"""EnvScaler as a native Verifiers taskset, with per-task tool inventories.

The sibling packages declare a fixed set of `@vf.tool` methods because their
worlds share one. EnvScaler does not: 191 worlds carry their own tools, 15 on
the first scenario alone, and the inventory is part of the task rather than of
the environment.

Verifiers allows this, in a way worth stating because it is not obvious from
the base class. Tool servers are per rollout, and `mcp/server.py` runs
`setup_task` *before* `register`:

    await self.setup()
    await self._setup_task_from_channel(...)   # setup_task(task)
    mcp = FastMCP(...)
    self.register(mcp)                          # after

So `register` can read the task its server was given and advertise that
world's tools. `discover_decorated` is bypassed entirely.

Grading is upstream's continuous reward over the checks the agent is asked to
flip, restricted to those false at reset — 15.5% start true, and paying for
them scores a no-op agent 0.166.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterator
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field

from reliquary_envscaler.environment import EnvScalerToolsEnvironment
from reliquary_envscaler.episode import (
    AssistantAction, EpisodeTrace, canonical_json,
)

ENVIRONMENT = "reliquary_envscaler_v2"
# 2,550 RL scenarios, and the band is measured on them. The split is frozen so
# a teacher is never handed a task the evaluation will ask about.
EVAL_SCENARIOS = 250


def _environment() -> EnvScalerToolsEnvironment:
    return EnvScalerToolsEnvironment()


def _index(index: int, split: str) -> int:
    """Evaluation takes the first EVAL_SCENARIOS positions; training the rest."""
    return index if split == "eval" else EVAL_SCENARIOS + index


class EnvScalerState(vf.State):
    # The world is a live instance built by `exec`-ing the release's own class
    # source, so it is deliberately not a serialisable field: it is held beside
    # the state, keyed by rollout, and never crosses the wire.
    world_key: str = ""


_WORLDS: dict[str, Any] = {}


class EnvScalerToolset(vf.Toolset[vf.ToolsetConfig, EnvScalerState]):
    TOOL_PREFIX = None

    async def setup_task(self, task) -> None:
        self._task = task
        environment = _environment()
        index = task.data.scenario_index
        _WORLDS[task.data.task_id] = environment.reset(
            environment.get_task(index), 0).state

    def register(self, mcp) -> None:
        """Advertise this world's tools, not a fixed roster."""
        environment = _environment()
        task = environment.get_task(getattr(self, "_task").data.scenario_index)
        for spec in task.tools:
            mcp.add_tool(
                self._bind(spec.name), name=spec.name,
                description=spec.description or None,
            )

    def _bind(self, name: str):
        async def call(**arguments: Any) -> str:
            environment = _environment()
            task = environment.get_task(self._task.data.scenario_index)
            result = environment.step(
                task, _WORLDS[self._task.data.task_id],
                AssistantAction(kind="tool", tool=name, arguments=arguments),
            )
            return canonical_json([e.to_wire() for e in result.events])

        call.__name__ = name
        return call


class EnvScalerData(vf.TaskData):
    task_id: str
    env_id: str
    scenario_index: int
    checks: int
    split: Literal["train", "eval"]


class EnvScalerTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class EnvScalerTask(vf.Task[EnvScalerData, EnvScalerState, EnvScalerTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.task_id

    @classmethod
    def toolsets(cls, config: EnvScalerTaskConfig) -> list[vf.Toolset]:
        return [EnvScalerToolset(config.tools)]

    @vf.reward(weight=1.0)
    async def verified_outcome(self, trace: vf.Trace) -> float:
        environment = _environment()
        task = environment.get_task(self.data.scenario_index)
        report = environment.grade(
            task, _WORLDS.get(self.data.task_id) or environment.reset(task, 0).state,
            EpisodeTrace(
                schema="reliquary/episode/v1", environment=ENVIRONMENT,
                task_id=task.id, seed=0, events=(), actions=(), tokens=(),
                assistant_spans=(), observation_digests=(),
                termination_reason="finished",
            ),
        )
        return float(report.reward)


class EnvScalerConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    task: EnvScalerTaskConfig = EnvScalerTaskConfig()


class EnvScalerTaskset(vf.Taskset[EnvScalerTask, EnvScalerConfig]):
    def load(self) -> Iterator[EnvScalerTask]:
        environment = _environment()
        for position in itertools.count():
            index = _index(position, self.config.split)
            task = environment.get_task(index)
            yield EnvScalerTask(
                EnvScalerData(
                    idx=index, prompt=task.prompt, network_allow=[],
                    task_id=task.id, env_id=task.private["env_id"],
                    scenario_index=index,
                    checks=len(task.private.get("checks") or ()),
                    split=self.config.split,
                ),
                self.config.task,
            )


__all__ = ["EnvScalerToolsEnvironment", "EnvScalerTaskset"]
