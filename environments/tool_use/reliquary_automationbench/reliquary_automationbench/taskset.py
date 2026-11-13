"""AutomationBench, api toolset, as a native Verifiers taskset.

The three tools are not a second implementation: every call replays the
episode's calls so far through `AutomationBenchEnvironment`, which is what a
validator runs, so the two surfaces cannot drift apart. Parallel tool calls in
one turn become consecutive actions, in the order the model wrote them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field

from reliquary_automationbench.environment import TOOL_SCHEMAS, AutomationBenchEnvironment

TOOL_NAMES = frozenset(schema["name"] for schema in TOOL_SCHEMAS)
Split = Literal["eval", "qualification", "sales", "marketing", "operations", "support", "finance", "hr", "simple"]


class AutomationBenchState(vf.State):
    index: int = 0
    split: Split = "eval"
    actions: list[dict[str, Any]] = Field(default_factory=list)
    ended: bool = False


class AutomationBenchToolset(vf.Toolset[vf.ToolsetConfig, AutomationBenchState]):
    TOOL_PREFIX = None

    def _run(self, name: str, **arguments: Any) -> str:
        if self.state.ended:
            return "Error: this episode has already ended"
        environment = AutomationBenchEnvironment(self.state.split)
        arguments = {key: value for key, value in arguments.items() if value is not None}
        self.state.actions.append({"tool": name, "arguments": arguments})
        state = environment.reset(self.state.index)["state"]
        result: dict[str, Any] = {}
        for action in self.state.actions:
            result = environment.step(self.state.index, state, action)
        self.state.ended = bool(result["done"])
        return result["events"][-1]["content"]

    @vf.tool
    def api_search(self, query: str, top_k: int | None = None) -> str:
        """Search available API endpoints by keyword.

        Use this to discover which endpoint to call before using api_fetch.
        Ranks results using BM25 over endpoint descriptions and parameter text.
        Returns full endpoint details including the URL to pass to api_fetch.
        """
        return self._run("api_search", query=query, top_k=top_k)

    @vf.tool
    def api_fetch(self, method: str, url: str, params: str | None = None, body: str | None = None) -> str:
        """Call an API endpoint by its full URL, routing to the appropriate world state mutation.

        Use api_search first to discover the correct URL and parameters for an endpoint.
        Returns a JSON string mimicking the real API response.
        """
        return self._run("api_fetch", method=method, url=url, params=params, body=body)

    @vf.tool
    def base64_encode(self, text: str) -> str:
        """Encode text to base64url, the format Gmail API body fields require."""
        return self._run("base64_encode", text=text)


class AutomationBenchData(vf.TaskData):
    task_id: str
    domain: str
    split: Split


class AutomationBenchTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class AutomationBenchTask(vf.Task[AutomationBenchData, AutomationBenchState, AutomationBenchTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.task_id

    @classmethod
    def toolsets(cls, config: AutomationBenchTaskConfig) -> list[vf.Toolset]:
        return [AutomationBenchToolset(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        trace.state.index = self.data.idx or 0
        trace.state.split = self.data.split

    @staticmethod
    def _actions(trace: vf.Trace) -> list[dict[str, Any]] | None:
        branches = trace.branches
        if not branches:
            return None
        actions: list[dict[str, Any]] = []
        for node in branches[-1].nodes:
            if not (node.sampled and isinstance(node.message, vf.AssistantMessage)):
                continue
            for call in node.message.tool_calls or []:
                if call.type != "function" or call.name not in TOOL_NAMES:
                    return None
                try:
                    arguments = json.loads(call.arguments)
                except (TypeError, ValueError, RecursionError):
                    return None
                if not isinstance(arguments, dict):
                    return None
                actions.append({"tool": call.name, "arguments": {k: v for k, v in arguments.items() if v is not None}})
        return actions

    @vf.reward(weight=1.0)
    async def task_completed_correctly(self, trace: vf.Trace) -> float:
        actions = self._actions(trace)
        if actions is None:
            return 0.0
        result = AutomationBenchEnvironment(self.data.split).replay(self.data.idx or 0, [*actions, {"final": ""}])
        return float(result["reward"]["reward"])

    async def validate(self, runtime: vf.Runtime) -> bool:
        """An empty episode does not score (upstream ships no reference solutions)."""
        del runtime
        result = AutomationBenchEnvironment(self.data.split).replay(self.data.idx or 0, [{"final": ""}])
        return result["reward"]["reward"] == 0.0


class AutomationBenchConfig(vf.TasksetConfig):
    split: Split = "eval"
    task: AutomationBenchTaskConfig = AutomationBenchTaskConfig()


class AutomationBenchTaskset(vf.Taskset[AutomationBenchTask, AutomationBenchConfig]):
    def load(self) -> Iterator[AutomationBenchTask]:
        environment = AutomationBenchEnvironment(self.config.split)
        for index in range(len(environment)):
            task = environment.task(index)
            yield AutomationBenchTask(
                AutomationBenchData(
                    idx=index,
                    prompt=task["prompt"],
                    network_allow=[],
                    task_id=task["id"],
                    domain=task["metadata"]["domain"],
                    split=self.config.split,
                ),
                self.config.task,
            )


__all__ = ["AutomationBenchTaskset"]
