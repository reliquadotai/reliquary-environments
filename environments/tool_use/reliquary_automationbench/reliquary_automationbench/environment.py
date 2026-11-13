"""Synchronous, JSON-shaped ABI used by Reliquary replay and local tests.

AutomationBench in upstream's default "api" toolset, the setup Artificial
Analysis reports: three flat tools (`api_search`, `api_fetch`,
`base64_encode`) over a simulated workspace of up to 47 SaaS services. The
episode ends when the agent answers without a tool call, as upstream's
`StatefulToolEnv` does, or at the turn limit. The reward is upstream's strict
pass (`task_completed_correctly`: every scored assertion holds);
`partial_credit` is reported beside it.

The episode is a pure function of the task, the seed and the action list:
the world is rebuilt by replaying the actions, and every id and timestamp the
world mints comes from `_determinism`. A process-local cache keeps the live
world between steps so that an episode is not replayed from the start at
every call; it is keyed by the full action prefix, and a miss rebuilds, so it
can change speed and never a result.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

from reliquary_automationbench.corpus import DOMAINS, SPLITS, Task, rows
from reliquary_automationbench.world import TOOLS, build_world, grade, run_tool, world_json

ENVIRONMENT = "reliquary_automationbench_v1"
TASK_FAMILY = "automationbench_api_v1"
CHECKER_VERSION = "automationbench-1.0.6-strict-pass-v1"

# Upstream caps an episode at 50 model responses, each of which may hold
# several tool calls ("favor parallel tool calls", its system prompt says).
# A Reliquary turn holds one call, so the same work takes more turns: 100
# leaves a sequential agent the room upstream's parallel one has.
MAX_TURNS = 100

# Upstream does not truncate. A tool result here is one event, bounded at
# 64 KiB on the wire after JSON escaping, which can double a JSON body's
# size; 24 KiB of UTF-8 stays under that. Longer results are cut on a
# character boundary with a marker the agent can read.
MAX_CONTENT_BYTES = 24 * 1024

TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "name": "api_search",
        "description": (
            "Search available API endpoints by keyword.\n\n"
            "Use this to discover which endpoint to call before using api_fetch.\n"
            "Ranks results using BM25 over endpoint descriptions and parameter text.\n"
            "Returns full endpoint details including the URL to pass to api_fetch."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        'Space-separated keywords (e.g. "inbox messages", "trash", "send label"). '
                        'Uses API-native terms: "messages" not "emails", "trash" not "delete".'
                    ),
                },
                "top_k": {"type": "integer", "description": "Maximum number of results to return (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "api_fetch",
        "description": (
            "Call an API endpoint by its full URL, routing to the appropriate world state mutation.\n\n"
            "Use api_search first to discover the correct URL and parameters for an endpoint.\n"
            "Returns a JSON string mimicking the real API response."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "description": "HTTP method (GET, POST, PUT, PATCH, DELETE)."},
                "url": {
                    "type": "string",
                    "description": (
                        'Full API URL from api_search results (e.g., "https://api.hubapi.com/crm/v3/objects/contacts").'
                    ),
                },
                "params": {
                    "type": "string",
                    "description": 'Query parameters as a JSON string (e.g., \'{"labelIds": "INBOX"}\').',
                },
                "body": {
                    "type": "string",
                    "description": 'Request body as a JSON string (e.g., \'{"subject": "Hi", "body": "Hello"}\').',
                },
            },
            "required": ["method", "url"],
        },
    },
    {
        "name": "base64_encode",
        "description": (
            "Encode text to base64url — the format Gmail API body fields require.\n\n"
            "The Gmail messages.send / drafts endpoints accept the message body only in "
            "base64url form: either ``raw`` (the full RFC 2822 message, base64url-encoded) "
            "or ``payload.body.data`` (the body text, base64url-encoded). Use this tool to "
            "produce that encoding, then pass the result to api_fetch. This is a local "
            "helper — it does not call any API endpoint."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Plaintext to encode (an email body, or a full RFC 2822 message).",
                }
            },
            "required": ["text"],
        },
    },
)

_REPLAY_REPORT_FIELDS = ("reward", "success", "checks", "state_digest", "environment_error")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def prompt_for(task: Task) -> str:
    """Upstream's system prompt, then the request: what the agent is told."""
    return f"{task.system}\n\n{task.user}"


def _truncate(content: str) -> str:
    raw = content.encode("utf-8")
    if len(raw) <= MAX_CONTENT_BYTES:
        return content
    kept = raw[:MAX_CONTENT_BYTES].decode("utf-8", errors="ignore")
    omitted = len(raw) - len(kept.encode("utf-8"))
    return f"{kept}\n...[truncated: {omitted} more bytes; narrow the request]"


def transcript_digest(events: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(list(events))


class _WorldCache:
    """Live worlds keyed by (split, index, seed, action prefix); popped on use."""

    def __init__(self, size: int = 64) -> None:
        self._size = size
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def pop(self, key: str) -> Any:
        with self._lock:
            return self._items.pop(key, None)

    def put(self, key: str, world: Any) -> None:
        with self._lock:
            self._items[key] = world
            while len(self._items) > self._size:
                self._items.popitem(last=False)


_CACHE = _WorldCache()


class AutomationBenchEnvironment:
    """Synchronous, JSON-shaped ABI used by Reliquary replay and local tests."""

    name = ENVIRONMENT
    max_turns = MAX_TURNS
    validator_authoritative_reward = True

    def __init__(self, split: str = "eval") -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split

    def __len__(self) -> int:
        return len(rows(self.split))

    def _task(self, index: int) -> Task:
        pool = rows(self.split)
        return pool[int(index) % len(pool)]

    def task(self, index: int) -> dict[str, Any]:
        task = self._task(index)
        return {
            "id": task.name,
            "prompt": prompt_for(task),
            "tools": [dict(schema) for schema in TOOL_SCHEMAS],
            "metadata": {
                "task_family": TASK_FAMILY,
                "domain": task.domain,
                "scored": task.domain in DOMAINS,
                "example_id": task.example_id,
                "assertions": len(task.assertions),
                "scenario_time": task.scenario_time.isoformat(),
                "checker_version": CHECKER_VERSION,
                "split": self.split,
            },
        }

    # -- the world behind a state ------------------------------------------------

    def _key(self, index: int, state: Mapping[str, Any]) -> str:
        return _sha256([self.split, int(index) % len(self), int(state["seed"]), state["actions"]])

    def _world(self, index: int, state: Mapping[str, Any]) -> Any:
        """The live world `state` describes: from the cache, or rebuilt by replay."""
        world = _CACHE.pop(self._key(index, state))
        if world is not None:
            return world
        task, seed = self._task(index), int(state["seed"])
        world = build_world(task, seed)
        for position, action in enumerate(state["actions"]):
            if action.get("tool") in TOOLS:
                try:
                    run_tool(world, task, seed, position, action["tool"], action["arguments"])
                except Exception:  # noqa: BLE001 - it failed the first time too
                    pass
        return world

    def reset(self, index: int, seed: int = 0) -> dict[str, Any]:
        state = {"seed": int(seed), "actions": [], "errors": 0, "finished": False, "closed": False}
        _CACHE.put(self._key(index, state), build_world(self._task(index), int(seed)))
        return {"state": state, "events": []}

    def step(self, index: int, state: dict[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
        if state.get("closed") or state.get("finished"):
            raise ValueError("episode state is closed")
        if isinstance(action, Mapping) and set(action) == {"final"}:
            # An answer without a tool call ends the episode, as upstream's does.
            state["finished"] = True
            return {"state": state, "events": [], "done": True, "termination_reason": "finished"}

        is_call = (
            isinstance(action, Mapping)
            and set(action) == {"tool", "arguments"}
            and isinstance(action["tool"], str)
            and isinstance(action["arguments"], Mapping)
        )
        name = action["tool"] if is_call else "__invalid_action__"
        if not is_call or name not in TOOLS:
            # Recorded, so that the positions of later calls (and their ids) do
            # not depend on whether a malformed turn is kept.
            content = "Error: expected a tool call" if not is_call else f"Error: Tool '{name}' not found."
            recorded = {"tool": name, "arguments": {}} if not is_call else {"tool": name, "arguments": dict(action["arguments"])}
            world = self._world(index, state)
            state["actions"].append(recorded)
            state["errors"] = int(state["errors"]) + 1
            _CACHE.put(self._key(index, state), world)
            return self._observation(state, name, content)

        task, seed = self._task(index), int(state["seed"])
        world = self._world(index, state)
        position = len(state["actions"])
        arguments = json.loads(_canonical(dict(action["arguments"])))
        state["actions"].append({"tool": name, "arguments": arguments})
        try:
            content = run_tool(world, task, seed, position, name, arguments)
        except Exception as error:  # noqa: BLE001 - reported to the agent, as upstream's harness does
            state["errors"] = int(state["errors"]) + 1
            content = f"Error: {type(error).__name__}: {error}"
        _CACHE.put(self._key(index, state), world)
        return self._observation(state, name, _truncate(content))

    def _observation(self, state: dict[str, Any], name: str, content: str) -> dict[str, Any]:
        done = len(state["actions"]) >= self.max_turns
        return {
            "state": state,
            "events": [{"role": "tool", "name": name, "content": content}],
            "done": done,
            "termination_reason": "turn_limit" if done else None,
        }

    def grade_report(self, index: int, state: Mapping[str, Any]) -> dict[str, Any]:
        """The full report: upstream's per-assertion results and partial credit."""
        task, seed = self._task(index), int(state["seed"])
        world = self._world(index, state)
        report = grade(task, world, seed)
        digest = _sha256(world_json(world))
        checks = [
            {
                "name": f"{i}:{result['type']}",
                "passed": bool(result["passed"]),
                "weight": 0.0 if result.get("excluded") else 1.0,
                "detail": _canonical(result.get("params", {}))[:512],
            }
            for i, result in enumerate(report["assertions"])
        ]
        if report["error"] is not None:
            checks.append({"name": "assertion_error", "passed": False, "weight": 1.0, "detail": report["error"][:512]})
        return {
            "reward": 1.0 if report["passed"] else 0.0,
            "success": bool(report["passed"]),
            "partial_credit": report["partial_credit"],
            "domain": task.domain,
            "checks": checks,
            "state_digest": digest,
            "environment_error": None,
            "checker_version": CHECKER_VERSION,
        }

    def grade(
        self,
        index: int,
        state: Mapping[str, Any],
        actions: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del actions
        report = self.grade_report(index, state)
        return {field: report[field] for field in _REPLAY_REPORT_FIELDS}

    def replay(self, index: int, actions: Sequence[Mapping[str, Any]], seed: int = 0) -> dict[str, Any]:
        """Run an action list from a fresh world and report what happened."""
        state = self.reset(index, seed)["state"]
        events: list[dict[str, Any]] = []
        applied: list[dict[str, Any]] = []
        reason = "turn_limit"
        for action in actions[: self.max_turns]:
            applied.append(dict(action))
            result = self.step(index, state, action)
            events.extend(result["events"])
            if result["done"]:
                reason = result["termination_reason"]
                break
        report = self.grade_report(index, state)
        return {
            "task": self.task(index),
            "events": events,
            "actions": applied,
            "termination_reason": reason,
            "transcript_digest": transcript_digest(events),
            "reward": {field: report[field] for field in _REPLAY_REPORT_FIELDS},
            "partial_credit": report["partial_credit"],
        }

    @staticmethod
    def close(state: dict[str, Any]) -> None:
        state["closed"] = True


__all__ = [
    "CHECKER_VERSION",
    "ENVIRONMENT",
    "MAX_TURNS",
    "TOOL_SCHEMAS",
    "AutomationBenchEnvironment",
    "prompt_for",
]
