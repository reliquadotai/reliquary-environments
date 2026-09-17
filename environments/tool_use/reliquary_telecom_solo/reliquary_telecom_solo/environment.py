"""Synchronous, JSON-shaped ABI used by Reliquary replay and local tests.

One actor, one ticket, 44 tools, no conversation. The episode ends when the
agent calls `done`, which is upstream's stop tool for its solo agent and the
only way a solo episode can say it is finished: the agent has nobody to say it
to.

The episode is a pure function of the task index and the action list. Nothing
here reads a clock, a random source or a socket — `test_nothing_reachable_is_
non_deterministic` greps the package for all three — so replaying the same
actions returns the same reward and, which matters more, the same transcript.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reliquary_telecom_solo._tau2.data_model import TelecomDB
from reliquary_telecom_solo._tau2.environment import TelecomSoloWorld, to_json_str
from reliquary_telecom_solo._tau2.user_data_model import TelecomUserDB
from reliquary_telecom_solo.corpus import (
    SPLITS,
    Task,
    databases,
    policy,
    rows,
    tool_schemas,
)
from reliquary_telecom_solo.grading import CHECKER_VERSION, grade, transcript_digest

ENVIRONMENT = "reliquary_telecom_solo_v1"
TASK_FAMILY = "telecom_solo_v1"

# Upstream's stop tool, name and return value both. `LLMSoloAgent` adds it to
# the agent's toolset rather than the environment's, because upstream's
# orchestrator owns the loop; here the environment owns the loop, so the tool
# has to be one of its own.
STOP_TOOL = "done"
STOP_TOKEN = "###STOP###"

# The longest reference solution in the corpus is 12 calls, and 95% are 9 or
# fewer; with the stop call that puts the floor for a perfect agent at 13. The
# rest is diagnosis: the workflow policy has the agent read the status bar, the
# network status, the SIM, the APN and the app permissions before it changes
# anything, and a wrong first guess costs a second pass. 40 leaves room for
# three passes and ends an episode that is going nowhere. Upstream allows 100,
# counting both sides of a conversation that solo mode does not have.
MAX_TURNS = 40

# Upstream's `max_errors`, same value and same meaning: cumulative failed tool
# calls, not consecutive ones. Errors are ordinary here — `get_customer_by_
# phone` raises on a number it does not know, and looking is how the agent
# finds out — so an error must not end the episode on its own.
MAX_ERRORS = 10

# Verbatim from τ²-bench `src/tau2/agent/llm_agent.py`: `AGENT_SOLO_
# INSTRUCTION` with its `{stop_function_name}` filled in, and `SYSTEM_PROMPT_
# SOLO` around it. The prompt is part of the environment, not of whatever
# harness runs it, because a rollout priced against this environment has to be
# reproducible from the task alone.
_AGENT_INSTRUCTION = f"""
You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a ticket that contains the user's request.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.
Stop when you consider that you have solved the ticket.
To do so, send a message containing a single tool call to the `{STOP_TOOL}` tool. Do not include any other tool calls in this last message.

Always follow the policy. Always make sure you generate valid JSON only.
""".strip()

_SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<ticket>
{ticket}
</ticket>
""".strip()


def prompt_for(task: Task) -> str:
    """The whole of what the agent is told, upstream's template."""
    return _SYSTEM_PROMPT.format(
        agent_instruction=_AGENT_INSTRUCTION,
        domain_policy=policy(),
        ticket=task.ticket,
    )


def build_world(task: Task) -> TelecomSoloWorld:
    """The world as the ticket found it.

    The shipped databases are the same two files for every task; what differs
    is the setup, a list of calls that breaks something — unseats the SIM,
    turns roaming off, corrupts the MMS configuration. Running them rather than
    shipping 2,285 broken databases is upstream's design, and it is the reason
    the starting state can be pinned by two small digests.
    """
    db, user_db = databases()
    world = TelecomSoloWorld(
        TelecomDB.model_validate(db), TelecomUserDB.model_validate(user_db)
    )
    for call in task.setup:
        world.run_function(call.env_type, call.func_name, call.arguments)
    return world


def _snapshot(world: TelecomSoloWorld) -> dict[str, Any]:
    """Both databases as JSON, for an episode record that has to travel."""
    return {
        "db": world.db.model_dump(mode="json"),
        "user_db": world.user_db.model_dump(mode="json"),
    }


class TelecomSoloEnvironment:
    """Synchronous, JSON-shaped ABI used by Reliquary replay and local tests."""

    name = ENVIRONMENT
    max_turns = MAX_TURNS
    max_errors = MAX_ERRORS
    validator_authoritative_reward = True

    def __init__(self, split: str = "train") -> None:
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
            "id": task.key,
            "prompt": prompt_for(task),
            "tools": [dict(schema) for schema in tool_schemas()],
            "metadata": {
                "task_family": TASK_FAMILY,
                "family": task.family,
                "reward_basis": list(task.reward_basis),
                "assertions": len(task.assertions),
                # The length of the reference solution, which is the closest
                # thing the corpus has to a difficulty dial: a ticket fixed in
                # two calls and one fixed in twelve are not the same exercise.
                "reference_actions": len(task.gold),
                "checker_version": CHECKER_VERSION,
                "split": self.split,
            },
        }

    def reset(self, index: int, seed: int = 0) -> dict[str, Any]:
        """A fresh world for one task.

        `seed` is accepted and ignored, and its being ignored is the point:
        there is nothing in this environment for a seed to vary.
        """
        return {
            "state": {
                "seed": int(seed),
                "world": build_world(self._task(index)),
                "calls": [],
                "errors": 0,
                "stopped": False,
                "closed": False,
            },
            "events": [],
        }

    def call(
        self, state: dict[str, Any], tool: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Run one tool and return what the agent would read.

        `content` is upstream's rendering, byte for byte: `to_json_str` of the
        return value, or `Error: ` and the exception's message. A validator
        reproducing the transcript has to see the same string, so the structure
        around it carries the verdict and never rewrites the string itself.
        """
        if state.get("closed"):
            return {"ok": False, "content": "Error: episode state is closed"}
        if not isinstance(arguments, Mapping):
            return {"ok": False, "content": "Error: arguments must be an object"}
        if tool == STOP_TOOL:
            state["stopped"] = True
            return {"ok": True, "content": STOP_TOKEN}

        world: TelecomSoloWorld = state["world"]
        state["calls"].append({"tool": tool, "arguments": dict(arguments)})
        try:
            if not world.has_tool(tool):
                # Upstream's live environment answers a hallucinated name the
                # same way it answers a failing call, and changes nothing.
                raise ValueError(f"Tool '{tool}' not found.")
            content = to_json_str(world.call(tool, dict(arguments)))
        except Exception as error:  # noqa: BLE001 - upstream reports, never raises
            state["errors"] = int(state["errors"]) + 1
            return {"ok": False, "content": f"Error: {error}"}
        return {"ok": True, "content": content}

    def step(
        self, index: int, state: dict[str, Any], action: Mapping[str, Any]
    ) -> dict[str, Any]:
        del index
        if (
            not isinstance(action, Mapping)
            or set(action) != {"tool", "arguments"}
            or not isinstance(action["tool"], str)
            or not isinstance(action["arguments"], Mapping)
        ):
            state["errors"] = int(state["errors"]) + 1
            return {
                "state": state,
                "events": [
                    {
                        "role": "tool",
                        "name": "__invalid_action__",
                        "content": "Error: expected a tool call",
                    }
                ],
                "done": True,
                "termination_reason": "invalid_action",
            }

        result = self.call(state, action["tool"], action["arguments"])
        done, reason = False, None
        if state["stopped"]:
            done, reason = True, "finished"
        elif state["errors"] >= self.max_errors:
            done, reason = True, "too_many_errors"
        return {
            "state": state,
            "events": [
                {
                    "role": "tool",
                    "name": action["tool"],
                    "content": result["content"],
                }
            ],
            "done": done,
            "termination_reason": reason,
        }

    def grade(
        self,
        index: int,
        state: Mapping[str, Any],
        actions: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del actions
        return grade(self._task(index), state["world"], state["calls"])

    def replay(
        self,
        index: int,
        actions: Sequence[Mapping[str, Any]],
        seed: int = 0,
    ) -> dict[str, Any]:
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
        reward = self.grade(index, state)
        return {
            "task": self.task(index),
            "state": _snapshot(state["world"]),
            "events": events,
            "actions": applied,
            "termination_reason": reason,
            "transcript_digest": transcript_digest(events),
            "reward": reward,
        }

    def reference_actions(self, index: int) -> list[dict[str, Any]]:
        """The reference solution as an action list, with the stop call on the end.

        This is what `validate` replays, and what the packaged goldens record.
        For the 2,253 tasks scored on assertions alone it is one route among
        many; for the 32 scored on actions too it is the only one.
        """
        task = self._task(index)
        return [
            {"tool": action.name, "arguments": dict(action.arguments)}
            for action in task.gold
        ] + [{"tool": STOP_TOOL, "arguments": {}}]

    @staticmethod
    def close(state: dict[str, Any]) -> None:
        state["closed"] = True



__all__ = [
    "ENVIRONMENT",
    "MAX_ERRORS",
    "MAX_TURNS",
    "STOP_TOKEN",
    "STOP_TOOL",
    "TASK_FAMILY",
    "TelecomSoloEnvironment",
    "build_world",
    "prompt_for",
]
