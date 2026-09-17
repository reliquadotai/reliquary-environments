"""What the ticket was worth, read off the world the agent left behind.

Two components, and the reward is their product, so it is 1.0 or 0.0:

- `env_assertions` — plain-Python predicates over the device and the carrier's
  records. `assert_can_send_mms` walks the same nine conditions the phone walks
  when it actually tries to send one; `assert_data_refueling_amount` compares a
  float to a tolerance. Every telecom task has at least one.
- `actions` — for the 32 tasks whose reward basis names it, every call in the
  reference solution has to be matched by something the agent called. All 32
  are tickets that cannot be solved, and the call they require is
  `transfer_to_human_agents`; the component exists so that giving up correctly
  scores differently from giving up silently.

There is no LLM judge here and no place to put one. That is not a simplification
of τ²-bench: it is what the telecom domain already is. Every other τ² domain
scores on `DB` — a hash of the database against a reference trajectory replayed
by the grader — or on `COMMUNICATE`, which greps prose the agent said to a user.
Telecom scores on predicates, which is why it is the domain worth porting.

An assertion is asked, never trusted to be asked safely: a predicate that trips
over the world it is handed scores zero rather than raising. Upstream raises,
and a raise here would cost a whole window over one rollout that happened to
leave a line in a shape a predicate did not expect.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from reliquary_telecom_solo.corpus import EnvAssertion, Task

CHECKER_VERSION = "tau2-telecom-solo-checker-v1"


def _met(world: Any, assertion: EnvAssertion) -> bool:
    """Whether one predicate holds, upstream's `run_env_assertion` without the raise."""
    try:
        result = world.run_function(
            assertion.env_type, assertion.func_name, assertion.arguments
        )
    except Exception:
        return False
    if not isinstance(result, bool):
        # Upstream raises here, and so would this if it were a packaging fault
        # worth failing on. It is not: every telecom assertion returns bool, and
        # `test_assertions_are_predicates` says so over all 2,285 tasks.
        return False
    return result == assertion.assert_value


def assertion_checks(task: Task, world: Any) -> list[dict[str, Any]]:
    """One row per assertion, named after the predicate it ran."""
    return [
        {
            "name": assertion.func_name,
            "passed": _met(world, assertion),
            "weight": 1.0,
            "detail": assertion.env_type,
        }
        for assertion in task.assertions
    ]


def action_checks(
    task: Task, calls: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """One row per required call, matched against everything the agent called.

    Failed calls count as calls, matching upstream: the check reads the
    trajectory's tool calls and never looks at whether they returned an error.
    """
    return [
        {
            "name": gold.name,
            "passed": any(
                gold.matched_by(call["tool"], call["arguments"]) for call in calls
            ),
            "weight": 1.0,
            "detail": gold.action_id,
        }
        for gold in task.gold
    ]


def grade(
    task: Task, world: Any, calls: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The reward for one finished episode, and every check behind it."""
    components: dict[str, list[dict[str, Any]]] = {
        "env_assertions": assertion_checks(task, world)
    }
    if "ACTION" in task.reward_basis:
        components["actions"] = action_checks(task, calls)

    breakdown = {
        name: float(all(check["passed"] for check in checks))
        for name, checks in components.items()
    }
    reward = 1.0
    for value in breakdown.values():
        reward *= value
    checks = [check for group in components.values() for check in group]
    return {
        "reward": reward,
        "success": reward >= 1.0,
        "checks": checks,
        "reward_breakdown": breakdown,
        "state_digest": world.digest(),
        "checker_version": CHECKER_VERSION,
        "environment_error": None,
    }


def transcript_digest(events: Sequence[Mapping[str, Any]]) -> str:
    """A hash of what the agent read, which is the thing replay has to reproduce.

    The reward is a fact about the final world; the transcript is a fact about
    every observation on the way there. A network that pays by replaying a
    rollout needs both to come back the same, and only the second one notices a
    tool that returns a fresh identifier each time it is called.
    """
    return hashlib.sha256(
        json.dumps(
            [
                [event["role"], event["name"], event["content"]]
                for event in events
            ],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CHECKER_VERSION",
    "action_checks",
    "assertion_checks",
    "grade",
    "transcript_digest",
]
