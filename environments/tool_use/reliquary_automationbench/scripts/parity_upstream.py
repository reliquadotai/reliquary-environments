"""Grade the same action lists with upstream AutomationBench and with this port.

    python scripts/parity_upstream.py <AutomationBench checkout at UPSTREAM_COMMIT> [split] [limit] [probe|fuzz]

For each task, the probe episode (`tests/probes.py`; `fuzz`: 40 calls drawn
from the task's own services' endpoints) runs twice: through
upstream's own world, tools and `partial_credit`, as its harness calls them
(wall clock, unseeded ids), and through `AutomationBenchEnvironment`. Prints
each task whose per-assertion verdicts, partial credit or pass differ, then a
summary. Exit status 1 on any difference.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _upstream_episode(task, actions):
    from automationbench.rubric import partial_credit
    from automationbench.runner import compute_allowed_services, strip_none_values
    from automationbench.schema.world import WorldState
    from automationbench.tools.api import api_fetch, api_search, base64_encode

    tools = {"api_fetch": api_fetch, "api_search": api_search, "base64_encode": base64_encode}
    initial = strip_none_values(copy.deepcopy(task.initial_state))
    assertions = [strip_none_values(a) for a in task.assertions]
    world = WorldState(**initial)
    world.meta.allowed_services = compute_allowed_services(initial, assertions, list(task.zapier_tools))
    observations = []
    for action in actions:
        if "final" in action:
            break
        kwargs = {k: v for k, v in action["arguments"].items() if not (isinstance(v, dict) and not v)}
        fn = tools[action["tool"]]
        try:
            observations.append(fn(world, **kwargs) if action["tool"] == "api_fetch" else fn(**kwargs))
        except Exception as error:  # noqa: BLE001
            observations.append(f"Error: {type(error).__name__}: {error}")
    state = {"info": {"assertions": assertions}, "world": world, "initial_state": copy.deepcopy(initial)}
    score = partial_credit(state)
    verdicts = [(r["type"], r["passed"], r["excluded"]) for r in state.get("_assertion_results", [])]
    return score, verdicts, observations


def main(upstream: Path, split: str = "eval", limit: int | None = None, kind: str = "probe") -> int:
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(ROOT / "tests"))
    from probes import fuzz_actions, probe_actions

    from reliquary_automationbench.corpus import rows
    from reliquary_automationbench.environment import AutomationBenchEnvironment
    from reliquary_automationbench.world import build_world, grade, run_tool

    environment = AutomationBenchEnvironment(split)
    pool = rows(split)[:limit]
    differ = passed = same_obs = total_obs = 0
    for index, task in enumerate(pool):
        actions = probe_actions(task) if kind == "probe" else fuzz_actions(task, 40)
        up_score, up_verdicts, up_obs = _upstream_episode(task, actions)
        episode = environment.replay(index, actions)
        world = build_world(task, 0)
        for position, action in enumerate(actions):
            if "final" in action:
                break
            try:
                run_tool(world, task, 0, position, action["tool"], action["arguments"])
            except Exception:  # noqa: BLE001
                pass
        ours = grade(task, world, 0)
        our_verdicts = [(r["type"], r["passed"], r["excluded"]) for r in ours["assertions"]]
        ours_obs = [event["content"] for event in episode["events"]]
        total_obs += len(up_obs)
        same_obs += sum(a == b for a, b in zip(up_obs, ours_obs))
        if (
            up_verdicts != our_verdicts
            or up_score != ours["partial_credit"]
            or (up_score == 1.0) != (episode["reward"]["reward"] == 1.0)
            or ours["partial_credit"] != episode["partial_credit"]
        ):
            differ += 1
            print(json.dumps({"task": task.name, "upstream": [up_score, up_verdicts], "ours": [ours["partial_credit"], our_verdicts]})[:2000])
        passed += up_score == 1.0
    print(
        f"{split} ({kind}): {len(pool)} tasks, {differ} differ; upstream passes {passed}; "
        f"observations byte-identical {same_obs}/{total_obs} (ids and timestamps differ by design)"
    )
    return 1 if differ else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = int(args[2]) if len(args) > 2 and args[2] != "all" else None
    raise SystemExit(main(Path(args[0]).resolve(), args[1] if len(args) > 1 else "eval", limit, *(args[3:4])))
