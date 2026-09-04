import asyncio
import hashlib
import importlib.resources
import json
from pathlib import Path

import verifiers.v1 as vf

import reliquary_stateful_tools as package
from reliquary_stateful_tools.taskset import (
    StatefulToolsEnvironment,
    StatefulToolsState,
)


def _trace(task: vf.Task, actions: list[dict], state: dict | None = None) -> vf.Trace:
    environment = StatefulToolsEnvironment()
    world = environment.reset(0, 0)["state"]
    nodes = []
    parent = None
    for position, action in enumerate(actions):
        if "tool" in action:
            call_id = f"call-{position}"
            nodes.append(
                vf.MessageNode(
                    parent=parent,
                    message=vf.AssistantMessage(
                        tool_calls=[
                            vf.ToolCall(
                                id=call_id,
                                name=action["tool"],
                                arguments=json.dumps(action["arguments"]),
                            )
                        ]
                    ),
                    sampled=True,
                )
            )
            parent = len(nodes) - 1
            result = environment.call(world, action["tool"], action["arguments"])
            nodes.append(
                vf.MessageNode(
                    parent=parent,
                    message=vf.ToolMessage(
                        tool_call_id=call_id,
                        name=action["tool"],
                        content=json.dumps(result),
                    ),
                )
            )
            parent = len(nodes) - 1
        else:
            nodes.append(
                vf.MessageNode(
                    parent=parent,
                    message=vf.AssistantMessage(content=action["final"]),
                    sampled=True,
                )
            )
            parent = len(nodes) - 1
    return vf.Trace(
        task=vf.TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        ),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        state=StatefulToolsState(world=world if state is None else state),
    )


def test_reference_goldens_and_rejections() -> None:
    rows = [
        json.loads(line)
        for line in importlib.resources.files("reliquary_stateful_tools")
        .joinpath("goldens/reference.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    for golden in rows:
        environment = StatefulToolsEnvironment(golden["split"])
        private = __import__("reliquary_stateful_tools.taskset", fromlist=["_build_task"])._build_task(golden["index"], golden["split"])["private"]
        replay = environment.replay(golden["index"], private["reference_actions"])
        assert replay["task"]["id"] == golden["task_id"]
        assert replay["task"]["metadata"]["family"] == golden["family"]
        assert replay["reward"]["reward"] == 1.0
        assert replay["reward"]["state_digest"] == golden["state_digest"]
        assert environment.replay(golden["index"], [{"final": "done"}])["reward"]["reward"] == 0.0

    environment = StatefulToolsEnvironment()
    task = __import__("reliquary_stateful_tools.taskset", fromlist=["_build_task"])._build_task(0, "train")
    actions = list(task["private"]["reference_actions"])
    actions.insert(2, {"tool": "update_shipping_address", "arguments": {"order_id": task["private"]["other_order_id"], "address": "1 Collateral Damage Road"}})
    assert environment.replay(0, actions)["reward"]["reward"] == 0.0


def test_verifiers_load_and_wire_action_replay() -> None:
    exported_tasksets = [item for name in package.__all__ if isinstance((item := getattr(package, name)), type) and issubclass(item, vf.Taskset)]
    assert exported_tasksets == [package.StatefulToolsTaskset]
    config_type = vf.taskset_config_type("reliquary-stateful-tools")
    taskset = vf.load_taskset(config_type(id="reliquary-stateful-tools"))
    task = next(iter(taskset))
    source = __import__("reliquary_stateful_tools.taskset", fromlist=["_build_task"])._build_task(0, "train")
    actions = source["private"]["reference_actions"]
    exact_state = StatefulToolsEnvironment().replay(0, actions)["state"]
    duplicate_key = _trace(task, actions, exact_state)
    duplicate_key.nodes[0].message.tool_calls[0].arguments = (
        '{"email":"ignored","email":'
        + json.dumps(actions[0]["arguments"]["email"])
        + "}"
    )
    non_finite = _trace(task, actions, exact_state)
    non_finite.nodes[0].message.tool_calls[0].arguments = '{"email":NaN}'
    overflow = _trace(task, actions, exact_state)
    overflow.nodes[0].message.tool_calls[0].arguments = '{"email":1e999}'
    oversized_arguments = _trace(task, actions, exact_state)
    oversized_arguments.nodes[0].message.tool_calls[0].arguments = json.dumps(
        {"email": "x" * (16 * 1024)}
    )
    traces = (
        (_trace(task, actions), 1.0),
        (_trace(task, [actions[-1]], exact_state), 0.0),
        (
            _trace(
                task,
                [{"tool": "unknown", "arguments": {}}, *actions],
                exact_state,
            ),
            0.0,
        ),
        (duplicate_key, 0.0),
        (non_finite, 0.0),
        (overflow, 0.0),
        (oversized_arguments, 0.0),
        (_trace(task, [{"final": "x" * (16 * 1024)}], exact_state), 0.0),
        (_trace(task, [actions[0]] * 4 + actions), 0.0),
    )
    for trace, expected in traces:
        assert asyncio.run(task.verified_outcome(trace)) == expected
        wire = vf.WireTrace.model_validate_json(trace.model_dump_json())
        assert asyncio.run(task.verified_outcome(wire)) == expected


def test_compatibility_contract_is_deterministic_and_fail_closed() -> None:
    environment = StatefulToolsEnvironment("eval")
    assert len(environment) == 1 << 31
    assert environment.task(7) == StatefulToolsEnvironment("eval").task(7)
    reset = environment.reset(7, 41)
    bad = environment.step(7, reset["state"], {"tool": "unknown", "arguments": {}})
    assert bad["done"] is True
    assert bad["termination_reason"] == "invalid_action"
    assert environment.grade(7, bad["state"], [])["reward"] == 0.0


def test_mutations_are_idempotent_for_transport_retries() -> None:
    module = __import__("reliquary_stateful_tools.taskset", fromlist=["_build_task"])
    mutating = {"update_shipping_address", "create_refund", "add_support_note"}
    for index in range(3):
        environment = StatefulToolsEnvironment()
        actions = module._build_task(index, "train")["private"]["reference_actions"]
        mutation = next(action for action in actions if action.get("tool") in mutating)
        replay = environment.replay(index, [*actions[:-1], mutation, actions[-1]])
        assert replay["reward"]["reward"] == 1.0


def test_generator_never_aliases_target_and_distractor() -> None:
    module = __import__("reliquary_stateful_tools.taskset", fromlist=["_build_task"])
    for split, index in (("eval", 35_803), ("train", 522_685)):
        task = module._build_task(index, split)
        private = task["private"]
        assert private["target_customer_id"] != private["other_customer_id"]
        assert private["target_order_id"] != private["other_order_id"]
        assert StatefulToolsEnvironment(split).replay(
            index, private["reference_actions"]
        )["reward"]["reward"] == 1.0


def test_artifact_manifest_hashes_installed_files() -> None:
    package_root = importlib.resources.files("reliquary_stateful_tools")
    root = package_root.parent
    artifact = json.loads(package_root.joinpath("artifact.json").read_text())
    assert artifact["entrypoints"]["taskset"] == "reliquary_stateful_tools:StatefulToolsTaskset"
    source_manifest = Path(__file__).parents[1] / "environment.toml"
    assert hashlib.sha256(source_manifest.read_bytes()).hexdigest() == artifact["source_manifest_sha256"]
    for name, expected in artifact["files"].items():
        assert hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() == expected
