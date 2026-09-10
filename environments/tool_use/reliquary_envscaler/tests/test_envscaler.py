import hashlib
import importlib.resources
import json
import tomllib
from pathlib import Path

import pytest
import verifiers.v1 as vf

import reliquary_envscaler as package
from reliquary_envscaler import EnvScalerToolsEnvironment
from reliquary_envscaler.corpus import EXPECTED_SCENARIOS, EXPECTED_WORLDS, load
from reliquary_envscaler.taskset import EVAL_SCENARIOS


def test_corpus_is_decoded_and_complete():
    """The release ships two fields as JSON strings inside the JSON.

    `tools` and `init_config` come back as text; the environment reads them as
    containers, and a loader that forgets fails only later, on a task nobody
    is looking at.
    """
    worlds, scenarios = load()
    assert len(worlds) == EXPECTED_WORLDS
    assert len(scenarios) == EXPECTED_SCENARIOS
    for world in list(worlds.values())[:20]:
        assert isinstance(world["tools"], list)
        assert all(isinstance(t, dict) for t in world["tools"])
    for scenario in scenarios[:20]:
        assert isinstance(scenario["init_config"], dict)
        # Only the RL scenarios carry checks, and an environment that cannot
        # grade is not an environment.
        assert scenario["checklist_with_func"]


def test_tasks_are_deterministic_and_carry_their_world():
    environment = EnvScalerToolsEnvironment()
    first = environment.get_task(7)
    assert first.id == EnvScalerToolsEnvironment().get_task(7).id
    assert first.tools, "a task without tools cannot be acted on"
    assert len({t.name for t in first.tools}) == len(first.tools)
    assert first.private["env_id"].endswith("_rl")


def test_splits_are_frozen_and_disjoint():
    """Evaluation takes the first scenarios and a teacher never sees them."""
    config = vf.taskset_config_type("reliquary-envscaler")
    train = [t.data.scenario_index
             for t in vf.load_taskset(config(id="reliquary-envscaler")).head(40)]
    held = [t.data.scenario_index
            for t in vf.load_taskset(
                config(id="reliquary-envscaler", split="eval")).head(40)]
    assert min(train) >= EVAL_SCENARIOS
    assert max(held) < EVAL_SCENARIOS
    assert not set(train) & set(held)


def test_verifiers_loads_the_taskset():
    exported = [item for name in package.__all__
                if isinstance((item := getattr(package, name)), type)
                and issubclass(item, vf.Taskset)]
    assert exported == [package.EnvScalerTaskset]
    config = vf.taskset_config_type("reliquary-envscaler")
    tasks = list(vf.load_taskset(config(id="reliquary-envscaler")).head(3))
    assert len(tasks) == 3
    assert all(t.data.checks > 0 for t in tasks)


def test_reward_counts_only_the_checks_the_agent_must_flip():
    """A no-op scores zero here; upstream's definition pays 0.166 for free.

    15.5% of checks are already true at reset. Averaging all of them, as
    upstream does, hands an idle agent a sixth of the range.
    """
    environment = EnvScalerToolsEnvironment()
    task = environment.get_task(EVAL_SCENARIOS + 1)
    state = environment.reset(task, 0).state
    from reliquary_envscaler.episode import EpisodeTrace

    trace = EpisodeTrace(
        schema="reliquary/episode/v1", environment="reliquary_envscaler_v2",
        task_id=task.id, seed=0, events=(), actions=(), tokens=(),
        assistant_spans=(), observation_digests=(),
        termination_reason="turn_limit",
    )
    report = environment.grade(task, state, trace)
    assert report.reward == 0.0
    assert not report.success


def test_goldens_pin_the_corpus():
    rows = [json.loads(line) for line in
            importlib.resources.files("reliquary_envscaler")
            .joinpath("goldens/reference.jsonl").read_text().splitlines() if line]
    environment = EnvScalerToolsEnvironment()
    for golden in rows:
        task = environment.get_task(golden["index"])
        assert task.id == golden["task_id"]
        assert task.private["env_id"] == golden["env_id"]
        assert len(task.tools) == golden["tools"]
        assert hashlib.sha256(task.prompt.encode()).hexdigest() == golden["prompt_sha256"]


def test_artifact_manifest_hashes_installed_files():
    root = importlib.resources.files("reliquary_envscaler")
    artifact = json.loads(root.joinpath("artifact.json").read_text())
    assert artifact["entrypoints"]["taskset"] == "reliquary_envscaler:EnvScalerTaskset"
    manifest = Path(__file__).parents[1] / "environment.toml"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == artifact["source_manifest_sha256"]
    for name, expected in artifact["files"].items():
        assert hashlib.sha256(root.parent.joinpath(name).read_bytes()).hexdigest() == expected


def test_manifest_declares_the_untrusted_execution():
    """Worlds and checkers are dataset-carried Python. Saying so is the point."""
    manifest = tomllib.loads(
        (Path(__file__).parents[1] / "environment.toml").read_text())
    assert manifest["execution"]["executes_untrusted_source"] is True
    assert manifest["execution"]["network"] is False
    assert manifest["execution"]["max_turns"] == 12
    assert manifest["data"]["redistributable"] is False


def test_the_toolset_advertises_the_task_s_own_tools():
    """The claim the package rests on, exercised rather than asserted.

    A fixed `@vf.tool` roster cannot serve 191 worlds, so `register` builds
    the roster from the task. That is only legal because Verifiers gives each
    rollout its own tool server and calls `setup_task` before `register`
    (`verifiers/v1/mcp/server.py`); if that order ever changes, this fails.
    """
    import asyncio

    from mcp.server.fastmcp import FastMCP

    from reliquary_envscaler.taskset import EnvScalerToolset

    config = vf.taskset_config_type("reliquary-envscaler")
    tasks = list(vf.load_taskset(config(id="reliquary-envscaler")).head(2))
    assert tasks[0].data.env_id != tasks[1].data.env_id, "need two worlds"

    environment = EnvScalerToolsEnvironment()
    seen = []
    for task in tasks:
        toolset = EnvScalerToolset(vf.ToolsetConfig())
        asyncio.run(toolset.setup_task(task))
        mcp = FastMCP("probe")
        toolset.register(mcp)
        advertised = {t.name for t in asyncio.run(mcp.list_tools())}
        expected = {s.name for s in
                    environment.get_task(task.data.scenario_index).tools}
        assert advertised == expected, task.data.env_id
        seen.append(advertised)

    # Two different worlds must not end up with the same roster, or the
    # registration is reading something other than the task.
    assert seen[0] != seen[1]
