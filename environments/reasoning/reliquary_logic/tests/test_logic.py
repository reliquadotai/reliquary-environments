import asyncio
import collections
import hashlib
import importlib.resources
import json
import tomllib
from pathlib import Path

import pytest
import verifiers.v1 as vf

import reliquary_logic as package
from reliquary_logic.taskset import (
    SPLITS,
    TASK_COUNT,
    LogicEnvironment,
    _build_task,
)

FAMILIES = 12


def _trace(task: vf.Task, completion: str) -> vf.Trace:
    return vf.Trace(
        task=vf.TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        ),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[
            vf.MessageNode(
                parent=None,
                message=vf.AssistantMessage(content=completion),
                sampled=True,
            )
        ],
        state=vf.State(),
    )


def _goldens() -> list[dict]:
    return [
        json.loads(line)
        for line in importlib.resources.files("reliquary_logic")
        .joinpath("goldens/reference.jsonl")
        .read_text()
        .splitlines()
        if line
    ]


def test_reference_goldens_and_rejections() -> None:
    """The goldens freeze the puzzles, not just the plumbing.

    `prompt_sha256` covers the generator and the template together, so a
    change to either has to be declared rather than discovered in a run.
    """
    rows = _goldens()
    assert {row["family"] for row in rows} == {
        "boolean_expressions",
        "dyck_language",
        "numbrix",
    }
    for golden in rows:
        environment = LogicEnvironment(golden["split"])
        task = environment.task(golden["index"])
        assert task["id"] == golden["task_id"]
        assert task["metadata"]["family"] == golden["family"]
        assert (
            hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest()
            == golden["prompt_sha256"]
        )
        reward = environment.grade(
            golden["index"], environment.reference_completion(golden["index"])
        )
        assert reward["reward"] == 1.0
        assert reward["success"] is True
        assert reward["state_digest"] == golden["state_digest"]
        wrong = environment.grade(golden["index"], '```json\n{"result": "no"}\n```')
        assert wrong["reward"] == 0.0
        assert wrong["state_digest"] != golden["state_digest"]


def test_every_family_is_generated_and_answerable() -> None:
    environment = LogicEnvironment()
    seen: dict[str, int] = {}
    for index in range(3000):
        seen.setdefault(environment.task(index)["metadata"]["family"], index)
    assert len(seen) == FAMILIES
    for family, index in seen.items():
        graded = environment.grade(index, environment.reference_completion(index))
        assert graded["reward"] == 1.0, family


def test_splits_are_disjoint_and_unbiased() -> None:
    """Interleaving must not sort families into splits.

    The family is drawn from a hash of the position rather than cycling with
    it, so taking every third index leaves the mix intact — but that is a
    property of the generator, so it is measured rather than assumed.
    """
    identities = {
        split: {LogicEnvironment(split).task(i)["id"] for i in range(400)}
        for split in SPLITS
    }
    for split, ids in identities.items():
        assert len(ids) == 400, split
    assert not set.intersection(*identities.values())

    prompts = {
        split: {LogicEnvironment(split).task(i)["prompt"] for i in range(400)}
        for split in SPLITS
    }
    assert not set.intersection(*prompts.values())

    for split in SPLITS:
        environment = LogicEnvironment(split)
        counts = collections.Counter(
            environment.task(i)["metadata"]["family"] for i in range(1200)
        )
        assert len(counts) == FAMILIES, split
        assert max(counts.values()) < 3 * min(counts.values()), (split, counts)


def test_indices_wrap_within_a_split() -> None:
    environment = LogicEnvironment()
    assert len(environment) == TASK_COUNT
    assert environment.task(0)["prompt"] == environment.task(TASK_COUNT)["prompt"]
    assert environment.task(0)["id"] != environment.task(TASK_COUNT)["id"]


@pytest.mark.parametrize(
    "completion",
    [
        '```json\n{"result": 1, "result": 2}\n```',
        '```json\n{"result": 1.5}\n```',
        '```json\n{"result": NaN}\n```',
        '```json\n[1, 2, 3]\n```',
        '```json\n{"result": 1, "extra": 2}\n```',
        "there is no answer here",
        "",
        "x" * (16 * 1024 + 1),
    ],
)
def test_the_answer_channel_fails_closed(completion: str) -> None:
    assert LogicEnvironment().grade(0, completion)["reward"] == 0.0


def test_reasoning_precedes_the_answer_without_hiding_it() -> None:
    """The measured defect: the model's own fences shift the pairing."""
    environment = LogicEnvironment()
    answer = environment.reference_completion(0)
    payload = answer.split("```json\n")[1].split("\n```")[0]
    noisy = f"Let me work it out.\n\n```python\nx = 1\n```\n\n```json\n{payload}\n```"
    unclosed = f"```python\nx = 1\n\nFinal answer:\n```json\n{payload}\n```"
    for completion in (noisy, unclosed):
        assert environment.grade(0, completion)["reward"] == 1.0


def test_reasoning_families_scopes_the_template() -> None:
    everywhere = LogicEnvironment()
    nowhere = LogicEnvironment(reasoning_families=())
    step_by_step = "Solve the following problem step by step."
    scoped_families = {"numbrix", "dyck_language_errors"}
    scoped = LogicEnvironment(reasoning_families=tuple(scoped_families))
    seen = set()
    for index in range(600):
        family = everywhere.task(index)["metadata"]["family"]
        seen.add(family)
        assert everywhere.task(index)["prompt"].startswith(step_by_step)
        assert not nowhere.task(index)["prompt"].startswith(step_by_step)
        assert scoped.task(index)["prompt"].startswith(step_by_step) is (
            family in scoped_families
        )
        # Scoping the prompt must not move the puzzle or its answer.
        assert scoped.grade(
            index, scoped.reference_completion(index)
        )["reward"] == 1.0
    assert seen & scoped_families


def test_compatibility_contract_is_deterministic() -> None:
    environment = LogicEnvironment("eval")
    assert environment.task(7) == LogicEnvironment("eval").task(7)
    assert environment.max_turns == 1
    assert environment.validator_authoritative_reward is True
    replay = environment.replay(7, environment.reference_completion(7))
    assert replay["reward"]["reward"] == 1.0
    with pytest.raises(ValueError):
        LogicEnvironment("nope")


def test_verifiers_load_and_wire_replay() -> None:
    exported = [
        item
        for name in package.__all__
        if isinstance((item := getattr(package, name)), type)
        and issubclass(item, vf.Taskset)
    ]
    assert exported == [package.LogicTaskset]
    config_type = vf.taskset_config_type("reliquary-logic")
    taskset = vf.load_taskset(config_type(id="reliquary-logic"))
    tasks = list(taskset.head(3))
    assert len(tasks) == 3
    environment = LogicEnvironment()
    for position, task in enumerate(tasks):
        assert task.key == environment.task(position)["id"]
        assert asyncio.run(task.validate(None)) is True
        cases = (
            (environment.reference_completion(position), 1.0),
            ('```json\n{"result": "__not_the_answer__"}\n```', 0.0),
            ("", 0.0),
        )
        for completion, expected in cases:
            trace = _trace(task, completion)
            assert asyncio.run(task.verified_answer(trace)) == expected
            wire = vf.WireTrace.model_validate_json(trace.model_dump_json())
            assert asyncio.run(task.verified_answer(wire)) == expected


def test_a_taskset_split_carries_through_to_the_task() -> None:
    config_type = vf.taskset_config_type("reliquary-logic")
    taskset = vf.load_taskset(config_type(id="reliquary-logic", split="eval"))
    task = next(iter(taskset))
    assert task.data.split == "eval"
    assert task.key == LogicEnvironment("eval").task(0)["id"]
    assert task.data.verifier_spec == (
        _build_task(0, "eval")["private"]["verifier_spec"]
    )


def test_artifact_manifest_hashes_installed_files() -> None:
    package_root = importlib.resources.files("reliquary_logic")
    root = package_root.parent
    artifact = json.loads(package_root.joinpath("artifact.json").read_text())
    assert artifact["entrypoints"]["taskset"] == "reliquary_logic:LogicTaskset"
    source_manifest = Path(__file__).parents[1] / "environment.toml"
    assert (
        hashlib.sha256(source_manifest.read_bytes()).hexdigest()
        == artifact["source_manifest_sha256"]
    )
    for name, expected in artifact["files"].items():
        assert hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() == expected


def test_declared_families_match_the_generator() -> None:
    manifest = tomllib.loads(
        (Path(__file__).parents[1] / "environment.toml").read_text()
    )
    environment = LogicEnvironment()
    generated = {
        environment.task(index)["metadata"]["family"] for index in range(3000)
    }
    assert set(manifest["task_families"]) == generated
    assert manifest["execution"]["max_turns"] == 1
    assert manifest["execution"]["network"] is False
    assert manifest["entrypoint"] == "reliquary_logic:LogicTaskset"
    assert manifest["compatibility_entrypoint"] == "reliquary_logic:LogicEnvironment"


def test_prime_rl_reference_config_and_pins() -> None:
    """The training lane must name this taskset and the repository's pins.

    Nothing here downloads a release: the logic wheel has none yet, so CI
    installs it from the checkout rather than asserting a hash that names no
    published artifact.
    """
    environment_root = Path(__file__).parents[1]
    config = tomllib.loads(
        (environment_root / "examples/prime_rl/rl.toml").read_text()
    )
    compatibility = tomllib.loads(
        (environment_root.parents[2] / "compatibility.toml").read_text()
    )

    assert compatibility["prime_rl"]["version"] == "0.9.0"
    assert (
        compatibility["verifiers"]["source_commit"]
        == compatibility["prime_rl"]["verifiers_commit"]
    )
    assert config["model"]["name"] == (
        compatibility["models"]["qwen3_4b_instruct_2507"]["id"]
    )
    assert config["orchestrator"]["algo"]["type"] == "grpo"
    assert config["orchestrator"]["renderer"]["name"] == "qwen3"
    assert config["orchestrator"]["group_size"] == 16
    for phase, split in (("train", "train"), ("eval", "eval")):
        source = config["orchestrator"][phase]["source"][0]
        assert source["env"]["taskset"] == {"id": "reliquary-logic", "split": split}
        assert source["env"]["agent"]["max_turns"] == 1
        assert source["env"]["agent"]["runtime"] == {"type": "docker", "allow": []}
    # Single turn with no tools: a tool-call parser here would be cargo.
    assert "tool_call_parser" not in config["inference"]["vllm"]
