from __future__ import annotations

import asyncio
import gzip
import hashlib
import importlib.resources
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

import reliquary_automationbench as package
from reliquary_automationbench import corpus
from reliquary_automationbench._determinism import _det_now, episode_context, uuid4
from reliquary_automationbench.environment import (
    MAX_CONTENT_BYTES,
    TOOL_SCHEMAS,
    AutomationBenchEnvironment,
    _CACHE,
)
from reliquary_automationbench.world import build_world

sys.path.insert(0, str(Path(__file__).parent))
from probes import fuzz_actions, probe_actions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "reliquary_automationbench"


def _sample(split: str = "eval", per_domain: int = 3) -> list[tuple[str, int]]:
    if split == "eval":
        return [(split, d * corpus.DOMAIN_SIZE + i) for d in range(len(corpus.DOMAINS)) for i in range(per_domain)]
    return [(split, i) for i in range(per_domain)]


def _digest(episodes: list[dict]) -> str:
    body = json.dumps(
        [[e["transcript_digest"], e["reward"], e["termination_reason"]] for e in episodes], sort_keys=True
    )
    return hashlib.sha256(body.encode()).hexdigest()


# -- the corpus ---------------------------------------------------------------


def test_the_corpus_matches_its_pin() -> None:
    body = gzip.decompress(importlib.resources.files(package).joinpath(corpus.TASKS_FILE).read_bytes())
    assert hashlib.sha256(body).hexdigest() == corpus.TASKS_SHA256
    assert len(corpus.tasks()) == corpus.TASK_COUNT == 800


def test_eval_is_the_six_scored_domains_in_contiguous_blocks() -> None:
    rows = corpus.rows("eval")
    assert len(rows) == 600
    for d, domain in enumerate(corpus.DOMAINS):
        block = rows[d * 100 : (d + 1) * 100]
        assert {task.domain for task in block} == {domain}
        assert block == corpus.rows(domain)
    assert {task.domain for task in corpus.rows("qualification")} == {"simple"}
    assert len(corpus.rows("qualification")) == 200
    assert not {t.name for t in rows} & {t.name for t in corpus.rows("qualification")}


def test_there_is_no_train_split() -> None:
    with pytest.raises(ValueError):
        AutomationBenchEnvironment("train")
    with pytest.raises(TypeError):
        AutomationBenchEnvironment()  # type: ignore[call-arg]


def test_every_task_has_a_system_prompt_a_request_and_assertions() -> None:
    for task in corpus.tasks():
        assert task.system.startswith("You are a workflow automation agent")
        assert task.user.strip()
        assert task.assertions
        assert task.scenario_time.tzinfo is not None


# -- the replay surface -------------------------------------------------------


def test_the_task_wire_fits_a_replay_consumer() -> None:
    environment = AutomationBenchEnvironment("eval")
    for index in range(len(environment)):
        task = environment.task(index)
        assert set(task) == {"id", "prompt", "tools", "metadata"}
        assert 0 < len(task["id"]) <= 128
        assert task["metadata"]["domain"] == corpus.DOMAINS[index // 100]
    assert [tool["name"] for tool in TOOL_SCHEMAS] == ["api_search", "api_fetch", "base64_encode"]
    for tool in TOOL_SCHEMAS:
        assert set(tool) == {"name", "description", "parameters"}
        text = json.dumps(tool["parameters"])
        # Flat: the ChatML dialect and the replay consumer need no combinators.
        assert not re.search(r'"(allOf|anyOf|oneOf|\$ref)"', text)


def test_an_empty_episode_scores_nothing_on_any_task() -> None:
    for split in ("eval", "qualification"):
        environment = AutomationBenchEnvironment(split)
        for index in range(len(environment)):
            episode = environment.replay(index, [{"final": "Nothing to do."}])
            assert episode["reward"]["reward"] == 0.0, environment.task(index)["id"]
            assert episode["termination_reason"] == "finished"


def test_a_final_answer_ends_the_episode_and_later_steps_are_refused() -> None:
    environment = AutomationBenchEnvironment("qualification")
    state = environment.reset(0)["state"]
    result = environment.step(0, state, {"final": "done"})
    assert result["done"] and result["termination_reason"] == "finished" and result["events"] == []
    with pytest.raises(ValueError):
        environment.step(0, state, {"tool": "api_search", "arguments": {"query": "x"}})


def test_bad_calls_are_reported_and_the_episode_goes_on() -> None:
    environment = AutomationBenchEnvironment("qualification")
    state = environment.reset(0)["state"]
    for action, prefix in (
        ({"tool": "nope", "arguments": {}}, "Error: Tool 'nope' not found."),
        ({"tool": "api_fetch", "arguments": {"wrong": 1}}, "Error: TypeError"),
        ({"raw_prefix": "garbage"}, "Error: expected a tool call"),
    ):
        result = environment.step(0, state, action)
        assert not result["done"]
        assert result["events"][0]["content"].startswith(prefix)
    assert state["errors"] == 3


def test_the_turn_limit_ends_the_episode() -> None:
    environment = AutomationBenchEnvironment("qualification")
    actions = [{"tool": "base64_encode", "arguments": {"text": str(i)}} for i in range(150)]
    episode = environment.replay(0, actions)
    assert episode["termination_reason"] == "turn_limit"
    assert len(episode["events"]) == environment.max_turns


def test_long_results_are_cut_under_the_event_bound() -> None:
    environment = AutomationBenchEnvironment("eval")
    state = environment.reset(0)["state"]
    content = environment.step(0, state, {"tool": "api_search", "arguments": {"query": "list get find", "top_k": 500}})["events"][0]["content"]
    assert "[truncated:" in content
    assert len(content.encode()) <= MAX_CONTENT_BYTES + 100
    wire = json.dumps([{"role": "tool", "name": "api_search", "content": content}], ensure_ascii=True)
    assert len(wire.encode()) <= 64 * 1024


# -- determinism ---------------------------------------------------------------


def test_world_code_refuses_to_run_outside_an_episode_context() -> None:
    from reliquary_automationbench._upstream.schema.world import WorldState

    with pytest.raises(RuntimeError):
        _det_now()
    with pytest.raises(RuntimeError):
        uuid4()
    with pytest.raises(RuntimeError):
        WorldState()


def test_the_clock_is_pinned_and_ids_are_seeded() -> None:
    task = corpus.rows("eval")[0]
    with episode_context(label="a", scenario_time=task.scenario_time, offset_seconds=3):
        first = (_det_now(), uuid4())
    with episode_context(label="a", scenario_time=task.scenario_time, offset_seconds=3):
        again = (_det_now(), uuid4())
    with episode_context(label="b", scenario_time=task.scenario_time, offset_seconds=3):
        other = uuid4()
    assert first == again and other != first[1]
    assert (first[0] - task.scenario_time.replace(tzinfo=None)).total_seconds() == 3


def test_tasks_without_a_scenario_time_run_at_the_pinned_default() -> None:
    task = corpus.rows("qualification")[0]
    assert "current_time" not in (task.initial_state.get("meta") or {})
    world = build_world(task, 0)
    assert world.meta.current_time == corpus.DEFAULT_SCENARIO_TIME


@pytest.mark.parametrize("split,index", _sample("eval", 2) + _sample("qualification", 3))
def test_a_replayed_episode_repeats_itself(split: str, index: int) -> None:
    environment = AutomationBenchEnvironment(split)
    task = corpus.rows(split)[index]
    for actions in (probe_actions(task), fuzz_actions(task, 30)):
        first = environment.replay(index, actions, seed=5)
        second = environment.replay(index, actions, seed=5)
        assert first == second


def test_the_cache_changes_speed_and_never_a_result() -> None:
    environment = AutomationBenchEnvironment("eval")
    for split, index in _sample("eval", 2):
        task = corpus.rows(split)[index]
        actions = fuzz_actions(task, 30)[:-1]
        cached = environment.reset(index, 1)["state"]
        cold = environment.reset(index, 1)["state"]
        for action in actions:
            warm_events = environment.step(index, cached, action)["events"]
            _CACHE._items.clear()  # every cold step rebuilds the world by replay
            cold_events = environment.step(index, cold, action)["events"]
            assert warm_events == cold_events
        assert environment.grade(index, cached) == environment.grade(index, cold)


def test_episodes_differ_by_seed_only_in_minted_values() -> None:
    environment = AutomationBenchEnvironment("eval")
    task = corpus.rows("eval")[0]
    actions = probe_actions(task)
    a, b = environment.replay(0, actions, seed=1), environment.replay(0, actions, seed=2)
    assert a["reward"]["reward"] == b["reward"]["reward"]
    assert a["partial_credit"] == b["partial_credit"]


_CROSS_PROCESS = """
import sys, json
sys.path.insert(0, sys.argv[1])
from probes import fuzz_actions, probe_actions
from reliquary_automationbench import corpus
from reliquary_automationbench.environment import AutomationBenchEnvironment
out = []
for split, index in json.loads(sys.argv[2]):
    env = AutomationBenchEnvironment(split)
    task = corpus.rows(split)[index]
    out.append(env.replay(index, probe_actions(task) [:-1] + fuzz_actions(task, 30), seed=9))
print(json.dumps([[e["transcript_digest"], e["reward"], e["termination_reason"]] for e in out], sort_keys=True))
"""


def test_replay_is_identical_across_hash_seeds_and_time_zones() -> None:
    """Set order and naive timestamps are where a process leaks into a world."""
    sample = json.dumps(_sample("eval", 2) + _sample("qualification", 2))
    outputs = set()
    for hash_seed, zone in (("1", "UTC"), ("2", "Asia/Tokyo"), ("random", "America/Los_Angeles")):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, TZ=zone, PYTHONPATH=os.pathsep.join([str(ROOT), *sys.path]))
        result = subprocess.run(
            [sys.executable, "-c", _CROSS_PROCESS, str(Path(__file__).parent), sample],
            env=env, capture_output=True, text=True, check=True,
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1


# A digest of probe episodes over the first three tasks of each scored domain
# and of the simple domain, pinned. It moves when the corpus, the vendored
# world, the grading or the determinism layer moves; that is its job.
PROBE_DIGEST = "a2994a5a5064927678b5e2589d112e75a61b3e78c0b0f8ded7e1dba2ac8c07fe"


def test_probe_episodes_match_their_pin() -> None:
    episodes = []
    for split, index in _sample("eval", 3) + _sample("qualification", 3):
        task = corpus.rows(split)[index]
        episodes.append(AutomationBenchEnvironment(split).replay(index, probe_actions(task), seed=0))
    assert _digest(episodes) == PROBE_DIGEST


def test_some_probe_episodes_pass() -> None:
    """The probes reach the passing branch of the grader, not only failures."""
    environment = AutomationBenchEnvironment("qualification")
    passed = sum(
        environment.replay(i, probe_actions(task))["reward"]["reward"] == 1.0
        for i, task in enumerate(corpus.rows("qualification")[:60])
    )
    assert passed >= 5


def test_nothing_reachable_reads_the_host() -> None:
    """The vendored code reads no wall clock, unseeded randomness, environment,
    host time zone or network: the rewrite left no such call behind."""
    forbidden = re.compile(
        r"datetime\.now|datetime\.utcnow|date\.today|time\.time\(\)|\.timestamp\(\)|fromtimestamp|localtime"
        r"|^\s*import (random|uuid|secrets|socket|requests|httpx)\b|^\s*from (random|uuid|openai|anthropic|httpx|requests) import"
        r"|os\.environ|os\.getenv|OpenAI\(",
        re.M,
    )
    offenders = []
    for path in sorted((PACKAGE / "_upstream").rglob("*.py")):
        for match in forbidden.finditer(path.read_text()):
            offenders.append(f"{path.relative_to(PACKAGE)}: {match.group(0)}")
    assert offenders == []


def test_the_chatgpt_service_answers_its_offline_stub() -> None:
    environment = AutomationBenchEnvironment("eval")
    task = next(i for i, t in enumerate(corpus.rows("eval")) if "chatgpt" in t.initial_state)
    state = environment.reset(task)["state"]
    result = environment.step(
        task,
        state,
        {
            "tool": "api_fetch",
            "arguments": {
                "method": "POST",
                "url": "https://api.openai.com/v1/chat/completions",
                "body": json.dumps({"model": "gpt-5-mini", "messages": [{"role": "user", "content": "classify this"}]}),
            },
        },
    )
    assert "[Simulated response to: classify this" in result["events"][0]["content"]


# -- grading ------------------------------------------------------------------


def test_grade_reports_upstreams_partial_credit_beside_the_strict_pass() -> None:
    environment = AutomationBenchEnvironment("eval")
    for index, task in enumerate(corpus.rows("eval")[:40]):
        state = environment.reset(index)["state"]
        for action in probe_actions(task)[:-1]:
            environment.step(index, state, action)
        report = environment.grade_report(index, state)
        assert report["reward"] == (1.0 if report["partial_credit"] == 1.0 else 0.0)
        assert report["success"] is (report["reward"] == 1.0)
        assert set(environment.grade(index, state)) == {"reward", "success", "checks", "state_digest", "environment_error"}
        assert len(report["state_digest"]) == 64


UPSTREAM = os.environ.get("AUTOMATIONBENCH_UPSTREAM")


@pytest.mark.skipif(not UPSTREAM, reason="set AUTOMATIONBENCH_UPSTREAM to an AutomationBench checkout at the pinned commit")
@pytest.mark.parametrize("split", ["eval", "qualification"])
def test_grades_match_upstream(split: str) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import parity_upstream

    assert parity_upstream.main(Path(UPSTREAM), split, 60 if split == "eval" else 30) == 0


# -- the Verifiers surface -----------------------------------------------------


def test_the_toolset_reads_what_replay_reads() -> None:
    import verifiers.v1 as vf

    from reliquary_automationbench.taskset import AutomationBenchState, AutomationBenchToolset

    environment = AutomationBenchEnvironment("eval")
    actions = probe_actions(corpus.rows("eval")[0])[:-1]
    toolset = AutomationBenchToolset(vf.ToolsetConfig())
    toolset._inert_state = AutomationBenchState(index=0, split="eval")
    observed = [toolset._run(action["tool"], **action["arguments"]) for action in actions]
    assert observed == [event["content"] for event in environment.replay(0, actions)["events"]]


def test_the_taskset_loads_and_validates() -> None:
    from reliquary_automationbench.taskset import AutomationBenchConfig

    taskset = package.AutomationBenchTaskset(AutomationBenchConfig(split="qualification"))
    tasks = [task for _, task in zip(range(3), taskset.load())]
    assert [task.key for task in tasks] == [t.name for t in corpus.rows("qualification")[:3]]
    for task in tasks:
        assert asyncio.run(task.validate(None)) is True


def test_every_module_is_in_a_regular_package() -> None:
    """Reliquary's verified loader serves only artifact-bound modules: a
    namespace package (no `__init__.py`) cannot be imported through it."""
    for path in PACKAGE.rglob("*.py"):
        for folder in path.relative_to(PACKAGE).parents:
            assert (PACKAGE / folder / "__init__.py").is_file(), folder


def test_artifact_manifest_hashes_the_package() -> None:
    artifact = json.loads((PACKAGE / "artifact.json").read_text())
    shipped = {
        str(path.relative_to(ROOT))
        for path in PACKAGE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.name != "artifact.json"
    }
    assert set(artifact["files"]) == shipped
    for name, digest in artifact["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    assert artifact["source_manifest_sha256"] == hashlib.sha256((ROOT / "environment.toml").read_bytes()).hexdigest()
    assert artifact["environment"] == "reliquary_automationbench_v1"


def test_the_scenario_time_is_parsed_as_upstream_seeds_it() -> None:
    times = {task.scenario_time for task in corpus.tasks()}
    assert corpus.DEFAULT_SCENARIO_TIME in times
    assert all(isinstance(value, datetime) for value in times)
