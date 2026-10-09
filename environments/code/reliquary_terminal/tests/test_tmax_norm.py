"""reliquary-terminal's tmax splits on the env norm (signed-episode sandboxes): a task
hash that does not depend on where the cache lives, the grade-role setup as the pristine
box's preparation, a reward that grades there, one task by index, plain-data cases."""

import base64
import pickle
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from test_tmax_split import IDS, SetupBox, _config, _load, tmax_env  # noqa: F401 (fixture)
from verifiers.v1.errors import TaskError
from verifiers.v1.tasksets.harbor.taskset import HarborData, VerifierConfig

import reliquary_terminal
from reliquary_terminal import conformance, grading, tmax, tmax_select
from reliquary_terminal.grading import TerminalTask, task_dir_digest
from reliquary_terminal.taskset import TerminalTaskset

SANDBOX = SimpleNamespace(config=SimpleNamespace(type="reliquary-sandbox", workdir="/"))
DOCKER = SimpleNamespace(config=SimpleNamespace(type="docker", workdir="/"))


def trace_of(task):
    return vf.Trace(agent=vf.AgentInfo(config=vf.AgentConfig()),
                    task=vf.TraceTask(type="TerminalTask", data=task.data))


def test_the_task_hash_does_not_depend_on_where_the_cache_lives(tmax_env, tmp_path,
                                                                 monkeypatch):
    src, _ = tmax_env
    first = _load(src, num_tasks=1)[0]
    monkeypatch.setattr(tmax, "CACHE", tmp_path / "another-home" / "cache")
    second = _load(src, num_tasks=1)[0]
    assert first.data.task_dir != second.data.task_dir
    assert first.hash == second.hash
    assert first.key == second.key


def test_the_hash_still_follows_the_tasks_content(tmax_env, tmp_path, monkeypatch):
    src, _ = tmax_env
    first = _load(src, num_tasks=1)[0]
    monkeypatch.setattr(tmax, "TEST_SH", tmax.TEST_SH + "# changed\n")
    monkeypatch.setattr(tmax, "CACHE", tmp_path / "elsewhere")
    assert _load(src, num_tasks=1)[0].hash != first.hash


def test_a_task_dir_without_a_stamp_is_digested_from_its_files(tmp_path):
    one, two = tmp_path / "one", tmp_path / "elsewhere" / "two"
    for root in (one, two):
        (root / "tests").mkdir(parents=True)
        (root / "tests" / "test.sh").write_bytes(b"exit 0\n")
    assert task_dir_digest(str(one)) == task_dir_digest(str(two))
    (two / "tests" / "test.sh").write_bytes(b"exit 1\n")
    assert task_dir_digest(str(one)) != task_dir_digest(str(two))
    (one / ".content").write_text("abc")
    assert task_dir_digest(str(one)) == "abc"


def test_task_at_is_the_index_th_task_of_load(tmax_env):
    src, _ = tmax_env
    taskset = TerminalTaskset(_config(split="tmax", tmax_source=src))
    loaded = list(taskset)
    assert len(taskset) == len(loaded) == 4
    for index, task in enumerate(loaded):
        assert taskset.task_at(index).hash == task.hash
        assert taskset.task_at(index).data.idx == index
    with pytest.raises(IndexError):
        taskset.task_at(4)
    with pytest.raises(IndexError):
        taskset.task_at(-1)
    with pytest.raises(ValueError):
        TerminalTaskset(_config(split="tmax", tmax_source=src,
                                tasks=[loaded[0].data.name])).task_at(0)


def test_len_and_task_at_follow_the_part_and_num_tasks(tmax_env):
    src, _ = tmax_env
    rl = TerminalTaskset(_config(split="tmax_rl", tmax_source=src))
    assert len(rl) == 2
    assert [rl.task_at(i).data.name for i in range(2)] == [t.data.name for t in rl]
    assert len(TerminalTaskset(_config(split="tmax", tmax_source=src, num_tasks=3))) == 3
    with pytest.raises(TypeError):
        len(TerminalTaskset(_config(split="eval")))
    with pytest.raises(ValueError):
        TerminalTaskset(_config(split="eval")).task_at(0)


async def test_grading_setup_runs_the_bundle_in_the_grade_role(tmax_env):
    src, _ = tmax_env
    task = _load(src, num_tasks=1)[0]
    box = SetupBox()
    await task.grading_setup(box)
    assert f"bash {tmax.SETUP_DIR}/setup.sh grade" in box.runs[-1][-1]
    assert task.setup_role == "agent"


async def test_a_task_without_a_separate_verifier_has_nothing_to_prepare():
    task = TerminalTask(HarborData(idx=0, name="x", prompt="p", image="i"))
    box = SetupBox()
    await task.grading_setup(box)
    assert box.runs == [] and box.files == {}


def _record_grading(monkeypatch):
    calls = []

    async def stage(self, runtime, wipe=False):
        calls.append(("stage", wipe))

    async def graded(self, runtime, trace):
        calls.append("graded")
        return 1.0

    monkeypatch.setattr(TerminalTask, "_stage_tests", stage)
    monkeypatch.setattr(TerminalTask, "_graded", graded)
    return calls


async def test_the_reward_stages_the_tests_then_grades_in_the_pristine_box(tmax_env,
                                                                           monkeypatch):
    src, _ = tmax_env
    task = _load(src, num_tasks=1)[0]
    calls = _record_grading(monkeypatch)
    assert await task.solved(SANDBOX, trace_of(task)) == 1.0
    assert calls == [("stage", True), "graded"]


async def test_the_reward_never_grades_a_separate_verifier_task_in_another_box(
        tmax_env, monkeypatch):
    """Outside a signed-episode sandbox the runtime is the agent's own box: verifiers'
    refusal stands (grade through the harbor env)."""
    src, _ = tmax_env
    task = _load(src, num_tasks=1)[0]
    calls = _record_grading(monkeypatch)
    for runtime in (DOCKER, object()):
        with pytest.raises(TaskError, match="separate verifier"):
            await task.solved(runtime, trace_of(task))
    assert calls == []


async def test_a_shared_verifier_task_keeps_verifiers_grading(monkeypatch):
    calls = _record_grading(monkeypatch)
    task = TerminalTask(HarborData(idx=0, name="x", prompt="p", image="i"))
    assert await task.solved(DOCKER, trace_of(task)) == 1.0
    assert calls == [("stage", False), "graded"]


async def test_graded_elsewhere_records_nothing(tmax_env):
    src, _ = tmax_env
    task = _load(src, num_tasks=1)[0]
    assert await task.graded_elsewhere().solved(SANDBOX, trace_of(task)) == {}
    assert task._graded_elsewhere is False


async def test_finalize_on_a_sandbox_collects_nothing_in_the_agents_box(tmax_env):
    """The sandbox archives the declared roots itself: verifiers' `collect` would run
    the agent box's own `tar` for nothing (and fail an honest rollout at its own cap)."""
    src, _ = tmax_env
    task = _load(src, num_tasks=1)[0]
    assert task.data.collect == []
    box = SetupBox()
    box.config = SANDBOX.config
    trace = trace_of(task)
    await task.finalize(trace, box)
    assert box.runs == [] and box.files == {}


async def test_finalize_elsewhere_still_collects(tmax_env, monkeypatch):
    src, _ = tmax_env
    task = _load(src, num_tasks=1)[0]
    seen = []

    async def collect(runtime, artifacts):
        seen.append([a.source for a in artifacts])
        return {}

    monkeypatch.setattr("verifiers.v1.tasksets.harbor.taskset.collect", collect)
    box = SetupBox()
    box.config = DOCKER.config
    await task.finalize(trace_of(task), box)
    assert seen == [["/app", "/home/user"]]


async def test_a_collect_hook_is_refused_on_a_sandbox():
    task = TerminalTask(HarborData(
        idx=0, name="x", prompt="p", image="i", verifier=VerifierConfig(),
        collect=[{"command": "true", "timeout_sec": 5}]))
    box = SetupBox()
    box.config = SANDBOX.config
    with pytest.raises(TaskError, match="collect"):
        await task.finalize(trace_of(task), box)
    assert box.runs == []


def test_the_task_pickles_and_declares_what_a_sandbox_accepts(tmax_env):
    src, _ = tmax_env
    task = _load(src, num_tasks=1)[0]
    assert pickle.loads(pickle.dumps(task)).hash == task.hash
    timeout = task.data.timeout
    # The bridge's bounds: a grading step in [35, 810] s, and at least 70 s for a task
    # with grading_setup; setup at most 600 s (None: the bridge's default).
    assert 70 <= max(t for t in (timeout.finalize, timeout.scoring) if t is not None) <= 810
    assert timeout.setup is None or 35 <= timeout.setup <= 600
    assert callable(getattr(task, "grading_setup", None))
    assert [(a.source, a.required, a.exclude) for a in task.data.artifacts] == [
        ("/app", False, []), ("/home/user", False, [])]
    assert set(task.runtime_env().values()) <= set(task.data.env.values())


def test_tmax_is_disjoint_from_terminal_bench():
    manifest = tmax_select.load_manifest()
    assert manifest["decontamination"]["terminal_bench_tasks"] == 90
    kept = tmax_select.kept_tasks(manifest)
    assert not any("decontaminated" in entry.get("reasons", []) for _, entry in kept)
    assert manifest["counts"]["reason"]["decontaminated"] > 0


def test_a_reference_runs_the_solve_script_uploaded_in_chunks():
    script = b"#!/bin/bash\n" + b"echo x\n" * 30_000
    calls = conformance.run_script_calls(script)
    assert all(c[0] == "bash" for c in calls)
    encoded = "".join(c[1]["command"].split(" ")[2] for c in calls[1:-2])
    assert base64.b64decode(encoded) == script
    assert all(len(c[1]["command"]) < 100_000 for c in calls)
    assert calls[-1][1]["command"].startswith(f"bash {conformance.SCRIPT}")


def test_every_reference_task_gets_declared_attacks():
    cases = conformance.task_cases(3, b"#!/bin/bash\ntrue\n")
    assert [c["expect"] for c in cases].count(1.0) == 1
    assert {c["name"] for c in cases if c["expect"] == 0.0} == {
        "conftest_in_the_roots_3", "usercustomize_verdict_writer_3", "sitecustomize_in_app_3"}
    assert all(c["index"] == 3 and c["calls"] for c in cases)


@pytest.fixture
def cases_source(tmax_env, monkeypatch):
    src, _ = tmax_env
    monkeypatch.setattr(conformance, "TMAX_SOURCE", src)
    conformance._taskset.cache_clear()
    yield src
    conformance._taskset.cache_clear()


def test_the_cases_cover_task_zero_and_a_task_whose_test_runs_programs(cases_source):
    src = cases_source
    order = sorted(IDS[:4], key=tmax_select.order_key)
    (src / order[2] / tmax.FINAL_TEST).write_text(
        "import subprocess\n\ndef test_out():\n    subprocess.run(['true'], check=True)\n")
    assert conformance.separation_index("tmax") == 2
    cases = conformance.conformance_cases("tmax")
    assert {c["index"] for c in cases} == {0, 2}
    assert len(cases) == 8
    assert conformance.conformance_cases("eval") == []
    assert conformance.conformance_cases("train") == []


def test_without_such_a_task_the_cases_cover_task_zero(cases_source):
    assert conformance.separation_index("tmax") is None
    assert {c["index"] for c in conformance.conformance_cases("tmax")} == {0}


def test_reference_calls_replay_the_tasks_recorded_solution(cases_source):
    calls = conformance.reference_calls("tmax", 1)
    task = TerminalTaskset(_config(split="tmax", tmax_source=cases_source)).task_at(1)
    solve = (grading.Path(task.data.task_dir) / "solution" / "solve.sh").read_bytes()
    assert calls == conformance.run_script_calls(solve)
    assert b"cat .truth.json > out.txt" in solve
    assert conformance.reference_calls("eval", 0) is None


def test_the_package_exports_the_hooks_and_one_taskset():
    assert reliquary_terminal.conformance_cases is conformance.conformance_cases
    assert reliquary_terminal.reference_calls is conformance.reference_calls
    from verifiers.v1.taskset import Taskset
    exported = [getattr(reliquary_terminal, n) for n in reliquary_terminal.__all__]
    assert sum(isinstance(e, type) and issubclass(e, Taskset) for e in exported) == 1
