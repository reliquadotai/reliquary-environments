"""reliquary-terminal's tmax splits on the env norm (signed-episode sandboxes): a task
hash that does not depend on where the cache lives, the grade-role setup as the pristine
box's preparation, a reward that grades there, one task by index, plain-data cases."""

import base64
import os
import pickle
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from test_tmax_split import IDS, SetupBox, _config, _load, tmax_env  # noqa: F401 (fixture)
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.errors import TaskError
from verifiers.v1.tasksets.harbor.taskset import HarborData, VerifierConfig

import reliquary_terminal
from reliquary_terminal import conformance, grading, tmax, tmax_select
from reliquary_terminal.grading import TerminalTask, task_dir_digest
from reliquary_terminal.taskset import TerminalTaskset, UnservedTask, refuse_unserved

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


def test_the_hash_still_follows_the_tasks_content(tmax_env, monkeypatch):
    """Same cache, same task_dir: only the content changes."""
    src, _ = tmax_env
    first = _load(src, num_tasks=1)[0]
    before = first.hash
    monkeypatch.setattr(tmax, "TEST_SH", tmax.TEST_SH + "# changed\n")
    second = _load(src, num_tasks=1)[0]
    assert second.data.task_dir == first.data.task_dir
    assert second.hash != before


def test_a_task_dir_without_a_stamp_is_digested_from_its_files(tmp_path):
    one, two = tmp_path / "one", tmp_path / "elsewhere" / "two"
    for root in (one, two):
        (root / "tests").mkdir(parents=True)
        (root / "tests" / "test.sh").write_bytes(b"exit 0\n")
    assert task_dir_digest(str(one)) == task_dir_digest(str(two))
    (two / "tests" / "test.sh").write_bytes(b"exit 1\n")
    assert task_dir_digest(str(one)) != task_dir_digest(str(two))


def test_a_stamp_is_trusted_only_while_the_files_match_it(tmax_env):
    src, _ = tmax_env
    task_dir = Path(_load(src, num_tasks=1)[0].data.task_dir)
    stamp = (task_dir / ".content").read_text()
    assert task_dir_digest(str(task_dir)) == stamp == tmax.tree_digest(task_dir)
    (task_dir / "tests" / "test.sh").write_text("exit 0\n")
    tmax._VERIFIED.clear()
    with pytest.raises(ValueError, match="stamp"):
        task_dir_digest(str(task_dir))
    # A stale cache is rewritten on the next load.
    _load(src, num_tasks=1)
    assert task_dir_digest(str(task_dir)) == stamp


def test_a_missing_task_dir_has_no_digest(tmp_path):
    with pytest.raises(FileNotFoundError):
        task_dir_digest(str(tmp_path / "gone"))


def test_concurrent_materialize_keeps_the_winner(tmax_env, tmp_path, monkeypatch):
    """Two writers of one task: both stage, then both rename. The second finds the
    first's directory with the same content and keeps it."""
    src, _ = tmax_env
    converted = tmax.convert(tmax.Source(src), IDS[0])
    root = tmp_path / "race"
    barrier = threading.Barrier(2)
    real_rename = os.rename
    winners = []

    def rename(a, b):
        if ".partial-" in str(a):
            barrier.wait(timeout=10)
            real_rename(a, b)
            winners.append(os.stat(b).st_ino)
            return
        real_rename(a, b)

    monkeypatch.setattr(tmax.os, "rename", rename)
    with ThreadPoolExecutor(2) as pool:
        dirs = list(pool.map(lambda _: tmax.materialize(converted, root), range(2)))
    assert dirs[0] == dirs[1]
    assert len(winners) == 1 and os.stat(dirs[0]).st_ino == winners[0]
    tmax._VERIFIED.clear()
    assert tmax.stamp_of(dirs[0]) == tmax._content_digest(converted)
    assert sorted(p.name for p in root.iterdir()) == [IDS[0]]


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
    assert task.runtime_env() == {"LC_ALL": "C.UTF-8", "PATH": "/opt/tool/bin:" + tmax.BASE_PATH}


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


def _uploaded(calls):
    """{path: bytes} of what a case's calls write."""
    out, staged = {}, ""
    for _, args in calls:
        command = args["command"]
        if command.startswith("printf %s "):
            staged += command.split(" ")[2]
        elif " && base64 -d " in command:
            out[command.split(" > ")[1].split(" && ")[0]] = base64.b64decode(staged)
            staged = ""
    return out


def test_every_reference_task_gets_canaries_that_only_leave_a_marker():
    cases = conformance.task_cases(3, b"#!/bin/bash\ntrue\n")
    assert [c["expect"] for c in cases].count(1.0) == 1
    canaries = {c["name"]: c for c in cases if c["expect"] == 0.0}
    assert set(canaries) == {
        "conftest_in_the_roots_3", "usercustomize_in_the_user_site_3",
        "sitecustomize_in_app_3", "programs_in_the_user_bin_3"}
    for name, case in canaries.items():
        assert case["index"] == 3 and case["absent"] == [f"/tmp/.canary-{name}"]
        files = _uploaded(case["calls"])
        assert files and all(path.startswith(("/app/", "/home/user/")) for path in files)
        for content in files.values():
            assert content in (conformance.python_canary(case["absent"][0]),
                               conformance.program_canary(case["absent"][0]))
    assert set(_uploaded(canaries["programs_in_the_user_bin_3"]["calls"])) == {
        "/home/user/.local/bin/bash", "/home/user/.local/bin/python3"}
    assert canaries["programs_in_the_user_bin_3"]["calls"][-1][1]["command"].startswith(
        "chmod 755 ")
    assert "/app/sitecustomize.py" in _uploaded(canaries["sitecustomize_in_app_3"]["calls"])


def test_a_canary_does_nothing_but_create_its_marker(tmp_path):
    # The whole of each canary: one statement creating the marker.
    assert conformance.python_canary("/tmp/.canary-x") == b"open('/tmp/.canary-x', 'a').close()\n"
    assert conformance.program_canary("/tmp/.canary-x") == b"#!/bin/sh\n: >> '/tmp/.canary-x'\n"
    mark = tmp_path / "m"
    program = tmp_path / "prog"
    program.write_bytes(conformance.program_canary(str(mark)))
    program.chmod(0o755)
    assert subprocess.run([str(program)], check=False).returncode == 0 and mark.exists()
    mark.unlink()
    (tmp_path / "canary_mod.py").write_bytes(conformance.python_canary(str(mark)))
    subprocess.run([sys.executable, "-I", "-c", "import canary_mod"], cwd=tmp_path, check=False,
                   capture_output=True)
    assert not mark.exists()  # -I: not imported from the cwd
    subprocess.run([sys.executable, "-c", "import canary_mod"], cwd=tmp_path, check=True)
    assert mark.exists()


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
    assert len(cases) == 10
    for split in ("eval", "train", "tmax_rl", "tmax_sft"):
        assert conformance.conformance_cases(split) == []


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
    assert conformance.reference_calls("tmax_rl", 0) is None


def test_the_package_exports_the_hooks_and_one_taskset():
    assert reliquary_terminal.conformance_cases is conformance.conformance_cases
    assert reliquary_terminal.reference_calls is conformance.reference_calls
    from verifiers.v1.taskset import Taskset
    exported = [getattr(reliquary_terminal, n) for n in reliquary_terminal.__all__]
    assert sum(isinstance(e, type) and issubclass(e, Taskset) for e in exported) == 1


# --------------------------------------------------------------------------
# The grading box never runs what the agent left in its roots.
# --------------------------------------------------------------------------


class LocalBox:
    """Runs commands on this machine, applying `env` over the box's own (`self.env`)
    as a sandbox's runtime does."""

    def __init__(self, env):
        self.env = dict(env)
        self.config = SANDBOX.config
        self.runs = []

    async def run(self, argv, env):
        self.runs.append(list(argv))
        merged = {"HOME": os.environ.get("HOME", "/root"), **self.env, **env}
        done = subprocess.run(argv, env=merged, capture_output=True, text=True, check=False)
        return ProgramResult(done.returncode, done.stdout, done.stderr)


@pytest.fixture
def agent_root(tmp_path, monkeypatch):
    """A directory standing for /home/user, with the programs-in-the-user-bin canary in
    its .local/bin and a sitecustomize canary on PYTHONPATH."""
    root = tmp_path / "home-user"
    bin_dir = root / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    marks = tmp_path / "marks"
    marks.mkdir()
    for name in ("bash", "python3", "sh", "rm", "env"):
        (bin_dir / name).write_bytes(conformance.program_canary(str(marks / name)))
        (bin_dir / name).chmod(0o755)
    (root / "app").mkdir()
    (root / "app" / "sitecustomize.py").write_bytes(
        conformance.python_canary(str(marks / "sitecustomize")))
    monkeypatch.setattr(tmax, "ARTIFACT_ROOTS", (str(root),))
    env = {"PATH": f"{bin_dir}:" + tmax.BASE_PATH, "PYTHONPATH": str(root / "app"),
           "BASH_ENV": str(root / ".bashrc"), "LC_ALL": "C.UTF-8"}
    (root / ".bashrc").write_bytes(conformance.program_canary(str(marks / "bash_env"))[10:])
    return env, marks


async def test_root_commands_never_run_the_agents_programs(agent_root):
    env, marks = agent_root
    box = LocalBox(env)
    hardened = grading._RootCommands(box, env)
    for argv in (["bash", "-c", "python3 -c pass && rm -f /nonexistent"],
                 ["sh", "-c", "bash -c true; python3 -c pass"],
                 ["rm", "-f", "/nonexistent"]):
        result = await hardened.run(argv, {})
        assert result.exit_code == 0, result.stderr
    assert sorted(p.name for p in marks.iterdir()) == []
    assert all(run[0] == "/usr/bin/env" for run in box.runs)
    # The same commands without the fixed PATH and unset variables run the canaries.
    await box.run(["bash", "-c", "python3 -c pass"], {})
    assert "bash" in {p.name for p in marks.iterdir()}


def test_root_argv_keeps_a_tasks_own_safe_values():
    env = {"PATH": "/opt/tool/bin:" + tmax.BASE_PATH, "PYTHONPATH": "/opt/lib",
           "LD_PRELOAD": "/home/user/x.so", "LC_ALL": "C.UTF-8"}
    argv = tmax.root_argv(["bash", "/tests/test.sh"], env)
    assert argv[0] == "/usr/bin/env" and argv[-2:] == ["/bin/bash", "/tests/test.sh"]
    assert "PATH=/opt/tool/bin:" + tmax.BASE_PATH in argv and "PYTHONPATH=/opt/lib" in argv
    unset = {argv[i + 1] for i, a in enumerate(argv) if a == "-u"}
    assert unset == {"LD_PRELOAD", "LD_LIBRARY_PATH", "BASH_ENV", "ENV", "PYTHONHOME",
                     "PYTHONSTARTUP"}
    assert f"PATH={tmax.BASE_PATH}" in tmax.root_argv(["sh"], {"PATH": "/app/bin:/usr/bin"})
    assert f"PATH={tmax.BASE_PATH}" in tmax.root_argv(["sh"], {"PATH": "bin:/usr/bin"})
    assert f"PATH={tmax.BASE_PATH}" in tmax.root_argv(["sh"], {})


def test_root_argv_pins_home_and_the_user_site():
    for env in ({}, {"HOME": "/home/user"}, {"HOME": "/app/h", "PATH": "/opt/x/bin"}):
        argv = tmax.root_argv(["bash", "/tests/test.sh"], env)
        assert f"HOME={tmax.GRADE_HOME}" in argv and "PYTHONNOUSERSITE=1" in argv, env
        assert not [a for a in argv if a.startswith("HOME=") and a != f"HOME={tmax.GRADE_HOME}"]
    assert not tmax._under_roots(tmax.GRADE_HOME)


def test_root_argv_unsets_every_loader_and_python_variable_off_the_allow_list():
    env = {"LD_AUDIT": "/lib/a.so", "LD_DEBUG": "all", "PYTHONINSPECT": "1",
           "PYTHONWARNINGS": "x", "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "0",
           "LC_ALL": "C"}
    argv = tmax.root_argv(["sh", "-c", "true"], env)
    unset = {argv[i + 1] for i, a in enumerate(argv) if a == "-u"}
    assert {"LD_AUDIT", "LD_DEBUG", "PYTHONINSPECT", "PYTHONWARNINGS"} <= unset
    assert not unset & {"PYTHONUNBUFFERED", "PYTHONHASHSEED", "LC_ALL", "PYTHONNOUSERSITE"}


def test_the_tests_python_ignores_the_user_site(tmp_path):
    """test.sh runs its helper with `-I`, and the helper runs pytest with `-I`: a
    usercustomize in the home's user site is never imported."""
    from reliquary_terminal import tmax_box

    assert tmax_box.test_argv("/r")[:2] == ["/usr/bin/python3", "-I"]
    assert "python3 -I /tests/tmax_box.py relay" in tmax.TEST_SH
    assert "python3 -I /tests/tmax_box.py run-tests" in tmax.TEST_SH
    env = {"HOME": str(tmp_path), "PATH": tmax.BASE_PATH}
    site = subprocess.run(["/usr/bin/python3", "-c", "import site;print(site.getusersitepackages())"],
                          env=env, capture_output=True, text=True, check=True).stdout.strip()
    Path(site).mkdir(parents=True)
    mark = tmp_path / "mark"
    (Path(site) / "usercustomize.py").write_bytes(conformance.python_canary(str(mark)))
    for flag in ("-s", "-I"):
        subprocess.run(["/usr/bin/python3", flag, "-c", "pass"], env=env, check=True)
        assert not mark.exists(), flag
    subprocess.run(["/usr/bin/python3", "-c", "pass"], env=env, check=True)
    assert mark.exists()


async def test_grading_runs_every_root_command_through_the_guard(tmax_env, monkeypatch):
    src, _ = tmax_env
    task = _load(src, num_tasks=1)[0]
    box = SetupBox()
    box.config = SANDBOX.config
    box.env = {"PATH": "/home/user/.local/bin:" + tmax.BASE_PATH}

    async def read(path, max_bytes=None):
        raise FileNotFoundError(path)

    box.read = read
    await task.grading_setup(box)
    assert await task.solved(box, trace_of(task)) == 0.0
    assert len(box.runs) >= 4
    for argv in box.runs:
        assert argv[0] == "/usr/bin/env" and f"PATH={tmax.BASE_PATH}" in argv, argv
        assert f"HOME={tmax.GRADE_HOME}" in argv and "PYTHONNOUSERSITE=1" in argv, argv
        program = argv[argv.index("PYTHONNOUSERSITE=1") + 1]
        assert program.startswith("/"), argv
    assert any(argv[-2:] == ["/bin/bash", "/tests/test.sh"] for argv in box.runs)


# --------------------------------------------------------------------------
# Tasks no grading box may run are refused at load.
# --------------------------------------------------------------------------


def test_a_task_whose_environment_points_at_the_roots_is_refused(tmax_env):
    src, _ = tmax_env
    order = sorted(IDS[:4], key=tmax_select.order_key)
    definition = (src / order[1] / "container.def").read_text()
    (src / order[1] / "container.def").write_text(definition.replace(
        "export PATH=/opt/tool/bin:$PATH",
        "export PATH=/opt/tool/bin:$PATH\nexport PATH=/home/user/.local/bin:$PATH"))
    loaded = _load(src)
    assert [t.data.idx for t in loaded] == [0, 2, 3]
    taskset = TerminalTaskset(_config(split="tmax", tmax_source=src))
    with pytest.raises(UnservedTask, match="environment_in_artifact_roots"):
        taskset.task_at(1)
    assert taskset.task_at(2).data.name == order[2]
    facts = tmax_select.task_facts(tmax.Source(src), order[1])
    assert facts.reasons["environment_in_artifact_roots"] == "PATH"
    assert "environment_in_artifact_roots" not in tmax_select.task_facts(
        tmax.Source(src), order[0]).reasons


def test_env_in_artifact_roots():
    assert tmax.env_in_artifact_roots({"PATH": "/opt/x/bin:" + tmax.BASE_PATH,
                                       "PYTHONUNBUFFERED": "1", "LC_ALL": "C"}) == []
    assert tmax.env_in_artifact_roots({
        "PATH": "/usr/bin:/home/user/.local/bin", "LD_PRELOAD": "/lib/a.so /app/b.so",
        "PYTHONUSERBASE": "/home/user/.local", "BASH_ENV": "/usr/../home/user/rc",
        "PYTHONPATH": "src", "ENV": "/etc/shrc"}) == [
        "BASH_ENV", "LD_PRELOAD", "PATH", "PYTHONPATH", "PYTHONUSERBASE"]


def test_env_in_artifact_roots_covers_the_tool_search_variables():
    env = {"HOME": "/home/user", "XDG_CONFIG_HOME": "/app/.config", "GIT_CONFIG_GLOBAL": "/app/g",
           "NODE_OPTIONS": "--require /app/x.js", "NODE_PATH": "/home/user/n",
           "PERL5LIB": "/app/p", "RUBYOPT": "-r/app/x", "JAVA_TOOL_OPTIONS": "-javaagent:/app/a",
           "CLASSPATH": "/lib/a.jar:/app/b.jar", "MALLOC_OPTIONS": "/app/m"}
    assert tmax.env_in_artifact_roots(env) == sorted(env)
    assert tmax.env_in_artifact_roots({
        "HOME": "/root", "XDG_CACHE_HOME": "/var/cache", "NODE_OPTIONS": "--max-old-space-size=4096",
        "CLASSPATH": "/opt/j.jar", "GIT_AUTHOR_NAME": "x", "JAVA_HOME": "/usr/lib/jvm"}) == []


def test_a_task_with_collect_hooks_is_refused_at_load():
    data = HarborData(idx=0, name="x", prompt="p", image="i", verifier=VerifierConfig(),
                      collect=[{"command": "true", "timeout_sec": 5}])
    with pytest.raises(UnservedTask, match="collect"):
        refuse_unserved(data)
    assert refuse_unserved(data.model_copy(update={"collect": []})).name == "x"
