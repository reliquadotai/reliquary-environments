"""reliquary-terminal's sandbox declarations and hooks, without boxes."""

import base64
import io
import json
import tarfile

import pytest
from sandbox_fakes import ScriptedRuntime

from reliquary_terminal import grading, sandbox, taskset
from reliquary_terminal.grading import TerminalTask

MIB = 1024**2
DIGEST = "xiaomimimo/mimo-v2.6-rl-oss@sha256:" + "c" * 64
PYTHONPATH_ROW = "candidate-1682-science-physics"


def row(**overrides):
    values = dict(instance_id="candidate-x", docker_image="general-agent-env-1:oss", cwd="/app",
                  cpus=1, memory_mb=2048, storage_mb=10240, agent_timeout_sec=900.0,
                  allow_internet=False, problem_statement="Do the thing.\n",
                  tests_files={"test.sh": base64.b64encode(b"exit 0\n").decode()})
    values.update(overrides)
    return values


def state_archive(roots, members):
    """An archive in the sandbox's state format: the manifest, then `<i>` / `<i>/<rel>`
    members. `members` maps a name to None (a directory), bytes (a file) or
    ("link", target)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        manifest = json.dumps({"version": 1, "roots": roots}).encode()
        head = tarfile.TarInfo(".reliquary-archive.json")
        head.size = len(manifest)
        tar.addfile(head, io.BytesIO(manifest))
        for name, value in members.items():
            info = tarfile.TarInfo(name)
            if value is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif isinstance(value, tuple):
                info.type, info.linkname = tarfile.SYMTYPE, value[1]
                tar.addfile(info)
            else:
                info.size = len(value)
                tar.addfile(info, io.BytesIO(value))
    return buffer.getvalue()


IN_PROCESS = b"""
import sys
sys.path.insert(0, "/app/src")
from solution import answer

def test_answer():
    assert answer() == 42
"""
SUBPROCESS = b"""
import subprocess

def test_cli():
    assert subprocess.run(["python3", "/app/cli.py"]).returncode == 0
"""


def _files(test_outputs):
    return {"test.sh": base64.b64encode(b"exit 0\n").decode(),
            "test_outputs.py": base64.b64encode(test_outputs).decode()}


APP = state_archive(["/app"], {"0": None, "0/main.py": b"print(1)\n"})


@pytest.fixture
def rows(monkeypatch, tmp_path):
    table = [row(), row(instance_id="candidate-net", allow_internet=True),
             row(instance_id=PYTHONPATH_ROW),
             row(instance_id="candidate-inproc", tests_files=_files(IN_PROCESS)),
             row(instance_id="candidate-subproc", tests_files=_files(SUBPROCESS))]

    def materialized(r):
        (tmp_path / r["instance_id"] / "tests").mkdir(parents=True, exist_ok=True)
        (tmp_path / r["instance_id"] / "tests" / "test.sh").write_text("exit 1\n")
        return tmp_path / r["instance_id"]

    monkeypatch.setattr(sandbox, "load_train_rows", lambda: tuple(table))
    monkeypatch.setattr(taskset, "materialize_tests", materialized)
    monkeypatch.setattr(sandbox, "_mimo_digests",
                        lambda: {f"{taskset.TRAIN_IMAGE_REPOSITORY}:general-agent-env-1": DIGEST})
    return table


def test_a_mimo_declaration_maps_the_rows_resources(rows):
    found = sandbox.declaration("train", 0)
    assert (found.image, found.workdir, found.roots, found.tools) == (
        DIGEST, "/app", ("/app",), ("bash",))
    assert found.grading_workdir == "/" and found.publish_state is False
    assert found.limits == {"cpus": 1.0, "memory_bytes": 2048 * MIB, "disk_bytes": 10240 * MIB,
                            "pids": 1024, "wall_s": 900, "per_call_timeout_s": 180}
    assert sandbox.sandbox_prompt("train", 0) == "Do the thing."


def test_a_row_that_needs_the_network_is_never_served(rows):
    with pytest.raises(ValueError, match="network"):
        sandbox.declaration("train", 1)


def test_a_row_whose_tests_import_from_app_is_never_served(rows):
    with pytest.raises(ValueError, match="PYTHONPATH"):
        sandbox.declaration("train", 2)


def test_exactly_the_rows_whose_test_sh_sets_pythonpath_are_refused():
    """On the pinned rows: the refusal list is the rows whose test.sh puts code under
    /app (the agent's) on PYTHONPATH, so the grader would import the agent's code."""
    setting = {r["instance_id"] for r in taskset.load_train_rows()
               if b"PYTHONPATH" in taskset.tests_files(r)["test.sh"]}
    assert setting == set(sandbox.UNSERVED) and len(setting) == 3


def test_terminal_bench_is_never_served():
    with pytest.raises(ValueError, match="agent's box"):
        sandbox.declaration("eval", 0)


def test_environment_templates_are_refused():
    with pytest.raises(ValueError, match="template"):
        sandbox.literal_env({"TOKEN": "${SECRET}"}, "env")
    assert sandbox.literal_env({"PATH": "/usr/bin:/bin"}, "env") == {"PATH": "/usr/bin:/bin"}


def test_states_round_trip_and_malformed_ones_are_refused():
    state = sandbox.encode_state(["/app"], b"TAR")
    assert sandbox.decode_state(state, ("/app", "/home/user")) == (["/app"], b"TAR")
    assert sandbox.decode_state(sandbox.encode_state([], b""), ("/app",)) == ([], b"")
    for bad in (b"junk", sandbox.STATE_MAGIC + b'{"roots": ["/etc"]}\nTAR',
                sandbox.STATE_MAGIC + b'{"roots": ["/app"]}\n',
                sandbox.STATE_MAGIC + b'{"roots": ["/home/user", "/app"]}\nTAR'):
        with pytest.raises(ValueError):
            sandbox.decode_state(bad, ("/app", "/home/user"))


async def test_extract_archives_the_present_roots_without_running_anything():
    runtime = ScriptedRuntime()
    runtime.dirs |= {"/app", "/home/user"}
    state = await sandbox.extract(runtime, roots=("/app", "/home/user"))
    assert runtime.runs == []
    assert runtime.reads == [("/app", 1), ("/home/user", 1)]
    assert runtime.archived == (["/app", "/home/user"], sandbox.MAX_ARCHIVE_BYTES)
    assert sandbox.decode_state(state, ("/app", "/home/user"))[0] == ["/app", "/home/user"]


async def test_extract_lists_a_deleted_root_as_absent():
    """`rm -rf /app` (also a dangling `/app` link: the helper's `[ -e ]` is false):
    the agent's box has no workdir left, and extract never needs one."""
    runtime = ScriptedRuntime()
    state = await sandbox.extract(runtime, roots=("/app",))
    assert runtime.runs == [] and runtime.archived is None
    assert sandbox.decode_state(state, ("/app",)) == ([], b"")


@pytest.mark.parametrize("probe", [b"a file", PermissionError("/app"), OSError(5, "odd")])
async def test_extract_keeps_a_root_it_cannot_read_as_a_file(probe):
    runtime = ScriptedRuntime()
    runtime.files["/app"] = probe
    state = await sandbox.extract(runtime, roots=("/app",))
    assert sandbox.decode_state(state, ("/app",))[0] == ["/app"]


async def test_extract_never_mistakes_a_deadline_for_an_absent_root():
    runtime = ScriptedRuntime()
    runtime.files["/app"] = TimeoutError("the step deadline passed")
    with pytest.raises(TimeoutError):
        await sandbox.extract(runtime, roots=("/app",))


@pytest.fixture
def graded(monkeypatch):
    calls = []

    async def setup(self, runtime):
        calls.append(("setup", getattr(self, "setup_role", None)))

    async def stage(self, runtime, wipe=False):
        calls.append(("stage", wipe))

    async def graded_(self, runtime, trace):
        await runtime.run(grading.VERIFIER, {})  # as the real `_graded` does
        trace.info["grading"] = {"exit_code": 0, "output_tail": "ok", "ctrf": None}
        trace.record_metrics({"tests_passed": 3.0})
        return calls_result.pop(0)

    calls_result = [1.0]
    monkeypatch.setattr(TerminalTask, "setup", setup)
    monkeypatch.setattr(TerminalTask, "_stage_tests", stage)
    monkeypatch.setattr(TerminalTask, "_graded", graded_)
    return calls, calls_result


async def test_grade_removes_an_absent_root_and_restores_the_rest(rows, graded):
    calls, _ = graded
    data = sandbox.declaration("train", 0).data
    runtime = ScriptedRuntime(workdir="/")
    archive = state_archive(["/home/user"], {"0": None, "0/notes": b"x"})
    state = sandbox.encode_state(["/home/user"], archive)
    result = await sandbox.grade(runtime, state, data=data, roots=("/app", "/home/user"))
    assert ["rm", "-rf", "--", "/app"] in [argv for argv, _ in runtime.runs]
    assert runtime.restored == (archive, ["/home/user"])
    assert calls == [("setup", "grade"), ("stage", True)]
    assert result.reward == 1.0 and result.facts["metrics"] == {"tests_passed": 3.0}


async def test_a_deleted_app_is_graded_zero_by_the_real_grading(rows):
    """No fake grading: the task's own setup, staging and `_graded` on a box without
    /app. test.sh fails, writes no reward: 0, with no state error."""
    data = sandbox.declaration("train", 0).data
    runtime = ScriptedRuntime(workdir="/")
    runtime.on(lambda argv: argv == ["bash", "/tests/test.sh"], exit_code=1)
    result = await sandbox.grade(runtime, sandbox.encode_state([], b""), data=data,
                                 roots=("/app",))
    assert runtime.restored is None
    assert result.reward == 0.0 and result.facts["grading"]["exit_code"] == 1
    assert "state_error" not in result.facts and "link_into_grader" not in result.facts


async def test_a_reward_outside_zero_one_is_graded_zero(rows, graded):
    _, results = graded
    results[:] = [5.0]
    data = sandbox.declaration("train", 0).data
    result = await sandbox.grade(ScriptedRuntime(workdir="/"), sandbox.encode_state(["/app"], APP),
                                 data=data, roots=("/app",))
    assert result.reward == 0.0 and "reward_out_of_range" in result.facts


async def test_a_malformed_state_is_ours_and_aborts(rows):
    """extract wrote the state; a state it cannot have written is never the agent's."""
    from reliquary_sandbox.episode_task import EnvInfraError

    data = sandbox.declaration("train", 0).data
    with pytest.raises(EnvInfraError, match="not a reliquary-terminal state"):
        await sandbox.grade(ScriptedRuntime(workdir="/"), b"junk", data=data, roots=("/app",))


async def test_clearing_an_absent_root_that_fails_is_ours(rows, graded):
    """Before any of the agent's state is restored, the pristine box is ours."""
    from reliquary_sandbox.episode_task import EnvInfraError

    data = sandbox.declaration("train", 0).data
    runtime = ScriptedRuntime(workdir="/")
    runtime.on(lambda argv: argv[:2] == ["rm", "-rf"], exit_code=1)
    with pytest.raises(EnvInfraError, match="could not remove /app"):
        await sandbox.grade(runtime, sandbox.encode_state([], b""), data=data, roots=("/app",))


async def test_our_unparseable_test_file_is_ours_and_found_before_the_restore(rows, graded,
                                                                              tmp_path):
    from reliquary_sandbox.episode_task import EnvInfraError

    data = sandbox.declaration("train", 0).data
    (tmp_path / "candidate-x" / "tests" / "test_outputs.py").write_text("def broken(:\n")
    runtime = ScriptedRuntime(workdir="/")
    with pytest.raises(EnvInfraError, match="test files"):
        await sandbox.grade(runtime, sandbox.encode_state(["/app"], APP), data=data,
                            roots=("/app",))
    assert runtime.restored is None


async def test_grading_that_never_ran_test_sh_is_ours_never_a_reward(rows, graded, monkeypatch):
    from reliquary_sandbox.episode_task import EnvInfraError

    async def skips_test_sh(self, runtime, trace):
        return 1.0

    monkeypatch.setattr(TerminalTask, "_graded", skips_test_sh)
    data = sandbox.declaration("train", 0).data
    with pytest.raises(EnvInfraError, match="stop_processes"):
        await sandbox.grade(ScriptedRuntime(workdir="/"), sandbox.encode_state(["/app"], APP),
                            data=data, roots=("/app",))


def test_served_indexes_are_the_rows_not_refused(rows):
    assert sandbox.served_indexes("train") == [0, 4]
    with pytest.raises(ValueError):
        sandbox.served_indexes("eval")


async def test_a_restore_failure_is_left_to_the_sandbox(rows, graded):
    """The sandbox records what it refused (state_unreadable, graded 0) or what failed
    on its side (aborted); grade does not second-guess it."""

    class Refusing(ScriptedRuntime):
        async def restore_archive(self, data, roots):
            raise OSError("archive: symlink '0' resolves outside the roots")

    data = sandbox.declaration("train", 0).data
    with pytest.raises(OSError, match="outside the roots"):
        await sandbox.grade(Refusing(workdir="/"), sandbox.encode_state(["/app"], APP),
                            data=data, roots=("/app",))


@pytest.mark.parametrize("members, flagged", [
    ({"0": None, "0/t": ("link", "/tests/test.sh")}, ["/app/t"]),
    ({"0": None, "0/r": ("link", "/logs/verifier/reward.txt")}, ["/app/r"]),
    ({"0": None, "0/d": ("link", "/usr/../tests")}, ["/app/d"]),
    ({"0": ("link", "/tests")}, ["/app"]),
    ({"0": None, "0/a": ("link", "b/reward.txt"), "0/b": ("link", "/logs/verifier")},
     ["/app/a", "/app/b"]),
    ({"0": None, "0/c": ("link", "d"), "0/d": ("link", "c")}, ["/app/c", "/app/d"]),
    # Magic links: /proc/self/root is `/` again, /dev/fd/N any file the reader holds.
    ({"0": None, "0/p": ("link", "/proc/self/root/tests/test_outputs.py")}, ["/app/p"]),
    ({"0": None, "0/f": ("link", "/dev/fd/3")}, ["/app/f"]),
    ({"0": None, "0/s": ("link", "/sys/kernel")}, ["/app/s"]),
    ({"0": None, "0/q": ("link", "../proc/1/cwd")}, ["/app/q"]),
])
def test_links_into_the_graders_files_are_flagged(members, flagged):
    assert sandbox.links_into_grader(state_archive(["/app"], members), ["/app"]) == flagged


async def test_a_link_into_the_graders_files_is_graded_zero(rows, graded):
    calls, _ = graded
    data = sandbox.declaration("train", 0).data
    members = {"0": None, "0/t": ("link", "/tests/test.sh")}
    state = sandbox.encode_state(["/app"], state_archive(["/app"], members))
    result = await sandbox.grade(ScriptedRuntime(workdir="/"), state, data=data, roots=("/app",))
    assert result.reward == 0.0 and result.facts == {"link_into_grader": ["/app/t"]}
    assert calls == [("setup", "grade")]  # tests never staged, never run


async def test_other_absolute_links_are_kept(rows, graded):
    """venvs need them: `.venv/bin/python -> /usr/bin/python3`, and uv's or pyenv's
    interpreters live under /root."""
    calls, _ = graded
    data = sandbox.declaration("train", 0).data
    members = {"0": None, "0/.venv": None, "0/.venv/bin": None,
               "0/.venv/bin/python": ("link", "/usr/bin/python3"),
               "0/.venv/bin/python3": ("link", "python"), "0/tmp": ("link", "/tmp/testsuite"),
               "0/.venv/bin/uvpy": ("link",
                                    "/root/.local/share/uv/python/cpython-3.12/bin/python3.12"),
               "0/pyenv": ("link", "/root/.pyenv/versions/3.12.4/bin/python")}
    state = sandbox.encode_state(["/app"], state_archive(["/app"], members))
    result = await sandbox.grade(ScriptedRuntime(workdir="/"), state, data=data, roots=("/app",))
    assert result.reward == 1.0 and calls[-1] == ("stage", True)


@pytest.mark.parametrize("source, evidence", [
    (IN_PROCESS, "sys.path"),
    (b"import importlib.util\nspec = importlib.util.spec_from_file_location('m', '/app/m.py')\n",
     "spec_from_file_location"),
    (b"namespace = {}\nexec(compile(open('/app/x.py').read(), 'x', 'exec'), namespace)\n",
     "exec"),
    (b"import importlib\nm = importlib.import_module('pkg')\n", "import_module"),
    (b"from app.module import f\n", "import app"),
    (b"def broken(:\n", "unparseable"),
])
def test_in_process_agent_code_is_found_statically(source, evidence):
    found = sandbox.in_process_agent_code({"test_outputs.py": source, "test.sh": b"exit 0\n"})
    assert any(evidence in item for item in found)


def test_subprocess_only_tests_are_not_in_process():
    assert sandbox.in_process_agent_code({"test_outputs.py": SUBPROCESS}) == []


def test_a_row_running_agent_code_in_pytest_is_never_served(rows):
    """The agent's code inside pytest's process can write a complete passing report and
    `os._exit(0)`: no check on the report survives that."""
    assert sandbox.EXCLUDE_IN_PROCESS_AGENT_CODE is True
    with pytest.raises(ValueError, match="in_process_agent_code"):
        sandbox.declaration("train", 3)
    assert sandbox.declaration("train", 4).image == DIGEST  # subprocess only: served


def test_the_pinned_rows_left_after_the_in_process_exclusion():
    rows = taskset.load_train_rows()
    tagged = {r["instance_id"] for r in rows
              if sandbox.in_process_agent_code(taskset.tests_files(r))}
    served = [r["instance_id"] for r in rows
              if sandbox._refusal(r) is None]
    assert len(tagged) == 45 and len(served) == 17
    assert "candidate-0674-ml-evaluation" in tagged  # its loader is a fixture, not the test file
    assert not tagged & set(served)


async def test_stray_processes_are_stopped_before_the_reward_is_read(rows):
    """After test.sh and before reward.txt or the report is read, every process left in
    the grading box is killed: none can rewrite either once test.sh is done."""
    data = sandbox.declaration("train", 0).data
    runtime = ScriptedRuntime(workdir="/")
    runtime.on(lambda argv: argv == ["bash", "/tests/test.sh"], exit_code=1)
    await sandbox.grade(runtime, sandbox.encode_state([], b""), data=data, roots=("/app",))
    test_sh = runtime.log.index(("run", ["bash", "/tests/test.sh"]))
    assert runtime.log[test_sh + 1] == ("stop_processes",)
    assert runtime.log.count(("stop_processes",)) == 1
    reads = [i for i, event in enumerate(runtime.log) if event[0] == "read"]
    assert reads and min(reads) > test_sh + 1
    result = await sandbox.grade(runtime, sandbox.encode_state([], b""), data=data,
                                 roots=("/app",))
    assert result.facts["stop_ran"] is True and "_grader" not in result.facts


def test_a_sandbox_without_stop_processes_is_refused_at_import(monkeypatch):
    """A gateway on a reliquary-sandbox before e0217aa fails when it loads the env,
    not on every episode."""
    import importlib

    from reliquary_sandbox import episode_task

    class Old:
        async def run(self, argv, env): ...

    with pytest.raises(ImportError, match="stop_processes"):
        sandbox.require_stop_processes(Old)
    sandbox.require_stop_processes(episode_task.TaskRuntime, None)
    monkeypatch.delattr(episode_task.TaskRuntime, "stop_processes")
    try:
        with pytest.raises(ImportError, match="e0217aa"):
            importlib.reload(sandbox)
    finally:
        monkeypatch.undo()
        importlib.reload(sandbox)


def test_every_served_row_writes_the_report_its_reward_needs():
    """On the pinned rows: test.sh runs `tests/test_outputs.py` with
    `--ctrf /logs/verifier/ctrf.json`, and the tests collected statically are the ones a
    reward of 1 must show as passed."""
    for index, r in enumerate(taskset.load_train_rows()):
        if sandbox._refusal(r) is not None:
            continue
        test_sh = taskset.tests_files(r)["test.sh"]
        assert b"--ctrf /logs/verifier/ctrf.json" in test_sh, r["instance_id"]
        assert b"test_outputs.py" in test_sh, r["instance_id"]
        data = sandbox.declaration("train", index).data
        tests = grading.collected_tests(data.task_dir)
        assert tests and {key[0] for key in tests} == {"test_outputs.py"}, r["instance_id"]


def test_sandbox_task_builds_the_gateways_contract(rows):
    from reliquary_sandbox import episode_task
    task = sandbox.sandbox_task("train", 0)
    assert isinstance(task, episode_task.SandboxTask)
    assert task.effective_grading_workdir == "/" and task.tools == ("bash",)
    assert task.limits.cpus == 1.0 and task.limits.memory_bytes == 2048 * MIB
    assert task.limits.pids == 1024


def test_an_unpinned_image_is_never_silently_left_out(rows, monkeypatch):
    monkeypatch.setattr(sandbox, "_mimo_digests", lambda: {})
    with pytest.raises(ValueError, match="no pinned digest"):
        sandbox.declaration("train", 0)
    with pytest.raises(ValueError, match="no pinned digest"):
        sandbox.sandbox_images("train")


def test_the_image_list_skips_only_the_refused_rows(rows):
    rows.append(row(instance_id="candidate-y", docker_image="general-agent-env-2:oss"))
    with pytest.raises(ValueError, match="no pinned digest"):
        sandbox.sandbox_images("train")
    rows.pop()
    assert sandbox.sandbox_images("train") == [DIGEST]


@pytest.mark.parametrize("update", [{"image": "another/image:tag"}, {"env": {"A": "1"}},
                                    {"healthcheck": {"command": "true"}}])
def test_a_verifier_box_unlike_the_agents_is_refused(rows, monkeypatch, update):
    """The sandbox grades in a pristine box of the agent's image, with the task's env."""
    real = taskset.train_data

    def changed(row, idx, config):
        data = real(row, idx, config)
        return data.model_copy(update={"verifier": data.verifier.model_copy(update=update)})

    monkeypatch.setattr(sandbox, "train_data", changed)
    with pytest.raises(ValueError, match="not servable"):
        sandbox.declaration("train", 0)


# -- the verifiers this module was checked against --------------------------

PINNED_URL = ('{"url": "https://github.com/PrimeIntellect-ai/verifiers.git", "vcs_info": '
              '{"vcs": "git", "commit_id": "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"}}')


def test_the_installed_verifiers_is_the_pinned_one():
    sandbox.require_pinned_verifiers(sandbox.installed_direct_url(),
                                     sandbox.installed_verifiers_sources())
    assert set(sandbox.PINNED_VERIFIERS_MODULES) == {"verifiers.v1.tasksets.harbor.taskset",
                                                     "verifiers.v1.tasksets.harbor.env"}


@pytest.mark.parametrize("direct_url", [None, "", "{}", PINNED_URL.replace("b2e4", "0000")])
def test_verifiers_from_another_commit_is_refused(direct_url):
    with pytest.raises(ImportError, match="b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"):
        sandbox.require_pinned_verifiers(direct_url, sandbox.installed_verifiers_sources())


@pytest.mark.parametrize("module", ["verifiers.v1.tasksets.harbor.taskset",
                                    "verifiers.v1.tasksets.harbor.env"])
def test_a_changed_verifiers_module_is_refused(module):
    sources = dict(sandbox.installed_verifiers_sources())
    sources[module] += b"\n# patched\n"
    with pytest.raises(ImportError, match=module.replace(".", r"\.")):
        sandbox.require_pinned_verifiers(PINNED_URL, sources)


def test_the_verifiers_check_runs_at_import(monkeypatch):
    import importlib
    import importlib.metadata

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    try:
        with pytest.raises(ImportError, match="verifiers"):
            importlib.reload(sandbox)
    finally:
        monkeypatch.undo()
        importlib.reload(sandbox)
