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


APP = state_archive(["/app"], {"0": None, "0/main.py": b"print(1)\n"})


@pytest.fixture
def rows(monkeypatch, tmp_path):
    table = [row(), row(instance_id="candidate-net", allow_internet=True),
             row(instance_id=PYTHONPATH_ROW)]

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
        trace.info["grading"] = {"exit_code": 0, "output_tail": "ok", "ctrf": None}
        trace.record_metrics({"tests_passed": 3.0})
        return calls_result.pop(0)

    calls_result = [1.0]
    monkeypatch.setattr(TerminalTask, "setup", setup)
    monkeypatch.setattr(TerminalTask, "_stage_tests", stage)
    monkeypatch.setattr(TerminalTask, "_graded", graded_)
    return calls, calls_result


async def test_grade_removes_an_absent_root_and_restores_the_rest(rows, graded):
    pytest.importorskip("reliquary_sandbox.episode_task")
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
    pytest.importorskip("reliquary_sandbox.episode_task")
    data = sandbox.declaration("train", 0).data
    runtime = ScriptedRuntime(workdir="/")
    runtime.on(lambda argv: argv == ["bash", "/tests/test.sh"], exit_code=1)
    result = await sandbox.grade(runtime, sandbox.encode_state([], b""), data=data,
                                 roots=("/app",))
    assert runtime.restored is None
    assert result.reward == 0.0 and result.facts["grading"]["exit_code"] == 1
    assert "state_error" not in result.facts and "link_into_grader" not in result.facts


async def test_a_reward_outside_zero_one_is_graded_zero(rows, graded):
    pytest.importorskip("reliquary_sandbox.episode_task")
    _, results = graded
    results[:] = [5.0]
    data = sandbox.declaration("train", 0).data
    result = await sandbox.grade(ScriptedRuntime(workdir="/"), sandbox.encode_state(["/app"], APP),
                                 data=data, roots=("/app",))
    assert result.reward == 0.0 and "reward_out_of_range" in result.facts


async def test_a_malformed_state_is_our_bug_and_raises(rows):
    """extract wrote the state; a state it cannot have written is never the agent's."""
    pytest.importorskip("reliquary_sandbox.episode_task")
    data = sandbox.declaration("train", 0).data
    with pytest.raises(ValueError):
        await sandbox.grade(ScriptedRuntime(workdir="/"), b"junk", data=data, roots=("/app",))


async def test_a_restore_failure_is_left_to_the_sandbox(rows, graded):
    """The sandbox records what it refused (state_unreadable, graded 0) or what failed
    on its side (aborted); grade does not second-guess it."""
    pytest.importorskip("reliquary_sandbox.episode_task")

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
async def test_a_link_into_the_graders_files_is_graded_zero(rows, graded, members, flagged):
    pytest.importorskip("reliquary_sandbox.episode_task")
    calls, _ = graded
    data = sandbox.declaration("train", 0).data
    state = sandbox.encode_state(["/app"], state_archive(["/app"], members))
    result = await sandbox.grade(ScriptedRuntime(workdir="/"), state, data=data, roots=("/app",))
    assert result.reward == 0.0 and result.facts == {"link_into_grader": flagged}
    assert calls == [("setup", "grade")]  # tests never staged, never run


async def test_other_absolute_links_are_kept(rows, graded):
    """venvs need them: `.venv/bin/python -> /usr/bin/python3`, and uv's or pyenv's
    interpreters live under /root."""
    pytest.importorskip("reliquary_sandbox.episode_task")
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


async def test_stray_processes_are_stopped_before_the_reward_is_read(rows):
    """After test.sh and before reward.txt or the report is read, every process left in
    the grading box is killed: none can rewrite either once test.sh is done."""
    pytest.importorskip("reliquary_sandbox.episode_task")
    data = sandbox.declaration("train", 0).data
    runtime = ScriptedRuntime(workdir="/")
    runtime.on(lambda argv: argv == ["bash", "/tests/test.sh"], exit_code=1)
    await sandbox.grade(runtime, sandbox.encode_state([], b""), data=data, roots=("/app",))
    test_sh = runtime.log.index(("run", ["bash", "/tests/test.sh"]))
    assert runtime.log[test_sh + 1] == ("stop_processes",)
    assert runtime.log.count(("stop_processes",)) == 1
    reads = [i for i, event in enumerate(runtime.log) if event[0] == "read"]
    assert reads and min(reads) > test_sh + 1


def test_every_served_row_writes_the_report_its_reward_needs():
    """On the pinned rows: test.sh runs `tests/test_outputs.py` with
    `--ctrf /logs/verifier/ctrf.json`, and the tests collected statically are the ones a
    reward of 1 must show as passed."""
    for index, r in enumerate(taskset.load_train_rows()):
        if r["instance_id"] in sandbox.UNSERVED:
            continue
        test_sh = taskset.tests_files(r)["test.sh"]
        assert b"--ctrf /logs/verifier/ctrf.json" in test_sh, r["instance_id"]
        assert b"test_outputs.py" in test_sh, r["instance_id"]
        data = sandbox.declaration("train", index).data
        tests = grading.collected_tests(data.task_dir)
        assert tests and {key[0] for key in tests} == {"test_outputs.py"}, r["instance_id"]


def test_sandbox_task_builds_the_gateways_contract(rows):
    episode_task = pytest.importorskip("reliquary_sandbox.episode_task")
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
