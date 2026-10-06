"""reliquary-terminal's sandbox declarations and hooks, without boxes."""

import base64

import pytest
from sandbox_fakes import ScriptedRuntime

from reliquary_terminal import sandbox, taskset
from reliquary_terminal.grading import TerminalTask

MIB = 1024**2
DIGEST = "xiaomimimo/mimo-v2.6-rl-oss@sha256:" + "c" * 64


def row(**overrides):
    values = dict(instance_id="candidate-x", docker_image="general-agent-env-1:oss", cwd="/app",
                  cpus=1, memory_mb=2048, storage_mb=10240, agent_timeout_sec=900.0,
                  allow_internet=False, problem_statement="Do the thing.\n",
                  tests_files={"test.sh": base64.b64encode(b"exit 0\n").decode()})
    values.update(overrides)
    return values


@pytest.fixture
def rows(monkeypatch, tmp_path):
    table = [row(), row(instance_id="candidate-net", allow_internet=True)]
    monkeypatch.setattr(sandbox, "load_train_rows", lambda: tuple(table))
    monkeypatch.setattr(taskset, "materialize_tests", lambda r: tmp_path / r["instance_id"])
    monkeypatch.setattr(sandbox, "_mimo_digests",
                        lambda: {f"{taskset.TRAIN_IMAGE_REPOSITORY}:general-agent-env-1": DIGEST})
    return table


def test_a_mimo_declaration_maps_the_rows_resources(rows):
    found = sandbox.declaration("train", 0)
    assert (found.image, found.workdir, found.roots, found.tools) == (
        DIGEST, "/app", ("/app",), ("bash",))
    assert found.grading_workdir == "/" and found.publish_state is False
    assert found.limits == {"cpus": 1.0, "memory_bytes": 2048 * MIB, "disk_bytes": 10240 * MIB,
                            "wall_s": 900, "per_call_timeout_s": 180}
    assert sandbox.sandbox_prompt("train", 0) == "Do the thing."


def test_a_row_that_needs_the_network_is_never_served(rows):
    with pytest.raises(ValueError, match="network"):
        sandbox.declaration("train", 1)


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


async def test_extract_archives_the_present_roots():
    runtime = ScriptedRuntime()
    runtime.on(lambda argv: argv[:2] == ["sh", "-c"], stdout="/app\n/home/user\n")
    state = await sandbox.extract(runtime, roots=("/app", "/home/user"))
    assert runtime.archived == (["/app", "/home/user"], sandbox.MAX_ARCHIVE_BYTES)
    assert sandbox.decode_state(state, ("/app", "/home/user"))[0] == ["/app", "/home/user"]


async def test_extract_lists_a_deleted_root_as_absent():
    runtime = ScriptedRuntime()
    runtime.on(lambda argv: argv[:2] == ["sh", "-c"], stdout="")
    state = await sandbox.extract(runtime, roots=("/app",))
    assert runtime.archived is None
    assert sandbox.decode_state(state, ("/app",)) == ([], b"")


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
    state = sandbox.encode_state(["/home/user"], b"TAR")
    result = await sandbox.grade(runtime, state, data=data, roots=("/app", "/home/user"))
    assert ["rm", "-rf", "--", "/app"] in [argv for argv, _ in runtime.runs]
    assert runtime.restored == (b"TAR", ["/home/user"])
    assert calls == [("setup", "grade"), ("stage", True)]
    assert result.reward == 1.0 and result.facts["metrics"] == {"tests_passed": 3.0}


async def test_a_reward_outside_zero_one_is_graded_zero(rows, graded):
    pytest.importorskip("reliquary_sandbox.episode_task")
    _, results = graded
    results[:] = [5.0]
    data = sandbox.declaration("train", 0).data
    result = await sandbox.grade(ScriptedRuntime(workdir="/"), sandbox.encode_state(["/app"], b"T"),
                                 data=data, roots=("/app",))
    assert result.reward == 0.0 and "reward_out_of_range" in result.facts


async def test_a_malformed_state_is_graded_zero_not_raised(rows):
    pytest.importorskip("reliquary_sandbox.episode_task")
    data = sandbox.declaration("train", 0).data
    result = await sandbox.grade(ScriptedRuntime(workdir="/"), b"junk", data=data, roots=("/app",))
    assert result.reward == 0.0 and "state_error" in result.facts


def test_sandbox_task_builds_the_gateways_contract(rows):
    episode_task = pytest.importorskip("reliquary_sandbox.episode_task")
    task = sandbox.sandbox_task("train", 0)
    assert isinstance(task, episode_task.SandboxTask)
    assert task.effective_grading_workdir == "/" and task.tools == ("bash",)
    assert task.limits.cpus == 1.0 and task.limits.memory_bytes == 2048 * MIB


def test_the_presence_check_keeps_symlinked_roots_and_drops_deleted_ones(tmp_path):
    """The script `extract` runs, run here by the real sh: a root the agent deleted is
    absent (graded with the root removed), a root it replaced by a symlink, even a
    dangling one, is present (archived as the link, as verifiers' collect does)."""
    import subprocess

    kept, linked, dangling = tmp_path / "kept", tmp_path / "linked", tmp_path / "dangling"
    kept.mkdir()
    linked.symlink_to("/etc")
    dangling.symlink_to(tmp_path / "nowhere")
    roots = [str(kept), str(tmp_path / "deleted"), str(linked), str(dangling)]
    listed = subprocess.run(["sh", "-c", sandbox._PRESENT, "roots", *roots],
                            capture_output=True, text=True, check=True)
    assert listed.stdout.splitlines() == [str(kept), str(linked), str(dangling)]


async def test_a_state_the_grading_box_refuses_is_graded_zero_not_raised(rows, graded):
    """restore_archive refuses an archive (a relative symlink leaving the roots, ...)
    with an OSError: the agent's outcome, graded 0, never an env failure."""
    pytest.importorskip("reliquary_sandbox.episode_task")
    calls, _ = graded

    class Refusing(ScriptedRuntime):
        async def restore_archive(self, data, roots):
            raise OSError("archive: symlink '0' resolves outside the roots")

    data = sandbox.declaration("train", 0).data
    result = await sandbox.grade(Refusing(workdir="/"), sandbox.encode_state(["/app"], b"T"),
                                 data=data, roots=("/app",))
    assert result.reward == 0.0 and "outside the roots" in result.facts["state_error"]
    assert calls == [("setup", "grade")]


def test_an_unpinned_image_is_never_silently_left_out(rows, monkeypatch):
    monkeypatch.setattr(sandbox, "_mimo_digests", lambda: {})
    with pytest.raises(ValueError, match="no pinned digest"):
        sandbox.declaration("train", 0)
    with pytest.raises(ValueError, match="no pinned digest"):
        sandbox.sandbox_images("train")


def test_the_image_list_skips_only_rows_that_need_the_network(rows):
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
