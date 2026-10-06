"""reliquary-swe's sandbox declarations and hooks, without boxes (see test_sandbox_parity.py
for the live goldens)."""

import pytest
from sandbox_fakes import ScriptedRuntime

from reliquary_swe import corpus, grading, sandbox, taskset

GIB = 1024**3
ROW = corpus.SweRow(instance_id="oauthlib__x", repo="oauthlib/oauthlib",
                    problem_statement="Fix it.", fail_to_pass=("t::a",), pass_to_pass=(),
                    gold_patch="", base_commit="origin/oauthlib__x~1",
                    image="jyangballin/swesmith.x86_64.o@sha256:" + "a" * 64)
DATA = taskset.task_for(ROW, 3, "train").data


def _train_row(split, index):
    return sandbox.SplitRef("train", 20), ROW


def test_cleanup_script_is_what_setup_runs():
    assert taskset.cleanup_script("train").startswith("set -e")
    assert taskset._TRAIN_GUARD_AND_REROOT in taskset.cleanup_script("train")
    assert taskset._TRAIN_GUARD_AND_REROOT not in taskset.cleanup_script("r2e")
    assert taskset._R2E_HIDE_TESTS in taskset.cleanup_script("r2e")
    assert taskset.cleanup_script("polyglot").endswith(taskset._STRIP_AND_GC)


def test_split_names():
    assert sandbox.parse_split("train") == sandbox.SplitRef("train", corpus.DEFAULT_SWESMITH_IMAGES)
    assert sandbox.parse_split("train:5") == sandbox.SplitRef("train", 5)
    assert sandbox.parse_split("r2e") == sandbox.SplitRef("r2e")
    for bad in ("train:", "train:05", "train:x", "polyglot:3", "", "TRAIN"):
        with pytest.raises(ValueError):
            sandbox.parse_split(bad)


def test_the_evaluation_set_is_never_served():
    with pytest.raises(ValueError, match="evaluation"):
        sandbox.parse_split("eval")


def test_a_declaration_maps_the_row(monkeypatch):
    monkeypatch.setattr(sandbox, "row_for", _train_row)
    found = sandbox.declaration("train", 3)
    assert (found.image, found.workdir, found.tools) == (ROW.image, "/testbed", ("bash", "edit"))
    assert found.grading_workdir is None and found.publish_state is True
    assert found.limits == {"memory_bytes": 4 * GIB, "disk_bytes": 10 * GIB, "pids": 1024,
                            "wall_s": 3600, "per_call_timeout_s": 600}
    assert found.grading_timeout_s <= 810
    assert found.data == DATA


def test_the_prompt_is_the_tasksets(monkeypatch):
    monkeypatch.setattr(sandbox, "row_for", _train_row)
    assert sandbox.sandbox_prompt("train", 3) == DATA.prompt


def test_a_tag_only_image_needs_a_pinned_digest(monkeypatch):
    row = corpus.SweRow(instance_id="p", repo="", problem_statement="", fail_to_pass=(),
                        pass_to_pass=(), gold_patch="", image="xiaomimimo/mimo-v2.6-rl-oss:p")
    monkeypatch.setattr(sandbox, "_polyglot_digests", lambda: {})
    with pytest.raises(ValueError, match="pinned digest"):
        sandbox.image_of(row)
    pinned = "xiaomimimo/mimo-v2.6-rl-oss@sha256:" + "b" * 64
    monkeypatch.setattr(sandbox, "_polyglot_digests", lambda: {row.image: pinned})
    assert sandbox.image_of(row) == pinned


def test_polyglot_tasks_index_one_cached_load(monkeypatch):
    # A gateway resolves many indexes: one load of the split, never one cached prefix per index.
    rows = tuple(corpus.SweRow(instance_id=f"t{i}", repo="", problem_statement="",
                               fail_to_pass=(), pass_to_pass=(), gold_patch="",
                               image=f"repo:t{i}") for i in range(3))
    calls = []
    monkeypatch.setattr(sandbox.corpus, "load_polyglot_rows",
                        lambda num_tasks: calls.append(num_tasks) or rows)
    assert sandbox.row_for("polyglot", 2) == (sandbox.SplitRef("polyglot"), rows[2])
    with pytest.raises(IndexError):
        sandbox.row_for("polyglot", 3)
    assert calls == [None, None]


async def test_prepare_runs_the_setup_cleanup_then_records_base_and_untracked():
    runtime = ScriptedRuntime()
    runtime.on(lambda argv: argv[:2] == ["sh", "-c"] and "ls-files" in argv[2],
               stdout="run_tests.sh\0install.sh\0")
    await sandbox.prepare(runtime, data=DATA)
    argv, env = runtime.runs[0]
    assert argv[2].startswith(taskset.cleanup_script("train"))
    assert argv[2].endswith(f'git -C "$WORKDIR" update-ref {sandbox.BASE_REF} HEAD')
    assert env == {"BASE_COMMIT": DATA.base_commit, "WORKDIR": "/testbed"}
    assert runtime.files["/testbed/.git/reliquary-untracked"] == b"run_tests.sh\0install.sh"


async def test_a_failed_cleanup_fails_the_open():
    runtime = ScriptedRuntime()
    runtime.on(lambda argv: argv[:2] == ["sh", "-c"], exit_code=1, stderr="not a Bug Patch")
    with pytest.raises(RuntimeError, match="preparation failed"):
        await sandbox.prepare(runtime, data=DATA)


def _capture(seen, patch="diff --git a/x b/x\n", error=None):
    async def fake_capture(trace, runtime, base_commit="", env=None, write_path=None, ignore=None):
        seen.update(base=base_commit, ignore=ignore)
        if error:
            trace.info["patch_error"] = error
        else:
            trace.info["patch"] = patch
    return fake_capture


async def test_extract_diffs_against_the_recorded_base_not_head(monkeypatch):
    runtime = ScriptedRuntime()
    runtime.files["/testbed/.git/reliquary-untracked"] = b"run_tests.sh\0"
    runtime.on(lambda argv: argv[:2] == ["git", "rev-parse"], stdout="abc123\n")
    seen = {}
    monkeypatch.setattr(sandbox.vf, "capture_patch", _capture(seen))
    state = await sandbox.extract(runtime, data=DATA)
    assert runtime.runs[0][0] == ["git", "rev-parse", "--verify", "-q", sandbox.BASE_REF]
    assert seen == {"base": "abc123", "ignore": ["run_tests.sh"]}
    assert state == b"diff --git a/x b/x\n"


async def test_extract_without_base_or_list_falls_back_like_verifiers(monkeypatch):
    runtime = ScriptedRuntime()
    runtime.on(lambda argv: argv[:2] == ["git", "rev-parse"], exit_code=1)
    seen = {}
    monkeypatch.setattr(sandbox.vf, "capture_patch", _capture(seen, error="exit=128 broken"))
    assert await sandbox.extract(runtime, data=DATA) == b""
    assert seen == {"base": "", "ignore": []}


@pytest.mark.parametrize("listed", [OSError(22, "not a regular file"), b"x" * (1024 * 1024 + 1)])
async def test_an_untracked_list_the_agent_broke_is_its_own_outcome(monkeypatch, listed):
    # The sandbox's read raises OSError subclasses (StateUnreadable, FileTooLarge) for a list
    # the agent replaced by a FIFO, a directory or a huge file: a value, never an exception.
    runtime = ScriptedRuntime()
    runtime.files["/testbed/.git/reliquary-untracked"] = listed
    runtime.on(lambda argv: argv[:2] == ["git", "rev-parse"], stdout="abc123\n")
    seen = {}
    monkeypatch.setattr(sandbox.vf, "capture_patch", _capture(seen))
    assert await sandbox.extract(runtime, data=DATA) == b"diff --git a/x b/x\n"
    assert seen == {"base": "abc123", "ignore": []}


async def test_grade_runs_the_packages_grading_and_reports_its_facts(monkeypatch):
    pytest.importorskip("reliquary_sandbox.episode_task")
    received = {}

    async def fake_grade(runtime, data, patch):
        received.update(data=data, patch=patch)
        return grading.Report(1.0, True, True, 1, 0, {"t::a": "PASSED"}, results_parsed=1,
                              test_command_exit_code=0)

    monkeypatch.setattr(sandbox.grading, "grade", fake_grade)
    result = await sandbox.grade(ScriptedRuntime(), b"diff --git a/x b/x\n", data=DATA)
    assert received == {"data": DATA, "patch": "diff --git a/x b/x\n"}
    assert result.reward == 1.0
    assert result.facts["applied"] is True and result.facts["fail_to_pass_total"] == 1
    assert "results" not in result.facts


async def test_a_pristine_image_that_cannot_be_prepared_raises(monkeypatch):
    # grading.grade raises for a box that is not what the corpus says: an env/image bug,
    # graded 0.0 `grading_failed` and counted by the gateway, never swallowed here.
    pytest.importorskip("reliquary_sandbox.episode_task")

    async def broken(runtime, data, patch):
        raise RuntimeError("could not check out")

    monkeypatch.setattr(sandbox.grading, "grade", broken)
    with pytest.raises(RuntimeError):
        await sandbox.grade(ScriptedRuntime(), b"", data=DATA)


def test_sandbox_task_builds_the_gateways_contract(monkeypatch):
    episode_task = pytest.importorskip("reliquary_sandbox.episode_task")
    monkeypatch.setattr(sandbox, "row_for", _train_row)
    task = sandbox.sandbox_task("train", 3)
    assert isinstance(task, episode_task.SandboxTask)
    assert task.limits == episode_task.TaskLimits(**sandbox.LIMITS)
    assert task.publish_state and task.effective_grading_workdir == "/testbed"


def test_r2e_row_at_follows_the_split_order():
    # Downloads the pinned R2E revision once, as tests/test_r2e_corpus.py does.
    assert [corpus.r2e_row_at(i) for i in range(3)] == list(corpus.load_r2e_rows(3))
    assert corpus.r2e_instance_ids()[:3] == [row.instance_id for row in corpus.load_r2e_rows(3)]


@pytest.mark.parametrize("cleanup_tail, expected",
                         [("false && true", 1), ("true && false", 1), ("true && true", 0)])
def test_the_base_ref_never_hides_a_failed_last_check(cleanup_tail, expected):
    # R2E's cleanup ends on `test ! -e A && test ! -e B`, a list `set -e` does not exit on:
    # the base ref must not run (and turn the exit into 0) after a failed check.
    import subprocess

    script = 'git() { echo ran; } ; set -e ; ' + cleanup_tail + sandbox._RECORD_BASE
    done = subprocess.run(["sh", "-c", script], capture_output=True, text=True,
                          env={"WORKDIR": "/nonexistent", "PATH": "/usr/bin:/bin"})
    assert done.returncode == expected
    assert done.stdout == ("ran\n" if expected == 0 else "")


def test_the_polyglot_pin_script_selects_a_prefix_and_named_instances():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "pin_polyglot_digests.py"
    spec = importlib.util.spec_from_file_location("pin_polyglot_digests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [corpus.SweRow(instance_id=f"t{i}", repo="", problem_statement="", fail_to_pass=(),
                          pass_to_pass=(), gold_patch="", image=f"repo:t{i}") for i in range(5)]
    assert module.images(rows, 2, ["t4", "t1"]) == ["repo:t0", "repo:t1", "repo:t4"]
