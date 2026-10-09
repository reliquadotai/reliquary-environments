"""reliquary-swe on the env norm (signed-episode sandboxes): state only in the box, the
patch as verifiers' convention artifact, a reward that grades the restored patch in a
pristine box, timeouts the sandbox accepts, one task by index, plain-data conformance
cases. No Docker, no dataset download (synthetic rows)."""

import base64
import dataclasses
import pickle
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from reliquary_swe import conformance, corpus, grading
from reliquary_swe.taskset import (
    BASE_REF,
    PATCH_PATH,
    UNTRACKED_FILE,
    SweTask,
    SweTaskset,
    task_for,
)


def row(i=0, split_image="r@sha256:" + "a" * 64):
    return corpus.SweRow(instance_id=f"inst-{i}", repo="o/r", problem_statement=f"bug {i}",
                         fail_to_pass=("t.py::a",), pass_to_pass=(), gold_patch="g",
                         base_commit="HEAD", image=split_image)


def a_task(split="r2e"):
    return task_for(row(), 0, split)


class Box:
    def __init__(self, files=None, base="b45e\n", captured=b"diff --git a/x b/x\n"):
        self.runs, self.files, self.captured = [], dict(files or {}), captured
        self.base, self.fail_capture = base, False

    async def run(self, argv, env):
        self.runs.append((list(argv), dict(env)))
        if argv[:3] == ["rm", "-f", "--"]:
            for path in argv[3:]:
                self.files.pop(path, None)
        if argv[:2] == ["git", "rev-parse"] and BASE_REF in argv:
            return SimpleNamespace(exit_code=0 if self.base else 1, stdout=self.base, stderr="")
        if argv[:2] == ["sh", "-c"] and "git ls-files --others" in argv[2]:
            return SimpleNamespace(exit_code=0, stdout="build/\0run_tests.sh\0", stderr="")
        if argv[:2] == ["sh", "-c"] and "git add -A" in argv[2] and self.fail_capture:
            return SimpleNamespace(exit_code=1, stdout="", stderr="index.lock exists")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def read(self, path, max_bytes=None):
        if path.startswith("/tmp/vf_agent_patch_"):
            return self.captured
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write(self, path, data):
        self.files[path] = bytes(data)


def trace(task):
    return vf.Trace(agent=vf.AgentInfo(config=vf.AgentConfig()),
                    task=vf.TraceTask(type="SweTask", data=task.data))


async def test_setup_keeps_the_base_ref_and_the_untracked_list_in_the_box():
    task, box = a_task(), Box()
    await task.setup(trace(task), box)
    script = box.runs[0][0][2]
    assert script.endswith(f'&& git -C "$WORKDIR" update-ref {BASE_REF} HEAD')
    assert box.files[f"/testbed/{UNTRACKED_FILE}"] == b"build/\0run_tests.sh"
    assert not hasattr(SweTask, "_heads") and not hasattr(SweTask, "_untracked")  # no host memory


async def test_finalize_captures_against_the_base_in_the_box():
    task = a_task()
    box = Box(files={f"/testbed/{UNTRACKED_FILE}": b"build/\0run_tests.sh"})
    await task.finalize(trace(task), box)
    capture = next((argv, env) for argv, env in box.runs
                   if argv[:2] == ["sh", "-c"] and "git add -A" in argv[2])
    assert capture[1]["VF_DIFF_BASE"] == "b45e" and capture[0][-2:] == ["build/", "run_tests.sh"]
    assert box.files[PATCH_PATH] == b"diff --git a/x b/x\n"


async def test_finalize_without_a_base_ref_captures_against_head():
    # No host memory to fall back on: a box whose base ref is gone (the agent deleted
    # it) is diffed against bare HEAD, which only drops the agent's own commits.
    task, box = a_task(), Box(base="")
    await task.finalize(trace(task), box)
    assert [env["VF_DIFF_BASE"] for _, env in box.runs if "VF_DIFF_BASE" in env] == ["HEAD"]


async def test_finalize_removes_a_planted_patch_before_it_captures():
    task = a_task()
    box = Box(files={PATCH_PATH: b"a patch the agent planted"})
    box.fail_capture = True  # git refuses: capture_patch writes nothing
    await task.finalize(trace(task), box)
    assert PATCH_PATH not in box.files


async def test_two_boxes_of_one_task_keep_their_own_base():
    task = a_task()
    first, second = Box(base="aaaa\n"), Box(base="bbbb\n")
    await task.finalize(trace(task), first)
    await task.finalize(trace(task), second)
    bases = [next(env["VF_DIFF_BASE"] for argv, env in box.runs if "VF_DIFF_BASE" in env)
             for box in (first, second)]
    assert bases == ["aaaa", "bbbb"]


async def test_the_reward_grades_the_restored_patch_in_the_prepared_box(monkeypatch):
    seen = {}

    async def prepared_data(runtime, data):
        return data.model_copy(update={"base_commit": "f00d"})

    async def grade_prepared(runtime, data, patch):
        seen.update(base=data.base_commit, patch=patch)
        return grading.Report(1.0, True, True, 1, 0, {}, results_parsed=1,
                              test_command_exit_code=0)

    monkeypatch.setattr(grading, "prepared_data", prepared_data)
    monkeypatch.setattr(grading, "grade_prepared", grade_prepared)
    task, box = a_task(), Box(files={PATCH_PATH: b"diff \xff"})
    t = trace(task)
    assert await task.patch_passes_tests(box, t) == 1.0
    assert seen == {"base": "f00d", "patch": b"diff \xff"}
    assert t.metrics["applied"] == 1.0 and t.metrics["fail_to_pass_total"] == 1.0


async def test_a_missing_patch_is_graded_as_an_empty_one(monkeypatch):
    seen = {}

    async def prepared_data(runtime, data):
        return data

    async def grade_prepared(runtime, data, patch):
        seen["patch"] = patch
        return grading.Report(0.0, True, True, 0, 0, {})

    monkeypatch.setattr(grading, "prepared_data", prepared_data)
    monkeypatch.setattr(grading, "grade_prepared", grade_prepared)
    task = a_task()
    assert await task.patch_passes_tests(Box(), trace(task)) == 0.0
    assert seen["patch"] == b""


async def test_graded_elsewhere_records_nothing_and_leaves_the_original_alone():
    task = a_task()
    assert await task.graded_elsewhere().patch_passes_tests(Box(), trace(task)) == {}
    assert task._graded_elsewhere is False


async def test_grading_setup_is_the_pristine_box_half(monkeypatch):
    calls = []

    async def prepare_box(runtime, data):
        calls.append(data.instance_id)
        return data

    monkeypatch.setattr(grading, "prepare_box", prepare_box)
    await a_task().grading_setup(Box())
    assert calls == ["inst-0"]


def test_the_task_pickles_and_declares_what_a_sandbox_accepts():
    task = a_task()
    assert pickle.loads(pickle.dumps(task)).hash == task.hash
    timeout = task.data.timeout
    assert timeout.setup <= 600 and 35 <= timeout.finalize <= 810 and 35 <= timeout.scoring <= 810
    assert task.data.resources.memory == 4 and task.data.resources.disk == 10
    assert task.data.artifacts == []  # the patch rides verifiers' /logs/artifacts sweep
    assert task.runtime_env() == {}


def test_task_at_is_the_index_th_task_of_load(monkeypatch):
    rows = tuple(row(i) for i in range(3))
    monkeypatch.setattr(corpus, "load_polyglot_rows", lambda num_tasks=None: rows)
    config = vf.taskset_config_type("reliquary-swe")(id="reliquary-swe", split="polyglot")
    ts = SweTaskset(config)
    loaded = list(ts)
    assert len(ts) == 3
    assert [ts.task_at(i).hash for i in range(3)] == [t.hash for t in loaded]
    with pytest.raises(IndexError):
        ts.task_at(3)


def test_a_reference_call_set_applies_the_patch_in_chunks():
    patch = b"x" * (3 * conformance.CHUNK) + b"\xff"
    calls = conformance.apply_calls(patch, "/testbed")
    encoded = "".join(c[1]["command"].split(" ")[2] for c in calls[1:-2])
    assert base64.b64decode(encoded) == patch
    assert all(len(c[1]["command"]) < 100_000 for c in calls)
    assert "git apply" in calls[-1][1]["command"]


def test_the_cases_of_a_golden_are_one_reference_and_declared_attacks():
    cases = conformance.golden_cases(7, b"diff --git a/x b/x\n", "/testbed", python=True)
    assert [c["expect"] for c in cases].count(1.0) == 1
    assert {c["name"] for c in cases if c["expect"] == 0.0} == {
        "gold_with_forced_ignored_path_7", "gold_with_symlink_7", "gold_with_binary_file_7",
        "patch_planted_in_artifacts_7", "conftest_forcing_passes_7"}
    assert all(c["index"] == 7 and "@" not in c["name"] for c in cases)


async def test_an_unreadable_patch_grades_zero_without_grading(monkeypatch):
    class Unreadable(Box):
        async def read(self, path, max_bytes=None):
            if path == PATCH_PATH:
                raise OSError(27, "File too large")  # FileTooLarge on a sandbox
            return await super().read(path, max_bytes)

    async def must_not_run(*args):
        raise AssertionError("graded an unreadable patch")

    monkeypatch.setattr(grading, "prepared_data", must_not_run)
    task = a_task()
    t = trace(task)
    assert await task.patch_passes_tests(Unreadable(), t) == 0.0
    assert t.metrics["patch_unreadable"] == 1.0


async def test_the_env_runs_the_agent_on_a_task_graded_elsewhere():
    from reliquary_swe.env import SweEnv

    ran = []

    async def run(task):
        ran.append(task)

    agents = SimpleNamespace(agent=SimpleNamespace(run=run))
    task = a_task()
    await SweEnv.run(object.__new__(SweEnv), task, agents)  # run reads no config
    assert ran[0]._graded_elsewhere is True and ran[0].hash == task.hash


SYMLINK_GOLD = ("diff --git a/link b/link\nnew file mode 120000\n--- /dev/null\n"
                "+++ b/link\n@@ -0,0 +1 @@\n+target\n\\ No newline at end of file\n")


def test_a_gold_patch_the_norm_refuses_is_no_reference(monkeypatch):
    symlinked = corpus.SweRow(instance_id="r2e__x__0", repo="x", problem_statement="p",
                              fail_to_pass=(), pass_to_pass=(), gold_patch=SYMLINK_GOLD,
                              base_commit="HEAD", image="r@sha256:" + "b" * 64)
    plain = dataclasses.replace(symlinked, gold_patch="diff --git a/x b/x\n")
    monkeypatch.setattr(corpus, "r2e_row_at", lambda index: symlinked)
    assert conformance.reference_calls("r2e", 0) is None
    monkeypatch.setattr(corpus, "r2e_row_at", lambda index: plain)
    assert conformance.reference_calls("r2e", 0) is not None
    assert conformance.reference_calls("eval", 0) is None
