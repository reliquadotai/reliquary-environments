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
        self.base, self.head, self.fail_capture = base, "c0de\n", False
        self.plant_on_capture, self.rm_exit, self.capture_raises = False, [], False
        self.config = SimpleNamespace(type="reliquary-sandbox", workdir="/testbed")

    async def run(self, argv, env):
        self.runs.append((list(argv), dict(env)))
        if argv[:3] == ["rm", "-f", "--"]:
            if self.rm_exit:
                return SimpleNamespace(exit_code=self.rm_exit.pop(0), stdout="", stderr="busy")
            for path in argv[3:]:
                self.files.pop(path, None)
        if argv[:2] == ["git", "rev-parse"]:
            out = self.base if BASE_REF in argv[-1] else self.head
            return SimpleNamespace(exit_code=0 if out else 1, stdout=out, stderr="")
        if argv[:2] == ["sh", "-c"] and "ls-files --others" in argv[2]:
            return SimpleNamespace(exit_code=0, stdout="build/\0run_tests.sh\0", stderr="")
        if argv[:2] == ["sh", "-c"] and "git add -A" in argv[2]:
            if self.capture_raises:
                raise vf.SandboxError("the box stopped answering")
            if self.plant_on_capture:  # a hook or filter the agent left, had one run
                self.files[PATCH_PATH] = b"a patch planted during the capture"
            if self.fail_capture:
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


def capture_of(box):
    return next((argv, env) for argv, env in box.runs
                if argv[:2] == ["sh", "-c"] and "git add -A" in argv[2])


def trace(task):
    return vf.Trace(agent=vf.AgentInfo(config=vf.AgentConfig()),
                    task=vf.TraceTask(type="SweTask", data=task.data))


async def test_setup_keeps_the_base_ref_and_the_untracked_list_in_the_box():
    task, box = a_task(), Box(captured=b"")
    await task.setup(trace(task), box)
    script = box.runs[0][0][2]
    assert script.endswith(f'&& git -C "$WORKDIR" update-ref {BASE_REF} HEAD')
    assert box.files[f"/testbed/{UNTRACKED_FILE}"] == b"build/\0run_tests.sh"


async def test_setup_lists_untracked_files_under_the_captures_ignore_rules():
    # Same excludes file as the capture (`info/exclude`, regular files only), no global
    # or system config (real git: test_capture_isolation.py).
    task, box = a_task(), Box(captured=b"")
    await task.setup(trace(task), box)
    argv, env = next((a, e) for a, e in box.runs if a[:2] == ["sh", "-c"]
                     and "ls-files --others" in a[2])
    assert argv[-1] == "/testbed" and '-c core.excludesFile="$x"' in argv[2]
    assert '[ -f "$x" ] && [ ! -L "$x" ]' in argv[2]
    assert env["GIT_CONFIG_NOSYSTEM"] == "1" and env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["HOME"].startswith("/tmp/reliquary-no-home-")


async def test_setup_captures_the_untouched_box_with_the_list_it_wrote():
    task, box = a_task(), Box(captured=b"")
    await task.setup(trace(task), box)
    argv, env = capture_of(box)
    assert argv[-2:] == ["build/", "run_tests.sh"]
    assert env["GIT_DIR"].startswith("/tmp/reliquary-capture-")
    assert not any(path.startswith("/logs/") for path in box.files)  # no patch written


@pytest.mark.parametrize("broken", ["diff", "refused"])
async def test_setup_fails_as_ours_when_the_untouched_box_does_not_capture_empty(broken):
    task, box = a_task(), Box(captured=b"diff --git a/x b/x\nold mode 100644\n")
    box.fail_capture = broken == "refused"
    with pytest.raises(RuntimeError, match="untouched box does not capture as an empty"):
        await task.setup(trace(task), box)


async def test_setup_fails_when_the_untracked_list_cannot_be_taken():
    class NoList(Box):
        async def run(self, argv, env):
            if argv[:2] == ["sh", "-c"] and "ls-files --others" in argv[2]:
                return SimpleNamespace(exit_code=128, stdout="", stderr="not a repo")
            return await super().run(argv, env)

    task = a_task()
    with pytest.raises(RuntimeError, match="untracked files"):
        await task.setup(trace(task), NoList(captured=b""))


async def test_finalize_captures_against_the_base_in_the_box():
    task = a_task()
    box = Box(files={f"/testbed/{UNTRACKED_FILE}": b"build/\0run_tests.sh"})
    await task.finalize(trace(task), box)
    argv, env = capture_of(box)
    assert env["VF_DIFF_BASE"] == "b45e" and argv[-2:] == ["build/", "run_tests.sh"]
    assert box.files[PATCH_PATH] == b"diff --git a/x b/x\n"


async def test_finalize_captures_in_a_scratch_repository_of_its_own():
    # The agent's .git (config, hooks, attributes, fsmonitor) is never the capture's
    # repository: only its objects are read, as an alternate (real git:
    # test_capture_isolation.py).
    task, box = a_task(), Box()
    await task.finalize(trace(task), box)
    _, env = capture_of(box)
    scratch = env["GIT_DIR"]
    assert scratch.startswith("/tmp/reliquary-capture-") and env["GIT_WORK_TREE"] == "/testbed"
    assert env["GIT_ALTERNATE_OBJECT_DIRECTORIES"] == "/testbed/.git/objects"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1" and env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["HOME"] == scratch and env["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert env["GIT_NO_LAZY_FETCH"] == "1"
    prepare = next(argv for argv, _ in box.runs if argv[:2] == ["sh", "-c"]
                   and "git init" in argv[2])
    assert prepare[-3:] == [scratch, "/testbed", "b45e"] and "--template=" in prepare[2]
    assert '"$3^{commit}"' in prepare[2]  # peeled in the scratch repository, no remote
    assert ["rm", "-rf", "--", scratch] in [argv for argv, _ in box.runs]
    second = Box()
    await task.finalize(trace(task), second)
    assert capture_of(second)[1]["GIT_DIR"] != scratch  # a host nonce per capture


async def test_the_base_lookup_reads_no_object_and_fetches_nothing():
    # Peeling in the agent's repository would read the object, and a missing one in a
    # partial clone is fetched through the agent's remote (real git:
    # test_capture_isolation.py); HOME is neither the scratch repository nor created.
    task, box = a_task(), Box()
    await task.finalize(trace(task), box)
    lookups = [(argv, env) for argv, env in box.runs if argv[:2] == ["git", "rev-parse"]]
    assert lookups and all(argv[-1] in (BASE_REF, "HEAD") for argv, _ in lookups)
    scratch = capture_of(box)[1]["GIT_DIR"]
    for _, env in lookups:
        assert env["GIT_NO_LAZY_FETCH"] == "1" and env["GIT_DIR"] == "/testbed/.git"
        assert env["HOME"].startswith("/tmp/reliquary-no-home-") and env["HOME"] != scratch


async def test_finalize_without_a_base_ref_captures_against_head():
    # No host memory to fall back on: a box whose base ref is gone (the agent deleted
    # it) is diffed against HEAD, which only drops the agent's own commits.
    task, box = a_task(), Box(base="")
    await task.finalize(trace(task), box)
    assert [env["VF_DIFF_BASE"] for _, env in box.runs if "VF_DIFF_BASE" in env] == ["c0de"]


async def test_finalize_without_any_base_captures_nothing():
    task, box = a_task(), Box(files={PATCH_PATH: b"planted"}, base="")
    box.head = ""
    t = trace(task)
    await task.finalize(t, box)
    assert not any("VF_DIFF_BASE" in env for _, env in box.runs)
    assert PATCH_PATH not in box.files and "no base" in t.info["patch_error"]


async def test_finalize_removes_a_planted_patch_before_it_captures():
    task = a_task()
    box = Box(files={PATCH_PATH: b"a patch the agent planted"})
    box.fail_capture = True  # git refuses: capture_patch writes nothing
    await task.finalize(trace(task), box)
    assert PATCH_PATH not in box.files


async def test_a_patch_planted_during_a_failed_capture_never_travels():
    task, box = a_task(), Box()
    box.plant_on_capture = box.fail_capture = True
    t = trace(task)
    await task.finalize(t, box)
    assert PATCH_PATH not in box.files and "patch" not in t.info and t.info["patch_error"]
    assert [argv for argv, _ in box.runs].count(["rm", "-f", "--", PATCH_PATH]) == 2


async def test_a_patch_planted_during_a_good_capture_is_overwritten():
    task, box = a_task(), Box()
    box.plant_on_capture = True
    await task.finalize(trace(task), box)
    assert box.files[PATCH_PATH] == b"diff --git a/x b/x\n"


@pytest.mark.parametrize("failing", [[1], [0, 1]], ids=["before", "after"])
async def test_a_planted_patch_finalize_cannot_remove_fails_the_extract(failing):
    task, box = a_task(), Box(files={PATCH_PATH: b"planted"})
    box.fail_capture, box.rm_exit = True, list(failing)
    with pytest.raises(RuntimeError, match="could not remove"):
        await task.finalize(trace(task), box)


async def test_a_capture_the_box_stops_answering_raises():
    task, box = a_task(), Box()
    box.capture_raises = True
    with pytest.raises(vf.SandboxError):
        await task.finalize(trace(task), box)


async def test_a_scratch_repository_that_fails_on_a_dead_box_raises():
    class Dead(Box):
        async def run(self, argv, env):
            if argv[:2] == ["sh", "-c"] and "git init" in argv[2] or argv == ["true"]:
                self.runs.append((list(argv), dict(env)))
                return SimpleNamespace(exit_code=137, stdout="", stderr="")
            return await super().run(argv, env)

    task = a_task()
    with pytest.raises(vf.SandboxError):
        await task.finalize(trace(task), Dead())


async def test_a_scratch_repository_git_refuses_is_the_agents_outcome():
    class Refused(Box):
        async def run(self, argv, env):
            if argv[:2] == ["sh", "-c"] and "git init" in argv[2]:
                return SimpleNamespace(exit_code=128, stdout="", stderr="bad object b45e")
            return await super().run(argv, env)

    task, box = a_task(), Refused(files={PATCH_PATH: b"planted"})
    t = trace(task)
    await task.finalize(t, box)
    assert PATCH_PATH not in box.files and "bad object" in t.info["patch_error"]
    assert not any("VF_DIFF_BASE" in env for _, env in box.runs)


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


@pytest.mark.parametrize("box_type", ["docker", "subprocess", None])
async def test_the_reward_never_grades_outside_a_sandbox_runtime(monkeypatch, box_type):
    # The runtime type separates a signed-episode sandbox from any other runtime (the
    # sandbox itself calls the reward only in its grading box).
    async def must_not_run(*args):
        raise AssertionError("graded outside a sandbox")

    monkeypatch.setattr(grading, "prepared_data", must_not_run)
    box = Box(files={PATCH_PATH: b"diff --git a/x b/x\n"})
    if box_type is None:
        del box.config
    else:
        box.config.type = box_type
    task = a_task()
    assert await task.patch_passes_tests(box, trace(task)) == {}
    assert not box.runs


async def test_the_reward_lets_the_step_deadline_through():
    class Late(Box):
        async def read(self, path, max_bytes=None):
            raise TimeoutError

    task = a_task()
    with pytest.raises(TimeoutError):
        await task.patch_passes_tests(Late(), trace(task))


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
    # A sandbox's bounds: setup in [35, 600]; each grading step (extract = finalize,
    # grade = scoring) in [70, 810] with grading_setup.
    assert 35 <= timeout.setup <= 600
    assert 70 <= timeout.finalize <= 810 and 70 <= timeout.scoring <= 810
    assert max(timeout.finalize, timeout.scoring) == 810  # the README's extract budget
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


def test_task_at_and_len_of_r2e(monkeypatch):
    rows = tuple(row(i) for i in range(4))
    monkeypatch.setattr(corpus, "r2e_instance_ids", lambda: [r.instance_id for r in rows])
    monkeypatch.setattr(corpus, "r2e_row_at", lambda index: rows[index])
    monkeypatch.setattr(corpus, "load_r2e_rows", lambda num_tasks=None: rows[:num_tasks])
    make = vf.taskset_config_type("reliquary-swe")
    ts = SweTaskset(make(id="reliquary-swe", split="r2e"))
    assert len(ts) == 4
    assert [ts.task_at(i).hash for i in range(4)] == [t.hash for t in ts]
    capped = SweTaskset(make(id="reliquary-swe", split="r2e", num_tasks=2))
    assert len(capped) == 2 and capped.task_at(1).key == "inst-1"
    with pytest.raises(IndexError):
        capped.task_at(2)
    assert ts.task_at(2).data.split == "r2e"


def test_task_at_and_len_of_train(monkeypatch):
    from reliquary_swe import swesmith_adapter

    rows = tuple(row(i) for i in range(3))
    seen = []
    monkeypatch.setattr(corpus, "swesmith_order",
                        lambda n, m=None: tuple((i, r.instance_id) for i, r in enumerate(rows)))
    monkeypatch.setattr(corpus, "swesmith_row_at",
                        lambda n, index, m=None: seen.append((n, m)) or rows[index])
    monkeypatch.setattr(corpus, "load_swesmith_rows", lambda n, m=None: rows)
    monkeypatch.setattr(swesmith_adapter, "ensure_python_profile", lambda repo: None)
    ts = SweTaskset(vf.taskset_config_type("reliquary-swe")(id="reliquary-swe", split="train"))
    assert len(ts) == 3
    assert [ts.task_at(i).hash for i in range(3)] == [t.hash for t in ts]
    assert seen[0] == (corpus.DEFAULT_SWESMITH_IMAGES, None)
    with pytest.raises(IndexError):
        ts.task_at(-1)


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
        "patch_planted_in_artifacts_7", "patch_planted_by_git_config_7",
        "patch_planted_without_a_repository_7", "conftest_forcing_passes_7"}
    assert all(c["index"] == 7 and "@" not in c["name"] for c in cases)


def commands(case):
    return [call[1]["command"] for call in case["calls"]]


def _forced_venv_command():
    cases = {c["name"]: c for c in conformance.golden_cases(0, b"G", "/testbed", python=False)}
    return commands(cases["gold_with_forced_ignored_path_0"])[-1]


def test_the_forced_venv_path_is_un_ignored_before_it_is_added():
    forced = _forced_venv_command()
    assert forced.index(".venv/.gitignore") < forced.index("git add -f .venv/zz_reliquary.pth")


@pytest.mark.parametrize("root_ignore, venv_ignore", [
    (None, None),
    (".venv/\n", None),  # a root rule on the directory
    (".venv/", "*"),  # no trailing newline anywhere, R2E's own `*`
    ("*.pyc\n.venv", "*.log"),  # last lines without a newline
    (None, "*\n"),
])
def test_the_forced_venv_file_reaches_a_fresh_index_capture(tmp_path, root_ignore, venv_ignore):
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.email=a@b", "-c", "user.name=n",
                               *args], check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    (repo / "src.py").write_text("x = 1\n")
    if root_ignore is not None:
        (repo / ".gitignore").write_text(root_ignore)
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    if venv_ignore is not None:
        (repo / ".venv").mkdir()
        (repo / ".venv" / ".gitignore").write_text(venv_ignore)
    command = _forced_venv_command().replace("cd /testbed", f"cd {repo}", 1)
    subprocess.run(["sh", "-c", command], check=True, capture_output=True)
    # What a capture sees: a fresh index and `git add -A`, which honours every ignore rule.
    env = {**__import__("os").environ, "GIT_INDEX_FILE": str(tmp_path / "fresh-index")}
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
    names = subprocess.run(["git", "-C", str(repo), "ls-files"], check=True, env=env,
                           capture_output=True, text=True).stdout.split()
    assert ".venv/zz_reliquary.pth" in names
    # No last line is glued to a rule: every rule line stands alone.
    if venv_ignore is not None:
        lines = (repo / ".venv" / ".gitignore").read_text().splitlines()
        assert venv_ignore.rstrip("\n") in lines and "!zz_reliquary.pth" in lines


def test_the_planted_patch_cases_break_or_arm_the_agents_git():
    cases = {c["name"]: c for c in conformance.golden_cases(0, b"G", "/testbed", python=False)}
    assert "touch /testbed/.git/index.lock" in commands(cases["patch_planted_in_artifacts_0"])
    armed = commands(cases["patch_planted_by_git_config_0"])[-1]
    for setting in ("filter.zz.clean", "filter.zz.required true", "core.fsmonitor",
                    "core.hooksPath", "diff.external", "* filter=zz"):
        assert setting in armed
    assert "/logs/artifacts/patch.diff" in conformance.PLANTER
    assert "exit 1" in conformance.PLANTER  # git fails: a capture that ran it writes nothing
    assert "mv /testbed/.git " in commands(cases["patch_planted_without_a_repository_0"])[-1]


def test_rewritten_tests_override_each_fail_to_pass_test_in_its_own_file():
    f2p = ("t/test_a.py::C::test_x", "t/test_a.py::test_f", "t/test_b.py::test_g[1]",
           "t/test_b.py::D::E::test_n", "t/test_c.py::D::test_y")
    assert conformance._passing_overrides(f2p) == {
        "t/test_a.py": "\nC.test_x = lambda self: None\n\n\ndef test_f():\n    pass\n",
        "t/test_c.py": "\nD.test_y = lambda self: None\n",
    }
    cases = conformance.golden_cases(1, b"G", "/testbed", python=True, fail_to_pass=f2p)
    case = next(c for c in cases if c["name"] == "tests_rewritten_1")
    assert case["expect"] == 0.0
    appends = [c for c in commands(case) if c.startswith("cat ")]
    assert appends == ["cat /tmp/reliquary-override.py >> /testbed/t/test_a.py",
                       "cat /tmp/reliquary-override.py >> /testbed/t/test_c.py"]
    assert not any(c["name"].startswith("tests_rewritten")
                   for c in conformance.golden_cases(1, b"G", "/testbed", python=True))


def test_train_goldens_get_the_rewritten_tests_case(monkeypatch):
    golden = dataclasses.replace(row(), instance_id="x", gold_patch="diff --git a/x b/x\n",
                                 fail_to_pass=("t/test_a.py::C::test_x",))
    monkeypatch.setattr(conformance, "GOLDENS", {"train": ("x",), "r2e": ("x",)})
    monkeypatch.setattr(conformance, "_golden_index", lambda split, key: 0)
    monkeypatch.setattr(conformance, "_row", lambda split, index: golden)
    assert "tests_rewritten_0" in {c["name"] for c in conformance.conformance_cases("train")}
    assert "tests_rewritten_0" not in {c["name"] for c in conformance.conformance_cases("r2e")}


def test_the_conformance_indices_are_those_of_the_default_split_config():
    config = vf.taskset_config_type("reliquary-swe")(id="reliquary-swe", split="train")
    assert config.num_images == corpus.DEFAULT_SWESMITH_IMAGES
    assert config.max_test_count is None and config.num_tasks is None


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
