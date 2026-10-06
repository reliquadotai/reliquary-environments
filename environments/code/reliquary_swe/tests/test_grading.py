"""Pure-function unit tests for grading.py's helpers -- no container needed.

The docker-marked goldens in test_goldens.py are what actually prove reward
is computed correctly against a real box; these pin the small pieces below
that support it (path parsing, the conftest.py ancestor walk, and the
restoration-strategy seam) fast enough to run on every edit.
"""

from types import SimpleNamespace

from reliquary_swe import grading


def test_paths_touched_by_a_modified_file():
    patch = (
        "diff --git a/foo/bar.py b/foo/bar.py\n"
        "--- a/foo/bar.py\n"
        "+++ b/foo/bar.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )
    assert grading._paths_touched_by(patch) == ["foo/bar.py"]


def test_paths_touched_by_several_hunks():
    patch = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    assert grading._paths_touched_by(patch) == ["a.py", "b.py"]


def test_paths_touched_by_empty_patch():
    assert grading._paths_touched_by("") == []


def test_conftest_ancestors_walks_every_directory_to_the_root():
    # pytest would load a conftest.py from any of these on the way to the
    # given file -- not only the file's own directory -- and its own
    # locate_config searches the identical walk for its five config
    # filenames (IMPORTANT 2): a nested one would otherwise win over a
    # restored root copy, since locate_config searches innermost-first.
    ancestors = grading._conftest_ancestors(
        ["astropy/modeling/tests/test_separable.py"]
    )
    names = ("conftest.py", *grading._TEST_CONFIG_FILES)
    assert ancestors == [
        *(f"astropy/modeling/tests/{name}" for name in names),
        *(f"astropy/modeling/{name}" for name in names),
        *(f"astropy/{name}" for name in names),
        *names,
    ]


def test_conftest_ancestors_deduplicates_across_paths():
    ancestors = grading._conftest_ancestors(["a/tests/test_x.py", "a/tests/test_y.py"])
    names = ("conftest.py", *grading._TEST_CONFIG_FILES)
    assert ancestors == [
        *(f"a/tests/{name}" for name in names),
        *(f"a/{name}" for name in names),
        *names,
    ]


def test_conftest_ancestors_of_a_root_level_file():
    names = ("conftest.py", *grading._TEST_CONFIG_FILES)
    assert grading._conftest_ancestors(["test_x.py"]) == list(names)


def test_conftest_ancestors_of_no_paths_is_empty():
    assert grading._conftest_ancestors([]) == []


def test_restore_strategy_is_the_test_patch_strategy_when_one_exists():
    data = SimpleNamespace(split="eval", test_patch="diff --git a/x b/x\n")
    assert grading._restore_strategy_for(data) is grading._restore_from_test_patch


def test_restore_strategy_is_the_polyglot_strategy_for_a_polyglot_row():
    # A polyglot row HAS a test_patch, so without its own branch it would
    # land in `_restore_from_test_patch`, whose infrastructure walk asks
    # SWE-bench's per-(repo, version) registry about a repo it has never
    # heard of.
    data = SimpleNamespace(split="polyglot", test_patch="diff --git a/x b/x\n")
    assert grading._restore_strategy_for(data) is grading._restore_polyglot


async def test_restore_strategy_is_the_pristine_image_strategy_with_no_test_patch():
    # The seam a SWE-smith-shaped row (no test_patch at all) lands in without
    # `grade()` changing: it restores fail-to-pass/pass-to-pass test files
    # from the pristine image instead of reapplying a test_patch (spec
    # section 8's correction). A repo with no fail-to-pass/pass-to-pass
    # entries has no files to restore, so this exercises the real strategy
    # (not a stub) with no runtime needed at all: `test_files` returns `[]`
    # and the checkout loop never runs.
    data = SimpleNamespace(
        split="train",
        instance_id="fake",
        repo="swesmith/oauthlib__oauthlib.1fd52536",
        base_commit="origin/fake~1",
        version="",
        fail_to_pass=(),
        pass_to_pass=(),
        gold_patch="",
        test_patch="",
    )
    strategy = grading._restore_strategy_for(data)
    assert strategy is grading._restore_from_pristine_image
    assert strategy is not grading._restore_from_test_patch
    assert await strategy(None, data) is True


def _fake_swe_data(repo: str, version: str, test_patch: str) -> SimpleNamespace:
    """A duck-typed stand-in for `SweData` carrying only what
    `_test_infrastructure_paths`/`_row_for` read: real `repo`/`version`
    pairs are needed so `swe_adapter.test_entrypoint` hits a real
    `MAP_REPO_VERSION_TO_SPECS` entry rather than raising.
    """
    return SimpleNamespace(
        instance_id="fake",
        repo=repo,
        base_commit="0" * 40,
        version=version,
        fail_to_pass=(),
        pass_to_pass=(),
        gold_patch="",
        test_patch=test_patch,
    )


def test_infrastructure_paths_include_djangos_entrypoint():
    # django/django's own test runner is not named by test_patch, conftest,
    # or any test-name list -- see swe_adapter.test_entrypoint and CRITICAL 2
    # in task-4-report.md.
    data = _fake_swe_data("django/django", "1.7", test_patch="")
    paths = grading._test_infrastructure_paths(data)
    assert "tests/runtests.py" in paths
    assert "tox.ini" in paths and "pytest.ini" in paths


def test_infrastructure_paths_include_sympys_entrypoint():
    data = _fake_swe_data("sympy/sympy", "1.0", test_patch="")
    assert "bin/test" in grading._test_infrastructure_paths(data)


def test_infrastructure_paths_omit_an_entrypoint_for_bare_pytest_repos():
    # astropy's test_cmd is bare `pytest`, resolved off PATH -- not a
    # repo-relative file, so nothing should be added on its account.
    data = _fake_swe_data("astropy/astropy", "3.0", test_patch="")
    paths = grading._test_infrastructure_paths(data)
    assert not any(path.endswith("pytest") for path in paths)


def test_infrastructure_paths_include_conftest_ancestors_of_test_patch():
    patch = (
        "diff --git a/pkg/tests/test_x.py b/pkg/tests/test_x.py\n"
        "--- a/pkg/tests/test_x.py\n+++ b/pkg/tests/test_x.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    data = _fake_swe_data("astropy/astropy", "3.0", test_patch=patch)
    paths = grading._test_infrastructure_paths(data)
    assert "pkg/tests/conftest.py" in paths
    assert "pkg/conftest.py" in paths
    assert "conftest.py" in paths


def test_infrastructure_paths_include_nested_pytest_config_not_only_root():
    # IMPORTANT 2. pytest's own locate_config searches innermost-first, so a
    # pytest.ini added at pkg/tests/ (nowhere near the repo root) would win
    # over a root-only restored copy and stay unrestored under the old,
    # root-only _TEST_CONFIG_FILES scheme -- see _conftest_ancestors'
    # docstring for the measured cost of closing this at every level.
    patch = (
        "diff --git a/pkg/tests/test_x.py b/pkg/tests/test_x.py\n"
        "--- a/pkg/tests/test_x.py\n+++ b/pkg/tests/test_x.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    data = _fake_swe_data("astropy/astropy", "3.0", test_patch=patch)
    paths = grading._test_infrastructure_paths(data)
    for name in grading._TEST_CONFIG_FILES:
        assert f"pkg/tests/{name}" in paths
        assert f"pkg/{name}" in paths
        assert name in paths  # still restored unconditionally at root too


def test_infrastructure_paths_include_djangos_settings_module():
    # IMPORTANT 1 (fix round 2): --settings=test_sqlite is DJANGO_SETTINGS_MODULE,
    # imported inside the grading process before a single test runs, and is
    # named by no test_patch, no conftest.py, and nothing test_entrypoint
    # returns on its own -- see swe_adapter.test_command_argument_paths.
    data = _fake_swe_data("django/django", "1.11", test_patch="")
    assert "tests/test_sqlite.py" in grading._test_infrastructure_paths(data)


class _FakeResult:
    """Stands in for `verifiers.v1.runtimes.base.ProgramResult`."""

    def __init__(self, exit_code: int = 0):
        self.exit_code = exit_code
        self.stdout = ""
        self.stderr = ""


class _FakeRuntime:
    """Records every `run`/`write` call `_restore_from_test_patch` makes,
    without a real box. `fail_on` names argv tuples that report a non-zero
    exit; everything else succeeds -- so a test can force exactly one step
    to fail and check that failure propagates to `ok`.
    """

    def __init__(self, fail_on: frozenset = frozenset()):
        self.calls: list[tuple] = []
        self._fail_on = fail_on

    async def run(self, argv, env):
        key = tuple(argv)
        self.calls.append(key)
        return _FakeResult(1 if key in self._fail_on else 0)

    async def write(self, path, data):
        self.calls.append(("write", path))


def _mixed_test_patch_data() -> SimpleNamespace:
    """CRITICAL 1's own shape (fix round 1): a `test_patch` that both
    modifies an existing file and adds a new one -- the exact 13-instance
    case CRITICAL 1 is about (see grading._checkout's docstring), which the
    sphinx golden in test_goldens.py (add-only) does not exercise. Pinned
    here with a fake runtime rather than a real container: the shape is
    about which git commands get issued, not about what a real repository
    contains.
    """
    test_patch = (
        "diff --git a/existing_test.py b/existing_test.py\n"
        "--- a/existing_test.py\n+++ b/existing_test.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/added_test.py b/added_test.py\n"
        "new file mode 100644\n--- /dev/null\n+++ b/added_test.py\n@@ -0,0 +1,1 @@\n+x\n"
    )
    return SimpleNamespace(
        instance_id="fake",
        repo="astropy/astropy",
        base_commit="0" * 40,
        version="3.0",
        fail_to_pass=(),
        pass_to_pass=(),
        gold_patch="",
        test_patch=test_patch,
    )


async def test_restore_from_test_patch_checks_out_existing_and_removes_added():
    data = _mixed_test_patch_data()
    runtime = _FakeRuntime()
    ok = await grading._restore_from_test_patch(runtime, data)
    assert ("git", "checkout", data.base_commit, "--", "existing_test.py") in runtime.calls
    assert ("rm", "-rf", "added_test.py") in runtime.calls
    assert ok is True


async def test_restore_from_test_patch_reports_a_failed_checkout():
    # Pins Report.restored's own signal (IMPORTANT 4): nothing previously
    # asserted `_restore_from_test_patch` can return False for a real
    # failure, only that the no-op strategy always returns True -- a
    # regression that hardcoded `ok = True` would have passed every existing
    # test.
    data = _mixed_test_patch_data()
    failing = ("git", "checkout", data.base_commit, "--", "existing_test.py")
    runtime = _FakeRuntime(fail_on=frozenset({failing}))
    ok = await grading._restore_from_test_patch(runtime, data)
    assert ok is False


# -- Every split refuses a patch outside the tracked tree ---------------------
#
# The agent's diff is agent-controlled (`diff.external` in .git/config, a moved
# base ref, a `!` ignore rule): a hunk can rewrite a file the image ships
# untracked or ignored -- `node_modules/.bin/jest` -> `exit 0` -- for any split.


class _AnsweringRuntime:
    """Answers a command from `answers` (first substring of the joined argv
    that matches), else exit 0 with a SHA-shaped stdout; records every one."""

    def __init__(self, answers):
        self.answers = answers
        self.commands: list[str] = []

    async def run(self, argv, env):
        command = " ".join(argv)
        self.commands.append(command)
        for needle, result in self.answers.items():
            if needle in command:
                return result
        return SimpleNamespace(exit_code=0, stdout="0" * 40 + "\n", stderr="")

    async def write(self, path, data):
        self.commands.append(f"write {path}")


def _data(split: str):
    from reliquary_swe.taskset import SweData

    return SweData(
        idx=0, name="x", prompt="p", instance_id="x", repo="", base_commit="HEAD", version="",
        fail_to_pass=(), pass_to_pass=(), gold_patch="", split=split,
        test_patch=(
            "diff --git a/mimo_test_command.sh b/mimo_test_command.sh\nnew file mode 100755\n"
            "--- /dev/null\n+++ b/mimo_test_command.sh\n@@ -0,0 +1 @@\n+npx jest\n"
        ) if split == "polyglot" else "",
        test_command="bash mimo_test_command.sh" if split == "polyglot" else "",
    )


def _ok(stdout=""):
    return SimpleNamespace(exit_code=0, stdout=stdout, stderr="")


def _runtime(patch_path: str, untracked: str = "", ignored: str = ""):
    return _AnsweringRuntime({
        "--numstat": _ok(f"1\t1\t{patch_path}\0"),
        "ls-files -z --others --directory": _ok(untracked),
        "check-ignore": SimpleNamespace(exit_code=0 if ignored else 1, stdout=ignored,
                                        stderr=""),
    })


def test_the_patch_path_refusal_is_split_agnostic():
    assert not hasattr(grading, "_r2e_patch_violations")
    assert not hasattr(grading, "_r2e_forbidden_paths")
    assert grading._forbidden_patch_paths(["a.py", ".venv/x"], [], []) == [".venv/x"]


async def test_a_polyglot_patch_to_an_ignored_runner_scores_zero_unapplied():
    runtime = _runtime("node_modules/.bin/jest", ignored="node_modules/.bin/jest\0")
    report = await grading.grade(runtime, _data("polyglot"),
                                 "diff --git a/node_modules/.bin/jest b/node_modules/.bin/jest\n")
    assert (report.reward, report.applied) == (0.0, False)
    assert "outside the tracked tree" in report.test_output_tail
    assert "git apply -v /tmp/agent.diff" not in runtime.commands


async def test_a_polyglot_patch_to_an_untracked_file_scores_zero_unapplied():
    runtime = _runtime("tools/runner.sh", untracked="tools/\0")
    report = await grading.grade(runtime, _data("polyglot"),
                                 "diff --git a/tools/runner.sh b/tools/runner.sh\n")
    assert (report.reward, report.applied) == (0.0, False)
    assert "git apply -v /tmp/agent.diff" not in runtime.commands


async def test_an_honest_polyglot_patch_still_passes():
    runtime = _runtime("src/index.js", untracked="node_modules/\0")
    report = await grading.grade(runtime, _data("polyglot"),
                                 "diff --git a/src/index.js b/src/index.js\n")
    assert (report.reward, report.applied) == (1.0, True)
    assert "git apply -v /tmp/agent.diff" in runtime.commands


async def test_a_train_patch_to_an_ignored_path_scores_zero_unapplied(monkeypatch):
    runtime = _runtime("build/lib/pkg/core.py", ignored="build/lib/pkg/core.py\0")
    report = await grading.grade(runtime, _data("train"),
                                 "diff --git a/build/lib/pkg/core.py b/build/lib/pkg/core.py\n")
    assert (report.reward, report.applied) == (0.0, False)
    assert "git apply -v /tmp/agent.diff" not in runtime.commands


async def test_an_honest_train_patch_still_passes(monkeypatch):
    # SWE-smith's per-repository command and parser stand in for a passing run.
    monkeypatch.setattr(grading.swesmith_adapter, "test_command", lambda row: ["run-tests"])
    monkeypatch.setattr(grading.swesmith_adapter, "parse_results",
                        lambda row, out: {"tests/test_a.py::test_a": "PASSED"})
    data = _data("train").model_copy(update={"repo": "swesmith/oauthlib__oauthlib.1fd52536",
                                             "fail_to_pass": ("tests/test_a.py::test_a",)})
    runtime = _runtime("pkg/core.py")
    report = await grading.grade(runtime, data, "diff --git a/pkg/core.py b/pkg/core.py\n")
    assert (report.reward, report.applied) == (1.0, True)
    assert "git apply -v /tmp/agent.diff" in runtime.commands
