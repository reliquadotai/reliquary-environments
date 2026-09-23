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
    data = SimpleNamespace(test_patch="diff --git a/x b/x\n")
    assert grading._restore_strategy_for(data) is grading._restore_from_test_patch


async def test_restore_strategy_is_the_pristine_image_strategy_with_no_test_patch():
    # The seam a SWE-smith-shaped row (no test_patch at all) lands in without
    # `grade()` changing: it restores fail-to-pass/pass-to-pass test files
    # from the pristine image instead of reapplying a test_patch (spec
    # section 8's correction). A repo with no fail-to-pass/pass-to-pass
    # entries has no files to restore, so this exercises the real strategy
    # (not a stub) with no runtime needed at all: `test_files` returns `[]`
    # and the checkout loop never runs.
    data = SimpleNamespace(
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
