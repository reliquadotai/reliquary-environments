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
    # given file -- not only the file's own directory.
    ancestors = grading._conftest_ancestors(["astropy/modeling/tests/test_separable.py"])
    assert ancestors == [
        "astropy/modeling/tests/conftest.py",
        "astropy/modeling/conftest.py",
        "astropy/conftest.py",
        "conftest.py",
    ]


def test_conftest_ancestors_deduplicates_across_paths():
    ancestors = grading._conftest_ancestors(["a/tests/test_x.py", "a/tests/test_y.py"])
    assert ancestors == ["a/tests/conftest.py", "a/conftest.py", "conftest.py"]


def test_conftest_ancestors_of_a_root_level_file():
    assert grading._conftest_ancestors(["test_x.py"]) == ["conftest.py"]


def test_conftest_ancestors_of_no_paths_is_empty():
    assert grading._conftest_ancestors([]) == []


def test_restore_strategy_is_the_test_patch_strategy_when_one_exists():
    data = SimpleNamespace(test_patch="diff --git a/x b/x\n")
    assert grading._restore_strategy_for(data) is grading._restore_from_test_patch


async def test_restore_strategy_is_a_no_op_with_no_test_patch():
    # The seam a SWE-smith-shaped row (no test_patch at all) lands in without
    # `grade()` changing: nothing needs restoring when the corpus never
    # modifies test files to begin with. True, not merely non-raising: a
    # no-op restoration is trivially "successful" for Report.restored's
    # purposes -- see _restore_strategy_for's own docstring for why that
    # does not mean such a row would actually work today.
    data = SimpleNamespace(test_patch="")
    strategy = grading._restore_strategy_for(data)
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
