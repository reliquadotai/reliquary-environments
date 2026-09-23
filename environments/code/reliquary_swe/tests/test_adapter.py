import asyncio

import pytest
import verifiers.v1 as vf
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT

from conftest import provisioned_runtime
from reliquary_swe import corpus, swe_adapter

docker = pytest.mark.docker

# astropy/astropy is in swebench's MAP_REPO_VERSION_TO_SPECS and
# MAP_REPO_TO_PARSER, so it exercises both image naming and log parsing.
INSTANCE = "astropy__astropy-12907"

# Linux's per-argv-element cap (`MAX_ARG_STRLEN`, fs/exec.c), independent of
# the much larger `ARG_MAX`. An earlier version of `test_command` ran
# FAIL_TO_PASS/PASS_TO_PASS names directly and blew past this on the
# corpus's largest instances (matplotlib__matplotlib-25122's combined list
# alone joins to ~265 KB) -- measured directly: exec of `docker` itself
# raised `OSError: Argument list too long` before any container was
# reached. Running upstream's own `get_test_directives` instead means an
# argv is now a handful of file paths, never near this cap by construction
# -- but the bound is worth pinning rather than trusting by inspection; it's
# exactly what used to bite, silently.
_MAX_ARG_STRLEN = 128 * 1024


def _sentinel_wrapped(body: str) -> str:
    return f"{START_TEST_OUTPUT}\n{body}\n{END_TEST_OUTPUT}\n"


def _row(instance_id: str) -> corpus.SweRow:
    for row in corpus.load_rows("eval"):
        if row.instance_id == instance_id:
            return row
    raise AssertionError(f"{instance_id} is not in the corpus")


def _task(instance_id: str) -> vf.Task:
    # Goes through the real SweTaskset rather than hand-building a SweTask,
    # so this can't quietly drift from what production code actually yields.
    config = vf.taskset_config_type("reliquary-swe")
    for task in vf.load_taskset(config(id="reliquary-swe", split="eval")):
        if task.data.instance_id == instance_id:
            return task
    raise AssertionError(f"{instance_id} is not in the taskset")


def test_image_reference_is_stable():
    row = _row(INSTANCE)
    assert swe_adapter.image_for(row) == swe_adapter.image_for(row)
    assert swe_adapter.image_for(row).strip()


def test_unparseable_output_yields_no_results_rather_than_raising():
    assert swe_adapter.parse_results(_row(INSTANCE), "") == {}


def test_command_matches_the_specs_test_cmd_for_this_repo_and_version():
    # Checked by hand against swebench==3.0.17:
    # MAP_REPO_VERSION_TO_SPECS["astropy/astropy"]["4.3"]["test_cmd"] == "pytest -rA",
    # and get_test_directives(row) == ["astropy/modeling/tests/test_separable.py"]
    # (the sole file astropy__astropy-12907's test_patch touches). Pinning
    # the row's version and the resulting argv catches the corpus, the spec
    # lookup, or the directive derivation drifting silently.
    row = _row(INSTANCE)
    assert row.version == "4.3"
    assert swe_adapter.test_command(row) == [
        "bash",
        "-c",
        "( export LC_ALL=C.UTF-8 && source /opt/miniconda3/bin/activate && conda activate testbed && "
        "pytest -rA astropy/modeling/tests/test_separable.py ) 2>&1",
    ]


def test_command_activates_testbed_for_a_repo_with_an_unrelated_test_runner():
    # django uses its own tests/runtests.py, not pytest -- confirmed by hand
    # on swebench/sweb.eval.x86_64.django_1776_django-10097 (see
    # swe_adapter._ACTIVATE_TESTBED's comment for how).
    #
    # django__django-10097's own test_patch touches only two *.txt* fixture
    # files (tests/validators/{valid,invalid}_urls.txt), never a *.py* test
    # module, so `get_test_directives` correctly returns `[]` for it -- not
    # a defect in this instance or in the derivation. It means the fix this
    # instance grades is exercised by an *existing*, unmodified test file
    # that reads those fixtures at run time, and running with no targets at
    # all is exactly what upstream's own eval script does too.
    # `test_command_derives_django_directives_as_dotted_module_labels` below
    # is the test that pins the non-empty, transformed case.
    row = _row("django__django-10097")
    assert row.version == "2.2"
    assert swe_adapter.test_command(row) == [
        "bash",
        "-c",
        "( export LC_ALL=C.UTF-8 && source /opt/miniconda3/bin/activate && conda activate testbed && "
        "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 ) 2>&1",
    ]


def test_command_derives_django_directives_as_dotted_module_labels():
    """The case this fix's second round revealed: django/django's own test
    runner does not accept a raw file path, let alone a FAIL_TO_PASS/
    PASS_TO_PASS entry, as a command-line label -- only a dotted
    `module.Class.method`-shaped one. Upstream's `get_test_directives`
    handles this per repo (`swebench/harness/test_spec/python.py`): for
    django/django only, it strips a touched file's `tests/` prefix and
    `.py` suffix and turns `/` into `.`. django__django-11099's test_patch
    touches exactly `tests/auth_tests/test_validators.py`, so its directive
    must be the dotted label `auth_tests.test_validators` -- not that path,
    and not any of the instance's own FAIL_TO_PASS entries (which include
    names like
    `test_ascii_validator (auth_tests.test_validators.UsernameValidatorsTests)`,
    themselves unittest `str()` reprs the runner also would not accept).
    """
    row = _row("django__django-11099")
    assert "diff --git a/tests/auth_tests/test_validators.py" in row.test_patch
    assert swe_adapter.test_command(row) == [
        "bash",
        "-c",
        "( export LC_ALL=C.UTF-8 && source /opt/miniconda3/bin/activate && conda activate testbed && "
        "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 "
        "auth_tests.test_validators ) 2>&1",
    ]


def test_no_argv_element_approaches_the_kernel_argument_length_cap():
    # matplotlib__matplotlib-25122's combined FAIL_TO_PASS + PASS_TO_PASS
    # list (2,488 entries) is the one that raised `OSError: Argument list
    # too long` under the old, name-inlining test_command (see this
    # module's `_MAX_ARG_STRLEN` comment). Running directives instead of
    # names makes this trivially true here -- its test_patch touches
    # exactly one file -- but "trivially true" is exactly the property
    # worth pinning, not skipping: a future edit that went back to inlining
    # names would fail this silently otherwise, the same way the original
    # bug was silent.
    row = _row("matplotlib__matplotlib-25122")
    argv = swe_adapter.test_command(row)
    assert all(len(part.encode()) < _MAX_ARG_STRLEN for part in argv)


# --- test_entrypoint / get_modified_files ---
#
# `test_entrypoint` is what CRITICAL 2's fix (task-4-report.md) restores
# alongside conftest.py: the file test_cmd itself executes, when that file
# lives in the repo rather than resolving off PATH. Checked against every
# distinct first-token shape the current corpus actually uses (verified by
# scanning MAP_REPO_VERSION_TO_SPECS for every (repo, version) pair the
# corpus's 12 repos use): bare PATH commands (pytest, tox), a repo-relative
# script with a `./` prefix (django), and one preceded by a shell
# `NAME=value` assignment (sympy).


def test_entrypoint_is_none_for_a_bare_path_command():
    assert swe_adapter.test_entrypoint(_row(INSTANCE)) is None  # astropy: bare `pytest`


def test_entrypoint_is_none_for_sphinxs_tox_invocation():
    row = _row("sphinx-doc__sphinx-8595")
    assert row.repo == "sphinx-doc/sphinx"
    assert swe_adapter.test_entrypoint(row) is None  # bare `tox`, resolved off PATH


def test_entrypoint_strips_the_dot_slash_prefix_for_django():
    row = _row("django__django-11099")
    assert swe_adapter.test_entrypoint(row) == "tests/runtests.py"


def test_entrypoint_skips_a_leading_shell_assignment_for_sympy():
    row = _row("sympy__sympy-11618")
    assert row.repo == "sympy/sympy"
    # test_cmd is `PYTHONWARNINGS='...' bin/test -C --verbose`: the
    # assignment is not the command, and `bin/test` has no `./` to strip.
    assert swe_adapter.test_entrypoint(row) == "bin/test"


def test_get_modified_files_excludes_a_purely_added_path():
    patch = (
        "diff --git a/new.py b/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1,1 @@\n"
        "+x\n"
    )
    assert swe_adapter.get_modified_files(patch) == []


def test_get_modified_files_includes_a_modified_existing_path():
    patch = (
        "diff --git a/old.py b/old.py\n"
        "--- a/old.py\n+++ b/old.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    assert swe_adapter.get_modified_files(patch) == ["old.py"]


# --- parse_results dispatch (not correctness) ---
#
# These four tests pin the three branches inside parse_results' dispatch --
# parser found and invoked, no parser for the repo, parser raises -- plus the
# XFAIL normalization. They deliberately wrap arbitrary, hand-written lines in
# the real START_TEST_OUTPUT/END_TEST_OUTPUT sentinels so the sentinel-slicing
# branch is actually reached; none of them (except the XFAIL one) claims the
# parser's *output* is correct for real repository output, only that the
# dispatch does not raise and returns a dict. That is a different, narrower
# claim than the deferred fixture tests (see the module report), which would
# need a genuine recorded log to make.


def test_parser_is_invoked_and_returns_a_dict():
    """Pins dispatch reaching `parser(content, None)`; makes no claim about
    whether the parsed statuses are correct for real astropy output."""
    row = _row(INSTANCE)
    log = _sentinel_wrapped("PASSED tests/test_a.py::test_x")
    result = swe_adapter.parse_results(row, log)
    assert isinstance(result, dict)


def test_repo_with_no_registered_parser_yields_no_results():
    """Pins the `parser is None` branch; not a claim about parse correctness."""
    row = corpus.SweRow(
        instance_id="nope__nope-1",
        repo="nope/not-a-real-repo",
        base_commit="0" * 40,
        version="1.0",
        problem_statement="n/a",
        fail_to_pass=(),
        pass_to_pass=(),
        gold_patch="",
        test_patch="",
    )
    log = _sentinel_wrapped("PASSED tests/test_a.py::test_x")
    assert swe_adapter.parse_results(row, log) == {}


def test_a_raising_parser_yields_no_results_rather_than_the_exception(monkeypatch):
    """Pins the `except Exception` branch; not a claim about parse correctness."""
    row = _row(INSTANCE)

    def _raises(content, test_spec):
        raise ValueError("simulated parser failure")

    monkeypatch.setitem(swe_adapter.MAP_REPO_TO_PARSER, row.repo, _raises)
    log = _sentinel_wrapped("PASSED tests/test_a.py::test_x")
    assert swe_adapter.parse_results(row, log) == {}


def test_xfail_is_normalized_to_passed():
    """This one does claim correctness -- of our normalization, not of
    swebench's parsing: swebench's own `test_passed` (grading.py:26-27)
    treats XFAIL as a pass, so parse_results must not hand grading a raw
    "XFAIL" that would fail an exact `== "PASSED"` check."""
    row = _row(INSTANCE)
    log = _sentinel_wrapped("XFAIL tests/test_a.py::test_expected_to_fail")
    result = swe_adapter.parse_results(row, log)
    assert result["tests/test_a.py::test_expected_to_fail"] == "PASSED"


@docker
@pytest.mark.slow
async def test_grading_a_large_instance_end_to_end_gets_a_real_result():
    """Re-verifies, under the directives scheme that replaced the file/
    xargs indirection, the same real grading round-trip finding #2
    originally asked for: run a real, large instance's test_command against
    its real image and confirm parse_results comes back with real,
    non-empty, mostly-resolving statuses -- not the "ran fine, parsed
    nothing" signature an oversized inlined argv used to produce (or, as
    measured directly, the `OSError` it never even got the chance to avoid
    raising).

    Checked against PASS_TO_PASS, not FAIL_TO_PASS: FAIL_TO_PASS entries
    describe tests the instance's own test_patch introduces or changes, and
    this task never applies test_patch (that's grading's job, in a
    different container -- see the module's own docstring). PASS_TO_PASS
    entries are existing tests that must already be collectible and
    passing on the bare base commit with no patch involved at all, which is
    exactly the state this test leaves the repo in.
    """
    row = _row("pydata__xarray-6744")
    assert len(row.fail_to_pass) + len(row.pass_to_pass) > 1500

    async with provisioned_runtime(_task(row.instance_id)) as runtime:
        checkout = await runtime.run(["git", "checkout", "-q", row.base_commit], {})
        assert checkout.exit_code == 0
        result = await asyncio.wait_for(
            runtime.run(swe_adapter.test_command(row), {}), timeout=300
        )

    parsed = swe_adapter.parse_results(row, _sentinel_wrapped(result.stdout))
    assert len(parsed) > 1500
    resolved_passing = sum(1 for t in row.pass_to_pass if parsed.get(t) == "PASSED")
    assert resolved_passing > len(row.pass_to_pass) * 0.9


@docker
@pytest.mark.slow
async def test_grading_django_end_to_end_is_not_actually_broken():
    """django/django was believed unrunnable through `test_command` an hour
    before this test existed: its FAIL_TO_PASS/PASS_TO_PASS entries are
    unittest `str()` reprs its own runner will not accept as a command-line
    label. That belief was about the wrong invocation, not about django
    itself -- upstream never runs those entries as arguments either (see
    `test_command`'s docstring). This proves it by actually grading
    django__django-10097 end to end: its own test_patch touches only two
    `.txt` fixture files, so `get_test_directives` (correctly) returns no
    directives at all for it, and `test_command` runs with none -- which,
    for django's own runner, means its *entire* test suite. Measured by
    hand: ~226s of test execution (12,311 tests) by the runner's own
    report, and a separately hand-timed ~3m45s (225s) wall clock including
    migrations and teardown -- two independent measurements of about the
    same run; the 1s gap is rounding, not a sign the wall clock is smaller
    than the execution it contains. Marked slow because that is still
    much longer than the rest of this suite combined, not because
    anything is wrong -- every one of this instance's 1,432 PASS_TO_PASS
    entries and all 438 FAIL_TO_PASS entries resolved in that run (the
    latter without test_patch applied at all: the fixture files it would
    change are read by pre-existing, already-passing test functions, not
    ones the patch introduces -- so both lists happen to fully resolve
    here without this task ever touching test_patch).
    """
    row = _row("django__django-10097")
    assert row.fail_to_pass  # the instance still names a real regression

    async with provisioned_runtime(_task(row.instance_id)) as runtime:
        checkout = await runtime.run(["git", "checkout", "-q", row.base_commit], {})
        assert checkout.exit_code == 0
        result = await asyncio.wait_for(
            runtime.run(swe_adapter.test_command(row), {}), timeout=600
        )

    parsed = swe_adapter.parse_results(row, _sentinel_wrapped(result.stdout))
    assert len(parsed) > 10000
    resolved_passing = sum(1 for t in row.pass_to_pass if parsed.get(t) == "PASSED")
    assert resolved_passing > len(row.pass_to_pass) * 0.95


@docker
async def test_djangos_own_non_ascii_output_does_not_crash_the_test_command():
    """The locale defect: `_ACTIVATE_TESTBED` runs inside a non-login,
    non-interactive `bash -c`, which never sources the image's own
    `/etc/profile.d/01-locale-fix.sh`. Under the resulting plain POSIX/C
    locale, django's own `migrate` management command writes a literal
    ellipsis (`"  Creating tables…\n"`) to stdout and raises
    `UnicodeEncodeError` before a single test result is printed -- on a
    BARE checkout, no patch involved at all, because the crash comes from
    django's own test-database setup, not from anything under test. Without
    the fix, `test_command`'s captured output has no PASSED/FAILED line for
    `parse_results` to find at all: exit_code == 1 and `parse_results`
    returns `{}`, the same silent-zero shape as this module's other
    defects (see its own docstring) -- reward 0, indistinguishable from a
    genuine failure, for gold and empty patches alike. Reproduced by hand
    on both django__django-10880 and django__django-10914 before this test
    existed.
    """
    row = _row("django__django-10880")
    async with provisioned_runtime(_task(row.instance_id)) as runtime:
        checkout = await runtime.run(["git", "checkout", "-q", row.base_commit], {})
        assert checkout.exit_code == 0
        result = await runtime.run(swe_adapter.test_command(row), {})

    assert "UnicodeEncodeError" not in (result.stdout or "")
    parsed = swe_adapter.parse_results(row, _sentinel_wrapped(result.stdout))
    assert len(parsed) > 0
