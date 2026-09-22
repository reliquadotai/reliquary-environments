import pytest
import verifiers.v1 as vf
from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT

from conftest import provisioned_runtime
from reliquary_swe import corpus, swe_adapter

docker = pytest.mark.docker

# astropy/astropy is in swebench's MAP_REPO_VERSION_TO_SPECS and
# MAP_REPO_TO_PARSER, so it exercises both image naming and log parsing.
INSTANCE = "astropy__astropy-12907"


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
    for task in vf.load_taskset(config(id="reliquary-swe")):
        if task.data.instance_id == instance_id:
            return task
    raise AssertionError(f"{instance_id} is not in the taskset")


def test_image_reference_is_stable():
    row = _row(INSTANCE)
    assert swe_adapter.image_for(row) == swe_adapter.image_for(row)
    assert swe_adapter.image_for(row).strip()


def test_unparseable_output_yields_no_results_rather_than_raising():
    assert swe_adapter.parse_results(_row(INSTANCE), "") == {}


def test_the_test_command_reads_tests_from_the_given_path():
    # Test names travel through a file, not the argv -- see
    # test_no_argv_element_approaches_the_kernel_argument_length_cap for why.
    # All this checks is that the command actually points at the path it
    # was given.
    row = _row(INSTANCE)
    argv = swe_adapter.test_command(row, "/tmp/my-tests.txt")
    assert any("/tmp/my-tests.txt" in part for part in argv)


def test_command_matches_the_specs_test_cmd_for_this_repo_and_version():
    # Checked by hand against swebench==3.0.17:
    # MAP_REPO_VERSION_TO_SPECS["astropy/astropy"]["4.3"]["test_cmd"] == "pytest -rA".
    # Pinning both the row's version and the resulting argv catches either the
    # corpus or the spec lookup drifting silently.
    row = _row(INSTANCE)
    assert row.version == "4.3"
    assert swe_adapter.test_command(row, "/tmp/tests.txt") == [
        "bash",
        "-c",
        "source /opt/miniconda3/bin/activate && conda activate testbed && "
        'test -s /tmp/tests.txt && exec xargs -r -d "\\n" -a /tmp/tests.txt -- pytest -rA',
    ]


def test_command_activates_testbed_for_a_repo_with_an_unrelated_test_runner():
    # `pytest` is astropy's own test_cmd; django uses its own tests/runtests.py.
    # Pinning both against the same "activate testbed" wrapping is what proves
    # the wrapping isn't an astropy-specific guess -- confirmed by hand on
    # swebench/sweb.eval.x86_64.django_1776_django-10097 (see
    # swe_adapter._ACTIVATE_TESTBED's comment for how).
    row = _row("django__django-10097")
    assert row.version == "2.2"
    assert swe_adapter.test_command(row, "/tmp/tests.txt") == [
        "bash",
        "-c",
        "source /opt/miniconda3/bin/activate && conda activate testbed && "
        'test -s /tmp/tests.txt && exec xargs -r -d "\\n" -a /tmp/tests.txt -- '
        "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
    ]


def test_no_argv_element_approaches_the_kernel_argument_length_cap():
    # matplotlib__matplotlib-25122 carries the corpus's largest combined
    # FAIL_TO_PASS + PASS_TO_PASS list (2,488 tests, ~265 KB joined) -- the
    # instance whose old, inlined argv raised `OSError: Argument list too
    # long` before ever reaching a container (see swe_adapter._MAX_ARG_STRLEN's
    # comment for the measurement). test_command's argv is now independent of
    # the test list's size by construction (tests travel through a file, not
    # the command line), so this also guards against a future edit
    # reintroducing that coupling, not just against this one instance's count.
    row = _row("matplotlib__matplotlib-25122")
    assert len(row.fail_to_pass) + len(row.pass_to_pass) > 2000
    argv = swe_adapter.test_command(row, "/logs/artifacts/tests.txt")
    assert all(len(part.encode()) < swe_adapter._MAX_ARG_STRLEN for part in argv)


def test_tests_file_contents_is_one_test_per_line():
    assert swe_adapter.tests_file_contents(("a", "b")) == b"a\nb\n"


def test_tests_file_contents_of_no_tests_is_empty():
    # Empty, not "\n" -- `test_command`'s `test -s` guard must see a
    # genuinely empty file, the same signal a missing file gives it.
    assert swe_adapter.tests_file_contents(()) == b""


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
    """The regression check for finding #2: run a real, large instance's
    full combined FAIL_TO_PASS + PASS_TO_PASS list -- comfortably over the
    128 KiB single-argv cap when inlined -- against its real image, through
    the file-based test_command, and check parse_results comes back with
    real, non-empty statuses. The old, inlined-argv command didn't fail
    this way; it never got the chance to: exec of `docker` itself raised
    `OSError: Argument list too long` before any container was reached (see
    swe_adapter._MAX_ARG_STRLEN's comment for the measurement), which is a
    worse failure than the "ran fine, parsed nothing" this test's name
    describes, not a better one.

    pydata__xarray-6744 rather than django__django-10097 or
    matplotlib__matplotlib-25122:
      - django's FAIL_TO_PASS/PASS_TO_PASS entries are unittest `str()`
        reprs ("test_method (module.Class)"), which its own runner does
        not accept as a command-line label at all -- `DiscoverRunner.
        build_suite` passes the label straight to `TestLoader.
        loadTestsFromName`, which wants a dotted `module.Class.method`
        path. A separate, pre-existing defect in what "tests" means for
        that repo, unrelated to argv size; not fixed here.
      - matplotlib-25122's image cannot even be pulled on this box: its
        layers contain a UID (197609) outside the docker daemon's
        configured subordinate UID/GID range, a host config limit, not a
        reliquary_swe bug.
      - A number of instances across the corpus -- including
        django__django-10097 and pydata__xarray-4687 among these five --
        also carry FAIL_TO_PASS/PASS_TO_PASS entries that are themselves
        malformed (parametrized pytest ids truncated at an internal comma,
        e.g. ending "...test_isel[float64-single" with no closing `]`) or
        that no longer resolve to a real, collectible test at this
        instance's base commit. Both are corpus/upstream-dataset quality
        issues, orthogonal to this fix; xarray-6744 happens to carry
        neither for its FAIL_TO_PASS/PASS_TO_PASS set, which is why it was
        chosen, not because every instance is this clean.
    """
    row = _row("pydata__xarray-6744")
    tests = tuple(row.fail_to_pass) + tuple(row.pass_to_pass)
    assert len(tests) > 1500  # comfortably past the old 128 KiB cap when inlined

    tests_path = "/tmp/reliquary-swe-tests.txt"
    async with provisioned_runtime(_task(row.instance_id)) as runtime:
        checkout = await runtime.run(["git", "checkout", "-q", row.base_commit], {})
        assert checkout.exit_code == 0
        await runtime.write(tests_path, swe_adapter.tests_file_contents(tests))
        result = await runtime.run(swe_adapter.test_command(row, tests_path), {})

    parsed = swe_adapter.parse_results(row, _sentinel_wrapped(result.stdout))
    assert len(parsed) > 400
    assert "PASSED" in parsed.values()
