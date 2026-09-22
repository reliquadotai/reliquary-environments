from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT

from reliquary_swe import corpus, swe_adapter

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


def test_image_reference_is_stable():
    row = _row(INSTANCE)
    assert swe_adapter.image_for(row) == swe_adapter.image_for(row)
    assert swe_adapter.image_for(row).strip()


def test_unparseable_output_yields_no_results_rather_than_raising():
    assert swe_adapter.parse_results(_row(INSTANCE), "") == {}


def test_the_test_command_names_the_tests_it_was_given():
    row = _row(INSTANCE)
    # A real fail_to_pass entry from the corpus row itself -- no container run
    # needed to know this is a genuine test name for this instance.
    known_test = row.fail_to_pass[0]
    argv = swe_adapter.test_command(row, (known_test,))
    assert any(known_test.split("::")[0] in part for part in argv)


def test_command_matches_the_specs_test_cmd_for_this_repo_and_version():
    # Checked by hand against swebench==3.0.17:
    # MAP_REPO_VERSION_TO_SPECS["astropy/astropy"]["4.3"]["test_cmd"] == "pytest -rA".
    # Pinning both the row's version and the resulting argv catches either the
    # corpus or the spec lookup drifting silently.
    row = _row(INSTANCE)
    assert row.version == "4.3"
    assert swe_adapter.test_command(row, ()) == ["pytest", "-rA"]


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
