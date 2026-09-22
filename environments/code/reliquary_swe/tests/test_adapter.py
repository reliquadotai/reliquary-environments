from reliquary_swe import corpus, swe_adapter

# astropy/astropy is in swebench's MAP_REPO_VERSION_TO_SPECS and
# MAP_REPO_TO_PARSER, so it exercises both image naming and log parsing.
INSTANCE = "astropy__astropy-12907"


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
