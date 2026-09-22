from reliquary_swe import corpus


def test_verified_split_has_the_published_instance_count():
    assert len(corpus.load_rows("eval")) == 500


def test_rows_carry_everything_grading_needs():
    row = corpus.load_rows("eval")[0]
    assert row.instance_id
    assert row.repo
    assert len(row.base_commit) == 40
    # MAP_REPO_VERSION_TO_SPECS is keyed by (repo, version); without this,
    # the swebench adapter's spec lookup has to guess.
    assert row.version.strip()
    assert row.problem_statement.strip()
    # An instance with no fail-to-pass test cannot express failure, so it
    # cannot express success either.
    assert row.fail_to_pass
    assert row.gold_patch.strip()
    # The test patch is what the grading box applies to restore the tests the
    # agent may have edited. Without it there is nothing to grade against.
    assert row.test_patch.strip()


def test_instance_ids_are_unique_and_usable_as_task_keys():
    ids = [row.instance_id for row in corpus.load_rows("eval")]
    assert len(ids) == len(set(ids))
