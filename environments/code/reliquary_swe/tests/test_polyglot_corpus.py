"""The polyglot corpus (MiMo-V2.6-RL-oss's `code` config), without a container."""

from __future__ import annotations

import verifiers.v1 as vf

from reliquary_swe import corpus, grading


def test_the_polyglot_corpus_loads_every_published_row():
    rows = corpus.load_polyglot_rows()
    assert len(rows) == 2698
    assert len({row.instance_id for row in rows}) == len(rows)


def test_a_polyglot_row_carries_what_grading_needs_and_no_reference_fix():
    row = corpus.load_polyglot_rows()[0]
    assert row.instance_id.startswith("format-code-task-")
    # The image is given by the corpus, published under one Docker Hub
    # repository with the instance id as its tag.
    assert row.image == f"{corpus.POLYGLOT_IMAGE_REPOSITORY}:{row.instance_id}"
    assert row.workdir in ("/testbed", "/workspace/repo")
    assert "mimo_test_command.sh" in row.test_command
    assert "mimo_test_command.sh" in row.test_patch
    # Upstream publishes hidden tests, not a fix, and not test lists.
    assert row.gold_patch == ""
    assert row.fail_to_pass == () and row.pass_to_pass == ()
    assert row.base_commit == "HEAD"


def test_both_workdirs_are_represented():
    workdirs = {row.workdir for row in corpus.load_polyglot_rows()}
    assert workdirs == {"/testbed", "/workspace/repo"}


def test_the_polyglot_split_yields_tasks_in_their_own_workdir():
    config = vf.taskset_config_type("reliquary-swe")
    tasks = list(vf.load_taskset(config(id="reliquary-swe", split="polyglot")).head(3))
    for task in tasks:
        row = next(r for r in corpus.load_polyglot_rows() if r.instance_id == task.data.instance_id)
        assert task.data.split == "polyglot"
        assert task.data.workdir == row.workdir
        assert task.data.image == row.image
        assert task.data.test_command == row.test_command
        assert f"at {row.workdir}." in str(task.data.prompt)
        assert task.data.network_allow == []


def test_restoration_names_every_ecosystem_runner_config_above_the_hidden_tests():
    paths = grading._polyglot_infrastructure_paths(
        "diff --git a/pkg/sub/x_test.go b/pkg/sub/x_test.go\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/pkg/sub/x_test.go\n"
        "@@ -0,0 +1 @@\n"
        "+package sub\n"
    )
    for expected in (
        "go.mod",
        "pkg/go.mod",
        "pkg/sub/go.mod",
        "package.json",
        "jest.config.js",
        "conftest.py",
        "pkg/sub/conftest.py",
        "pytest.ini",
        "Cargo.toml",
        "pom.xml",
    ):
        assert expected in paths, expected
    # Duplicates would restore the same path twice; harmless, but noise.
    assert len(paths) == len(set(paths))
