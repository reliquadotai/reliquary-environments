"""The R2E-Gym-Subset corpus (`--taskset.split r2e`), without a container."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest
import verifiers.v1 as vf

from reliquary_swe import corpus

DIGEST = re.compile(r"\Anamanjain12/[a-z0-9]+_final@sha256:[0-9a-f]{64}\Z")

# The three goldens' fix commits (see tests/test_r2e_goldens.py).
COVERAGEPY = "c1bfa7352368b63f3a9b30c02f242408d07a7ab2"
TORNADO = "b5ec807edc83c8e7d1d12553d635ebe765e5c614"
PANDAS = "fadb72cf5ef8489e409d4d33625bd16a76fa7a42"


def test_the_r2e_corpus_loads_every_published_row():
    rows = corpus.load_r2e_rows()
    assert len(rows) == 4578
    assert len({row.instance_id for row in rows}) == len(rows)
    assert Counter(row.repo for row in rows) == {
        "pandas": 1444,
        "numpy": 781,
        "pillow": 620,
        "orange3": 482,
        "aiohttp": 299,
        "tornado": 261,
        "scrapy": 215,
        "pyramid": 189,
        "datalad": 179,
        "coveragepy": 108,
    }


def test_an_r2e_row_carries_what_grading_needs():
    row = corpus.r2e_row(COVERAGEPY)
    assert row.instance_id == f"r2e__coveragepy__{COVERAGEPY[:12]}"
    assert row.repo == "coveragepy"
    assert DIGEST.match(row.image), row.image
    assert row.image.startswith("namanjain12/coveragepy_final@")
    assert row.workdir == "/testbed"
    assert row.base_commit == "HEAD"
    assert row.fail_to_pass == () and row.pass_to_pass == ()
    assert row.test_patch == "" and row.test_command == ""
    expected = json.loads(row.expected_output_json)
    assert len(expected) == 7
    assert set(expected.values()) <= {"PASSED", "FAILED", "ERROR"}
    # R2E's own prompt keeps only what sits inside [ISSUE]...[/ISSUE].
    assert "[ISSUE]" not in row.problem_statement
    assert "[/ISSUE]" not in row.problem_statement
    assert row.problem_statement.strip()
    # The gold patch fixes the source and leaves the repo's own tests alone.
    assert "+++ b/coverage/debug.py" in row.gold_patch
    assert "tests/test_debug.py" not in row.gold_patch


def test_every_r2e_image_is_pinned_by_digest():
    for row in corpus.load_r2e_rows():
        assert DIGEST.match(row.image), row.instance_id


def test_an_unpinned_image_is_a_clear_error():
    with pytest.raises(KeyError, match="pin_r2e_digests"):
        corpus._pinned_r2e_image("namanjain12/nothing_final:0")


def test_problem_statement_keeps_the_inside_of_the_issue_tags():
    assert corpus._r2e_problem_statement("[ISSUE]\nbody\n[/ISSUE]") == "\nbody\n"
    # Greedy and multi-line, as R2E's own `get_task_instruction` regex is.
    assert corpus._r2e_problem_statement("x[ISSUE]a[/ISSUE] b [ISSUE]c[/ISSUE]y") == (
        "a[/ISSUE] b [ISSUE]c"
    )
    # No tag pair (100 rows of the pinned revision): the whole text, unchanged.
    assert corpus._r2e_problem_statement("plain report") == "plain report"
    assert corpus._r2e_problem_statement("[ISSUE] unterminated") == "[ISSUE] unterminated"


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_debug.py",
        "pandas/tests/frame/test_block_internals.py",
        "tornado/test/websocket_test.py",
        "Orange/widgets/tests/test_context_handler.py",
        "test_root_level.py",
        "pkg/thing_test.py",
        "tests/sample_data/page.html",
        "r2e_tests/test_1.py",
    ],
)
def test_test_files_are_recognised(path):
    assert corpus.is_r2e_test_file(path)


@pytest.mark.parametrize(
    "path",
    [
        "coverage/debug.py",
        "pandas/core/generic.py",
        "doc/source/whatsnew/v1.1.2.rst",
        # Library code that merely has "test" in its name is source.
        "numpy/testing/utils.py",
        "pandas/_testing.py",
        "pandas/util/testing/__init__.py",
        "contest.py",
        "testfixtures.py",
        "latest_test.txt",
    ],
)
def test_source_files_are_not_test_files(path):
    assert not corpus.is_r2e_test_file(path)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


def _assert_gold_patch_reproduces_the_fix(row_index: int, tmp_path: Path) -> None:
    """Lay the row's non-test files out at their pre-fix content in a fresh git
    repository, apply the gold patch with `git apply`, and compare every file
    to its post-fix content -- symlinks and modes included."""
    raw = corpus._r2e_dataset()[row_index]
    diffs = json.loads(raw["parsed_commit_content"])["file_diffs"]
    _git(tmp_path, "init", "-q")
    for fd in diffs:
        path = fd["header"]["file"]["path"]
        misc = fd["header"]["misc_line"] or ""
        if corpus.is_r2e_test_file(path) or misc.startswith("new file mode"):
            continue
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fd["old_file_content"])
    _git(tmp_path, "add", "-A")
    patch = corpus.r2e_gold_patch(diffs)
    (tmp_path / ".gold.diff").write_text(patch)
    applied = _git(tmp_path, "apply", "-v", ".gold.diff")
    assert applied.returncode == 0, applied.stderr
    for fd in diffs:
        path = fd["header"]["file"]["path"]
        misc = fd["header"]["misc_line"] or ""
        target = tmp_path / path
        if corpus.is_r2e_test_file(path):
            continue
        if misc.startswith("deleted file mode"):
            assert not target.exists(), path
        elif misc == "new file mode 120000":
            assert target.is_symlink() and os.readlink(target) == fd["new_file_content"]
        else:
            assert target.read_text() == fd["new_file_content"], path
            if misc.endswith("new mode 100755"):
                assert os.access(target, os.X_OK), path


def _index_of(commit_hash: str) -> int:
    return corpus._r2e_dataset()["commit_hash"].index(commit_hash)


@pytest.mark.parametrize("commit_hash", [COVERAGEPY, TORNADO, PANDAS])
def test_the_goldens_gold_patches_reproduce_the_fix(commit_hash, tmp_path):
    _assert_gold_patch_reproduces_the_fix(_index_of(commit_hash), tmp_path)


@pytest.mark.parametrize(
    "row_index",
    [
        1540,  # adds an empty file (no hunk at all)
        1757,  # adds a symlink (mode 120000, target without a trailing newline)
        2262,  # changes a file's mode 100644 -> 100755 as well as its content
        4550,  # deletes a file
        # A post-fix file with no trailing newline, the first in dataset order.
        next(
            i
            for i, raw in enumerate(corpus._r2e_dataset())
            if any(
                fd["new_file_content"]
                and not fd["new_file_content"].endswith("\n")
                and not corpus.is_r2e_test_file(fd["header"]["file"]["path"])
                and fd["header"]["misc_line"] is None
                for fd in json.loads(raw["parsed_commit_content"])["file_diffs"]
            )
        ),
    ],
)
def test_gold_patches_cover_every_file_shape_the_corpus_has(row_index, tmp_path):
    _assert_gold_patch_reproduces_the_fix(row_index, tmp_path)


def test_num_tasks_is_a_prefix_of_a_mixed_order():
    full = corpus.load_r2e_rows()
    assert corpus.load_r2e_rows(200) == full[:200]
    assert corpus.load_r2e_rows(10_000) == full
    # The dataset's own order is grouped by repository (its first 482 rows
    # are all orange3); the prefix must not be.
    assert len({row.repo for row in full[:100]}) >= 8


def test_num_tasks_rejects_zero():
    with pytest.raises(ValueError):
        corpus.load_r2e_rows(0)


def test_no_repo_overlap_with_the_evaluation_set():
    """R2E-Gym-Subset's 10 repositories share none with SWE-bench Verified's
    12, the way the SWE-smith split is checked
    (`test_swesmith_adapter.test_no_repo_overlap_between_the_two_corpora`).
    R2E names a repository by its bare name, Verified by `owner/name`, so
    the comparison is on the name part, lower-cased."""
    eval_repos = {row.repo for row in corpus.load_rows("eval")}
    assert len(eval_repos) == 12
    eval_names = {repo.split("/", 1)[1].lower() for repo in eval_repos}
    r2e_names = {row.repo.lower() for row in corpus.load_r2e_rows()}
    assert len(r2e_names) == 10
    assert r2e_names.isdisjoint(eval_names)


def test_the_r2e_split_yields_tasks_that_carry_the_expected_verdicts():
    config = vf.taskset_config_type("reliquary-swe")
    tasks = list(vf.load_taskset(config(id="reliquary-swe", split="r2e", num_tasks=3)))
    assert len(tasks) == 3
    rows = corpus.load_r2e_rows(3)
    for task, row in zip(tasks, rows, strict=True):
        assert task.data.split == "r2e"
        assert task.data.instance_id == row.instance_id
        assert task.data.image == row.image
        assert task.data.workdir == "/testbed"
        assert task.data.expected_output_json == row.expected_output_json
        assert task.data.network_allow == []
        assert "at /testbed." in str(task.data.prompt)
