"""Unit tests for the SWE-smith corpus and adapter -- no Docker needed.

`corpus.load_swesmith_rows`/`swesmith_image_rank` do make a real, network
dataset load (same convention as `test_corpus.py`'s SWE-bench Verified
tests), but nothing here starts a container.

This machine has previously OOM'd on multiple concurrent/sequential full
loads of a large HF dataset (see MEMORY.md's local-VPS-limits note); the
raw `datasets.load_dataset` call this file needs for its two independent
(not-through-`corpus.py`) checks is therefore made exactly once, in the
`_raw_counts` fixture below, and every test that would otherwise reload it
takes the fixture instead. `corpus.swesmith_image_rank`/`load_swesmith_rows`
are separately `lru_cache`d in production code, so calls through them are
already cheap on repeat.
"""

from __future__ import annotations

from collections import Counter

import pytest
from datasets import load_dataset

from reliquary_swe import corpus, swesmith_adapter
from reliquary_swe.corpus import _reverse_unified_diff


@pytest.fixture(scope="module")
def _raw_counts() -> Counter:
    """image_name -> row count, built directly from the pinned dataset --
    independent of `corpus.swesmith_image_rank`, so tests using this fixture
    are checking that function against ground truth, not against itself.
    """
    name, hf_split, revision = corpus._SOURCES["train"]
    dataset = load_dataset(name, split=hf_split, revision=revision)
    return Counter(dataset["image_name"])


# Measured directly against the pinned revision (see the implementation
# report): 222 distinct images, and these exact task counts for the top N by
# image. A drift here means either the pin moved or the ranking broke.
def test_image_rank_matches_the_measured_corpus(_raw_counts):
    ranked = corpus.swesmith_image_rank()
    assert len(ranked) == 222 == len(_raw_counts)
    # Non-increasing count, and ties broken by image name ascending -- checked
    # against the independently-built `counts` map, not against the function
    # under test's own output (a self-referential check here would pass on a
    # ranker that ignored counts entirely).
    ranked_counts = [_raw_counts[image] for image in ranked]
    assert ranked_counts == sorted(ranked_counts, reverse=True)
    for a, b in zip(ranked, ranked[1:]):
        if _raw_counts[a] == _raw_counts[b]:
            assert a < b


@pytest.mark.parametrize(
    ("num_images", "expected_before_exclusions"),
    [(5, 9639), (20, 23844), (30, 30560), (50, 39500)],
)
def test_top_n_task_counts_match_the_brief(_raw_counts, num_images, expected_before_exclusions):
    # `load_swesmith_rows` additionally excludes rows whose patch cannot be
    # reversed or whose test count is over the cap (see its own docstring)
    # -- pre-existing at every N, so this checks the *pre-exclusion* count
    # via the same Counter the brief's own numbers were computed from,
    # independent of that later, separate exclusion step.
    ranked = sorted(_raw_counts, key=lambda image: (-_raw_counts[image], image))
    assert sum(_raw_counts[image] for image in ranked[:num_images]) == expected_before_exclusions


def test_default_num_images_is_twenty():
    assert corpus.DEFAULT_SWESMITH_IMAGES == 20


def test_load_swesmith_rows_default_is_uncapped_and_stays_unique():
    rows = corpus.load_swesmith_rows(20)
    # 23,844 minus the 41 add/delete/rename/copy rows -- `max_test_count`
    # defaults to `None` (uncapped): real measurement showed the cost this
    # would guard against is not a problem for this corpus (see
    # corpus.DEFAULT_SWESMITH_MAX_TEST_COUNT's own docstring).
    assert len(rows) == 23844 - 41 == 23803
    ids = [row.instance_id for row in rows]
    assert len(ids) == len(set(ids))


def test_max_test_count_cap_is_enforced_when_given():
    rows = corpus.load_swesmith_rows(20, max_test_count=100)
    assert rows  # the cap must not empty the corpus at a realistic value
    for row in rows:
        assert len(row.fail_to_pass) + len(row.pass_to_pass) <= 100


def test_max_test_count_default_matches_explicit_none():
    default = corpus.load_swesmith_rows(20)
    explicit_none = corpus.load_swesmith_rows(20, max_test_count=None)
    assert default == explicit_none


def test_the_documented_cap_value_excludes_real_rows():
    uncapped = corpus.load_swesmith_rows(20, max_test_count=None)
    capped = corpus.load_swesmith_rows(20, max_test_count=corpus.DEFAULT_SWESMITH_MAX_TEST_COUNT)
    assert len(capped) < len(uncapped)
    assert len(capped) == 23172


def test_load_swesmith_rows_rejects_a_nonpositive_max_test_count():
    with pytest.raises(ValueError):
        corpus.load_swesmith_rows(20, max_test_count=0)


def test_swesmith_row_defect_flags_empty_fail_to_pass():
    # No row in the pinned revision actually triggers this (checked against
    # all 59,136, not sampled -- see load_swesmith_rows's own docstring), so
    # this exercises the extracted, pure predicate directly rather than
    # relying on `load_swesmith_rows` to find a real row that never exists;
    # a test that only asserted `all(row.fail_to_pass for row in rows)`
    # against the real corpus would pass identically whether or not the
    # filter existed at all.
    from reliquary_swe.corpus import _swesmith_row_defect

    assert _swesmith_row_defect((), ("some::test",), max_test_count=None) == "empty_fail_to_pass"
    assert _swesmith_row_defect(("f::t",), (), max_test_count=None) is None


def test_swesmith_row_defect_flags_too_costly_only_when_capped():
    from reliquary_swe.corpus import _swesmith_row_defect

    f2p, p2p = ("f::a", "f::b"), ("p::a", "p::b", "p::c")  # 5 total
    assert _swesmith_row_defect(f2p, p2p, max_test_count=4) == "too_costly"
    assert _swesmith_row_defect(f2p, p2p, max_test_count=5) is None
    assert _swesmith_row_defect(f2p, p2p, max_test_count=None) is None


def test_swesmith_rows_carry_the_fields_grading_needs():
    row = corpus.load_swesmith_rows(20)[0]
    assert row.instance_id
    assert row.repo
    # No literal base commit exists for a SWE-smith row; base_commit is
    # instead a git revision expression -- see SweRow's own docstring.
    assert row.base_commit == f"origin/{row.instance_id}~1"
    assert row.version == ""
    assert row.test_patch == ""
    assert row.image is not None and row.image.startswith("jyangballin/")
    assert row.fail_to_pass
    assert row.gold_patch.strip()


def test_load_swesmith_rows_rejects_a_nonpositive_image_count():
    with pytest.raises(ValueError):
        corpus.load_swesmith_rows(0)


# Frozen, hand-checked example: a two-line replacement with no surrounding
# edits, matching the shape of a real SWE-smith bug patch.
_FORWARD = """\
diff --git a/x.py b/x.py
index 111..222 100644
--- a/x.py
+++ b/x.py
@@ -1,3 +1,3 @@
 def f():
-    return 1
+    return 2
"""


def test_reverse_unified_diff_swaps_hunk_direction():
    reversed_patch = _reverse_unified_diff(_FORWARD)
    assert "+    return 1" in reversed_patch
    assert "-    return 2" in reversed_patch
    assert "@@ -1,3 +1,3 @@" in reversed_patch  # symmetric hunk: header unchanged


def test_reverse_unified_diff_is_its_own_inverse():
    assert _reverse_unified_diff(_reverse_unified_diff(_FORWARD)) == _FORWARD


def test_reverse_unified_diff_rejects_file_creation():
    added = """\
diff --git a/y.py b/y.py
new file mode 100644
--- /dev/null
+++ b/y.py
@@ -0,0 +1,1 @@
+x = 1
"""
    with pytest.raises(ValueError):
        _reverse_unified_diff(added)


# The actual bug-injection patch for the golden instance used in
# test_swesmith_goldens.py, captured verbatim from the pinned dataset --
# reversing it and applying the result forward, inside a real container, is
# what test_swesmith_goldens.py's reference-patch golden checks. This test
# only pins the *reversal*, not the container outcome.
_OAUTHLIB_BUG_PATCH = """\
diff --git a/oauthlib/oauth2/rfc6749/utils.py b/oauthlib/oauth2/rfc6749/utils.py
index 7dc27b3..c5db6ba 100644
--- a/oauthlib/oauth2/rfc6749/utils.py
+++ b/oauthlib/oauth2/rfc6749/utils.py
@@ -15,8 +15,8 @@ def list_to_scope(scope):
     \"\"\"Convert a list of scopes to a space separated string.\"\"\"
     if isinstance(scope, str) or scope is None:
         return scope
-    elif isinstance(scope, (set, tuple, list)):
-        return " ".join([str(s) for s in scope])
+    elif isinstance(scope, (tuple, list)):
+        return " ".join(str(s) for s in reversed(scope))
     else:
         raise ValueError("Invalid scope (%s), must be string, tuple, set, or list." % scope)
"""


def test_reverse_unified_diff_on_the_real_oauthlib_bug_patch():
    fixed = _reverse_unified_diff(_OAUTHLIB_BUG_PATCH)
    added_lines = [
        line[1:] for line in fixed.splitlines() if line.startswith("+") and not line.startswith("+++")
    ]
    removed_lines = [
        line[1:] for line in fixed.splitlines() if line.startswith("-") and not line.startswith("---")
    ]
    # Reversed = the pre-bug, correct implementation added back: supports
    # `set`, casts with `str()`, and does not reverse the list. The buggy
    # lines are still present, just now on the removed ("-") side.
    assert any("isinstance(scope, (set, tuple, list))" in line for line in added_lines)
    assert any("str(s) for s in scope" in line for line in added_lines)
    assert any("reversed(scope)" in line for line in removed_lines)
    assert not any("reversed(scope)" in line for line in added_lines)


# SWE-bench Verified's own 12 repositories (`corpus.load_rows("eval")`,
# checked directly rather than hand-copied -- see test below).
def test_no_repo_overlap_between_the_two_corpora():
    """Contamination check the task explicitly requires reporting: SWE-smith's
    default (and, checked separately, its full 222-image) corpus must share
    no repository with the SWE-bench Verified evaluation set.
    """
    eval_repos = {row.repo for row in corpus.load_rows("eval")}
    assert len(eval_repos) == 12  # pins the number this test's claim depends on

    def owner_repo(image_name: str) -> str:
        # jyangballin/swesmith.x86_64.<owner>_1776_<repo>.<commit8>
        core = image_name.split(".", 2)[-1].rsplit(".", 1)[0]
        owner, repo = core.split("_1776_", 1)
        return f"{owner}/{repo}"

    for num_images in (20, 222):
        train_repos = {owner_repo(image) for image in corpus.swesmith_image_rank()[:num_images]}
        assert train_repos.isdisjoint(eval_repos)


# 92 of the pinned corpus's 222 repositories are not Python (Go, PHP, Java,
# Rust -- checked against every one of them, not sampled). None of them are
# in the top 30 by task count; the first one appears at rank 31.
def test_ensure_python_profile_accepts_a_python_repo():
    swesmith_adapter.ensure_python_profile("swesmith/oauthlib__oauthlib.1fd52536")


def test_ensure_python_profile_rejects_a_go_repo():
    with pytest.raises(ValueError, match="only supports Python"):
        swesmith_adapter.ensure_python_profile("swesmith/caddyserver__caddy.77dd12cc")


def test_default_corpus_is_entirely_python():
    for repo in {row.repo for row in corpus.load_swesmith_rows(20)}:
        swesmith_adapter.ensure_python_profile(repo)  # must not raise


def test_test_command_activates_the_testbed_env():
    row = corpus.load_swesmith_rows(20)[0]
    argv = swesmith_adapter.test_command(row)
    assert argv[:2] == ["bash", "-c"]
    assert "conda activate testbed" in argv[2]
    assert "pytest" in argv[2]


def test_test_files_covers_every_fail_to_pass_file():
    row = corpus.load_swesmith_rows(20)[0]
    files = swesmith_adapter.test_files(row)
    f2p_file_prefixes = {name.split("::", 1)[0] for name in row.fail_to_pass}
    assert f2p_file_prefixes <= set(files)


def test_unparseable_output_yields_no_results_rather_than_raising():
    row = corpus.load_swesmith_rows(20)[0]
    assert swesmith_adapter.parse_results(row, "") == {}
    assert swesmith_adapter.parse_results(row, "not a pytest log at all") == {}


def test_swesmith_instance_dict_always_carries_fail_to_pass():
    """Pins the invariant `_instance`'s own docstring names as the one that
    actually matters (corrected after review -- an earlier version of both
    the docstring and this test pinned `KEY_PATCH`'s absence instead, which
    changes nothing: `get_test_cmd`'s `min_testing and FAIL_TO_PASS in
    instance` branch returns before a patch key is ever consulted). If
    `FAIL_TO_PASS` were ever dropped from this dict, a `min_testing=True`
    profile would fall through into a live GitHub clone attempt
    (`_get_cached_test_paths` -> `self.clone()`) inside a
    `network_allow=[]` grading box.
    """
    row = corpus.load_swesmith_rows(20)[0]
    instance = swesmith_adapter._instance(row)
    assert "FAIL_TO_PASS" in instance


def test_ensure_no_patch_key_in_swesmith_instance_dict():
    # Independent of the above: `KEY_PATCH` is left out because `row.
    # gold_patch` is the fix, not the bug-introducing diff upstream's own
    # `patch` key means -- not because of which branch of `get_test_cmd`
    # it would otherwise reach (see `_instance`'s own docstring).
    row = corpus.load_swesmith_rows(20)[0]
    instance = swesmith_adapter._instance(row)
    assert "patch" not in instance


def test_only_the_owning_adapter_imports_its_third_party_library():
    """The one-module invariant the design spec establishes for `swebench`
    (`swe_adapter.py` docstring) and this package extends to `swesmith`:
    each external test-harness library is imported behind exactly one name
    we own, so a version bump touches one file. A static source scan, not
    an import-time check, so it catches a future `import swesmith` dropped
    into an unrelated module even if that module is never exercised by any
    other test.
    """
    import ast
    from pathlib import Path

    package_dir = Path(swesmith_adapter.__file__).parent
    owners = {"swebench": "swe_adapter.py", "swesmith": "swesmith_adapter.py"}
    violations = []
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name in owners and path.name != owners[name]:
                    violations.append(f"{path.name} imports {name!r}")
    assert violations == []
