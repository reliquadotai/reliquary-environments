"""The four goldens spec section 9 requires, run against SWE-smith's own
restoration strategy (`grading._restore_from_pristine_image`) rather than
`_restore_from_test_patch`. Mirrors `test_goldens.py`'s shape; kept as a
separate file so nothing here risks the existing SWE-bench Verified goldens.
"""

from __future__ import annotations

import difflib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes import provision_runtime

from conftest import _trace, run_gold_episode
from reliquary_swe import corpus, grading
from reliquary_swe.taskset import SweTask, task_for

docker = pytest.mark.docker


def _unified_diff(path: str, old: str, new: str) -> str:
    """A `git apply`-ready patch rewriting `path` from `old` to `new`,
    computed rather than hand-written -- same helper as `test_goldens.py`'s
    own, duplicated rather than imported to keep this file self-contained.
    """
    body = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    return f"diff --git a/{path} b/{path}\n{body}"


def _gold_patch_fingerprints(patch: str) -> list[str]:
    """Distinctive text from the gold patch's own added lines, safe to
    assert absent from a full object-store dump.

    A single added line is not safe on its own: the first cut of this check
    used exactly that (any `+` line over 15 characters) and failed on real
    data -- `        return None` is one of this golden's own added lines,
    and it also occurs, unrelated, in a different file already shipped in
    oauthlib's own tree (a false positive that would have hidden a real
    regression behind noise). Only *contiguous* runs of two or more `+`
    lines are used: in the final file they land adjacent to each other with
    no unrelated context between them, so the joined block is exactly what
    the object store's blob content contains verbatim if the fix survives
    there -- and a two-line-plus block of real code is not the kind of
    thing that coincides by accident.
    """
    lines = patch.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            current.append(line[1:])
        else:
            if len(current) >= 2:
                blocks.append("\n".join(current))
            current = []
    if len(current) >= 2:
        blocks.append("\n".join(current))
    return [block for block in blocks if len(block.strip()) > 30]


# Already pulled on the container host, and already ground-truthed by hand
# (see the implementation report): base_commit checks out clean and cold, the
# dataset's own `patch` breaks 5 sampled fail-to-pass tests when applied
# forward, and its reversal (this instance's `gold_patch`) fixes them.
SWESMITH_GOLDEN = "oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc"


def _train_task(instance_id: str = SWESMITH_GOLDEN) -> SweTask:
    # Built alone, through the same row and task construction the taskset
    # uses: materializing the whole 20-image split to find one instance cost
    # more than 5 GB and took the 7 GB CI runner down with it. That this
    # instance IS in the default split is checked separately, cheaply, by
    # `test_the_golden_is_in_the_default_train_split` below.
    return task_for(corpus.swesmith_row(instance_id), 0, "train")


def test_the_golden_is_in_the_default_train_split():
    rank = corpus.swesmith_image_rank()
    image = corpus.swesmith_row(SWESMITH_GOLDEN).image
    assert image in rank[: corpus.DEFAULT_SWESMITH_IMAGES]


@asynccontextmanager
async def _provisioned(task: SweTask) -> AsyncIterator[vf.Runtime]:
    docker_config = vf.DockerConfig(
        image=task.data.image, workdir=task.data.workdir, allow=task.data.network_allow
    )
    async with provision_runtime(docker_config) as box:
        await box.prepare_setup()
        await box.prepare_execution([])
        yield box


@pytest.fixture
async def swesmith_runtime() -> AsyncIterator[vf.Runtime]:
    async with _provisioned(_train_task()) as box:
        yield box


@docker
async def test_the_reference_patch_scores_one(swesmith_runtime):
    # `data.gold_patch` is corpus.py's reversal of SWE-smith's own `patch`
    # column -- proven necessary and correct three independent ways (see the
    # implementation report): the diff text itself reads as a regression,
    # a real container run showed the sampled fail-to-pass tests passing
    # before `patch` and failing after it, and SWE-smith's own harness
    # applies gold predictions with `git apply --reverse`, commented "fix =
    # revert". This is the one golden that would silently invert every
    # SWE-smith reward if that reversal were ever wrong.
    data = _train_task().data
    report = await grading.grade(swesmith_runtime, data, data.gold_patch)
    assert report.applied is True
    assert report.restored is True
    assert report.reward == 1.0


@docker
async def test_an_empty_patch_scores_zero(swesmith_runtime):
    data = _train_task().data
    report = await grading.grade(swesmith_runtime, data, "")
    assert report.reward == 0.0


@docker
async def test_fail_to_pass_really_fails_before_any_patch(swesmith_runtime):
    data = _train_task().data
    report = await grading.grade(swesmith_runtime, data, "")
    for name in data.fail_to_pass:
        assert report.results.get(name) != "PASSED"


@docker
async def test_pass_to_pass_really_passes_before_any_patch(swesmith_runtime):
    data = _train_task().data
    report = await grading.grade(swesmith_runtime, data, "")
    for name in data.pass_to_pass:
        assert report.results.get(name) == "PASSED"


# The golden that proves `_restore_from_pristine_image` actually restores.
# oauthlib ships no conftest.py anywhere in its tree (confirmed on the box),
# so this is a `new file mode` patch at the repo root -- a hook that forces
# every test's outcome to "passed", the same attack `test_goldens.py`'s own
# FORCE_PASS_PATCH uses against the SWE-bench Verified path. Nothing about
# this file is named by a fail-to-pass/pass-to-pass test id, so only the
# conftest-ancestor walk `_restore_from_pristine_image` shares with the
# SWE-bench Verified strategy -- not the plain per-file test restoration --
# is what removes it.
FORCE_PASS_PATCH = """\
diff --git a/conftest.py b/conftest.py
new file mode 100644
--- /dev/null
+++ b/conftest.py
@@ -0,0 +1,9 @@
+import pytest
+
+
+@pytest.hookimpl(hookwrapper=True)
+def pytest_runtest_makereport(item, call):
+    outcome = yield
+    report = outcome.get_result()
+    if report.when == "call":
+        report.outcome = "passed"
"""


@docker
async def test_a_patch_that_forces_fake_passes_scores_zero(swesmith_runtime):
    data = _train_task().data
    report = await grading.grade(swesmith_runtime, data, FORCE_PASS_PATCH)
    # `applied` must be True: otherwise the zero proves only that git
    # rejected the patch, and `restored` must be True: a grader-side
    # restoration failure reading as "the agent's fault" is exactly the
    # silent-zero shape this whole package exists to avoid.
    assert report.applied is True
    assert report.restored is True
    assert report.reward == 0.0


# The golden the conftest.py attack above does NOT cover: `_restore_from_
# pristine_image`'s *primary* mechanism is the plain per-file checkout of
# each fail-to-pass/pass-to-pass test file, not the conftest-ancestor walk.
# Overwrites a real fail-to-pass test file outright, rather than adding a
# new one, so the interesting outcome is `applied is True` (the tampering
# lands cleanly) with the fix still absent -- the same shape as
# `test_goldens.py`'s django/settings-module controls, just via the file
# checkout rather than the ancestor walk.
_TAMPERED_TEST_UTILS = """\
from tests.unittest import TestCase


class UtilsTests(TestCase):
    def test_host_from_uri(self):
        pass

    def test_list_to_scope(self):
        pass

    def test_params_from_uri(self):
        pass

    def test_scope_to_list(self):
        pass
"""

_TAMPERED_NAMES = tuple(
    f"tests/oauth2/rfc6749/test_utils.py::UtilsTests::{name}"
    for name in (
        "test_host_from_uri",
        "test_list_to_scope",
        "test_params_from_uri",
        "test_scope_to_list",
    )
)


@docker
async def test_a_patch_that_overwrites_a_test_file_is_reverted(swesmith_runtime):
    # The image's default HEAD is not base_commit; read the file at the
    # exact commit `grade()` itself checks out, matching
    # `test_goldens.py`'s own django controls, or a diff built against the
    # wrong version may not even apply.
    await swesmith_runtime.run(["git", "checkout", "-q", "origin/" + SWESMITH_GOLDEN + "~1"], {})
    original = (
        await swesmith_runtime.run(["cat", "tests/oauth2/rfc6749/test_utils.py"], {})
    ).stdout
    patch = _unified_diff(
        "tests/oauth2/rfc6749/test_utils.py", original, _TAMPERED_TEST_UTILS
    )
    data = _train_task().data
    report = await grading.grade(swesmith_runtime, data, patch)
    assert report.applied is True
    assert report.restored is True
    # The tampered names must show the real, pre-fix outcome (FAILED --
    # confirmed by a separate golden above that these four genuinely fail on
    # an untouched checkout) rather than the forced pass the tamper tried to
    # substitute. Checking `results` directly, not just `reward == 0.0`:
    # every other fail-to-pass test is untouched by this patch and would
    # keep the reward at 0.0 even if this specific restoration were
    # completely broken, so `reward` alone would not discriminate.
    for name in _TAMPERED_NAMES:
        assert report.results.get(name) == "FAILED"
    assert report.reward == 0.0


@docker
async def test_setup_raises_on_a_mis_shaped_branch_rather_than_paying_an_empty_patch():
    # CRITICAL/IMPORTANT 3's own failure mode, observed directly rather than
    # inferred: upstream only creates the "Remove F2P Tests" commit when a
    # bug's fail-to-pass entries derive at least one file (`gather.py`'s
    # `if f2p_test_files:`). A row that never got that commit has a 2-commit
    # branch, not 3, and `~1` from its tip lands on the *pristine* "Initial
    # commit" instead of "Bug Patch" -- no bug present, so an EMPTY patch
    # would score every fail-to-pass test PASSED for free. Simulated here by
    # pointing `base_commit` at the real branch's own pristine ancestor
    # (`origin/main`, confirmed by hand to have commit message "Initial
    # commit", never "Bug Patch") rather than waiting for such a row to
    # exist in the dataset -- none does today (checked against all 59,136
    # rows), which is exactly why this needs a direct simulation, not a
    # fixture.
    task = _train_task()
    bad_data = task.data.model_copy(
        update={"base_commit": "origin/main"}, deep=False
    )
    bad_task = SweTask(bad_data)
    async with _provisioned(bad_task) as box:
        with pytest.raises(RuntimeError, match="environment preparation failed"):
            await bad_task.setup(_trace(bad_task), box)


@docker
async def test_setup_leaves_no_trace_of_the_gold_patch_in_the_object_store():
    # CRITICAL 1's actual gate. Everything this asserts was previously only
    # a raw shell probe transcribed into a comment -- a review caught that
    # no test ran either check, and that `test_episode_reference_patch_
    # scores_one` above passes identically with `_TRAIN_GUARD_AND_REROOT`
    # deleted (it only checks the reward, and the bug is still fixable by
    # deriving it normally, so removing the reroot step changes nothing
    # that test can see). A commit count alone would not catch a dangling
    # blob either -- checked directly, both matter.
    task = _train_task()
    async with _provisioned(task) as box:
        await task.setup(_trace(task), box)
        count = await box.run(["git", "rev-list", "--count", "HEAD"], {})
        assert count.stdout.strip() == "1"
        scan = await box.run(["sh", "-c", "git cat-file --batch-all-objects --batch"], {})
        fingerprints = _gold_patch_fingerprints(task.data.gold_patch)
        assert fingerprints  # the fixture must actually exercise this check
        for line in fingerprints:
            assert line not in scan.stdout, f"gold patch line survives in object store: {line!r}"


@docker
async def test_grading_leaves_no_trace_of_the_gold_patch_in_the_object_store(swesmith_runtime):
    # The same leak, one box over: `grading.grade` checks out `base_commit`
    # in a box freshly provisioned from the image, never put through
    # `setup()`'s cleanup, so this box has the same two-commit history with
    # the fix one parent away until `grade` itself severs it. The agent's
    # patched source *executes* here, after restoration, during the real
    # test run -- a patch that reads git objects at import time to recover
    # the fix would make the suite genuinely pass without deriving
    # anything, which no restoration strategy defends against by
    # construction (grading.py's own docstring already disclaims the
    # sibling monkeypatch vector; this is a second way into the same
    # residual). The patch argument does not matter for this check --
    # empty is enough, since the object-store state after `grade` returns
    # is what is being tested, not the reward.
    data = _train_task().data
    await grading.grade(swesmith_runtime, data, "")
    count = await swesmith_runtime.run(["git", "rev-list", "--count", "HEAD"], {})
    assert count.stdout.strip() == "1"
    scan = await swesmith_runtime.run(
        ["sh", "-c", "git cat-file --batch-all-objects --batch"], {}
    )
    fingerprints = _gold_patch_fingerprints(data.gold_patch)
    assert fingerprints
    for line in fingerprints:
        assert line not in scan.stdout, f"gold patch line survives in object store: {line!r}"


@docker
async def test_episode_reference_patch_scores_one():
    # Exercises the *real* setup()->finalize()->env.finalize() pipeline end
    # to end for a train task -- not `grading.grade` called directly, like
    # every golden above -- so it is the one test that actually runs
    # `SweTask.setup()`'s new SWE-smith-only guard-and-reroot step
    # (`taskset._TRAIN_GUARD_AND_REROOT`) rather than assuming it works.
    episode = await run_gold_episode(_train_task())
    solution = episode.traces[0]
    assert solution.rewards["patch_passes_tests"].score == 1.0
