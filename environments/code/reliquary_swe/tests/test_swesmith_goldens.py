"""The four goldens spec section 9 requires, run against SWE-smith's own
restoration strategy (`grading._restore_from_pristine_image`) rather than
`_restore_from_test_patch`. Mirrors `test_goldens.py`'s shape; kept as a
separate file so nothing here risks the existing SWE-bench Verified goldens.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes import provision_runtime

from reliquary_swe import grading
from reliquary_swe.taskset import SweTask

docker = pytest.mark.docker

# Already pulled on the container host, and already ground-truthed by hand
# (see the implementation report): base_commit checks out clean and cold, the
# dataset's own `patch` breaks 5 sampled fail-to-pass tests when applied
# forward, and its reversal (this instance's `gold_patch`) fixes them.
SWESMITH_GOLDEN = "oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc"


def _train_task(instance_id: str = SWESMITH_GOLDEN) -> SweTask:
    config_cls = vf.taskset_config_type("reliquary-swe")
    taskset = vf.load_taskset(config_cls(id="reliquary-swe", split="train", num_images=20))
    for task in taskset:
        if task.data.instance_id == instance_id:
            return task
    raise AssertionError(f"{instance_id} is not in the top-20-image train taskset")


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
