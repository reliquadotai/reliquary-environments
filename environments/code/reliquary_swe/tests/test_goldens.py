import pytest
import verifiers.v1 as vf

from conftest import GOLDEN
from reliquary_swe import grading

docker = pytest.mark.docker

# A patch that tampers with collection rather than fixing anything. It applies
# cleanly, which is the point: the zero must come from the tampering being
# discarded (or, here, from every collected test reporting SKIPPED rather than
# PASSED), not from `git apply` refusing it. Astropy already ships a root
# `conftest.py` (confirmed on the box), so this modifies it rather than adding
# a new one -- a `new file mode` version of this patch fails to apply here
# with "already exists in working directory", which would make the golden
# vacuous for the wrong reason.
TEST_EDITING_PATCH = """\
diff --git a/conftest.py b/conftest.py
--- a/conftest.py
+++ b/conftest.py
@@ -62,3 +62,7 @@
 # them after testing, because they are only changed for the duration of the
 # Python process, and this configuration only matters if running pytest
 # directly, not from e.g. an IPython session.
+
+def pytest_collection_modifyitems(items):
+    for item in items:
+        item.add_marker("skip")
"""

# A stronger version of the same attack: instead of skipping (which a naive
# reading might credit to luck -- SKIPPED just happens not to equal PASSED),
# this forces every test's *outcome* to "passed" outright, via a conftest.py
# hook `test_patch` never touches and never will (it lives at the repo root,
# `test_patch` only ever touches files under the instance's own test
# directories). Confirmed by hand before the fix: without walking conftest.py
# ancestors, this patch alone -- with no real fix behind it -- scored 1.0.
FORCE_PASS_PATCH = """\
diff --git a/conftest.py b/conftest.py
--- a/conftest.py
+++ b/conftest.py
@@ -62,3 +62,12 @@
 # them after testing, because they are only changed for the duration of the
 # Python process, and this configuration only matters if running pytest
 # directly, not from e.g. an IPython session.
+
+import pytest
+
+@pytest.hookimpl(hookwrapper=True)
+def pytest_runtest_makereport(item, call):
+    outcome = yield
+    report = outcome.get_result()
+    if report.when == "call":
+        report.outcome = "passed"
"""


def _data(instance_id: str):
    config = vf.taskset_config_type("reliquary-swe")
    for task in vf.load_taskset(config(id="reliquary-swe")):
        if task.data.instance_id == instance_id:
            return task.data
    raise AssertionError(f"{instance_id} is not in the corpus")


@docker
async def test_the_reference_patch_scores_one(grading_runtime):
    data = _data(GOLDEN)
    report = await grading.grade(grading_runtime, data, data.gold_patch)
    assert report.applied
    assert report.reward == 1.0


@docker
async def test_an_empty_patch_scores_zero(grading_runtime):
    data = _data(GOLDEN)
    report = await grading.grade(grading_runtime, data, "")
    assert report.reward == 0.0


@docker
async def test_a_patch_that_edits_the_tests_scores_zero(grading_runtime):
    # The golden that justifies the whole design. `applied` must be True:
    # otherwise the zero proves only that git rejected the patch, and the test
    # is worthless.
    data = _data(GOLDEN)
    report = await grading.grade(grading_runtime, data, TEST_EDITING_PATCH)
    assert report.applied is True
    assert report.reward == 0.0


@docker
async def test_a_patch_that_forces_fake_passes_scores_zero(grading_runtime):
    # The conftest.py this forces its hook into is not among test_patch's own
    # touched paths, so `applied is True` here is the interesting invariant:
    # the tampering applies cleanly and still cannot survive the transfer.
    data = _data(GOLDEN)
    report = await grading.grade(grading_runtime, data, FORCE_PASS_PATCH)
    assert report.applied is True
    assert report.reward == 0.0


@docker
async def test_fail_to_pass_really_fails_before_any_patch(grading_runtime):
    data = _data(GOLDEN)
    report = await grading.grade(grading_runtime, data, "")
    for name in data.fail_to_pass:
        assert report.results.get(name) != "PASSED"


@docker
async def test_pass_to_pass_really_passes_before_any_patch(grading_runtime):
    # If these do not pass on an untouched checkout, the instance is broken or
    # flaky and no reward computed from it means anything.
    data = _data(GOLDEN)
    report = await grading.grade(grading_runtime, data, "")
    for name in data.pass_to_pass:
        assert report.results.get(name) == "PASSED"
