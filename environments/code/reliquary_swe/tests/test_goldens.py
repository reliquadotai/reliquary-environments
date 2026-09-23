import difflib

import pytest
import verifiers.v1 as vf

from conftest import DJANGO_GOLDEN, GOLDEN, SPHINX_GOLDEN
from reliquary_swe import grading, swe_adapter

docker = pytest.mark.docker
slow = pytest.mark.slow


def _unified_diff(path: str, old: str, new: str) -> str:
    """A `git apply`-ready patch rewriting `path` from `old` to `new`,
    computed rather than hand-written so its hunk headers are always
    correct. Used by the entry-point/config-rewrite goldens below, which
    need to replace a real file's content without hand-counting context
    lines.
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


@docker
@slow
async def test_a_patch_that_rewrites_djangos_test_runner_scores_zero(django_runtime):
    # CRITICAL 2's negative control for the django family. Nothing about
    # test_patch, conftest.py, or FAIL_TO_PASS/PASS_TO_PASS names
    # `tests/runtests.py` -- it is simply the file `test_command` executes
    # for this repo -- so a patch rewriting it to fake every result is a
    # total bypass restoration would otherwise never touch. Genuinely slow
    # once restored: this instance's own test_patch touches only .txt
    # fixtures, so get_test_directives is empty and the *real*,
    # restored `runtests.py` runs django's entire suite (~4 min; confirmed
    # already unavoidable in test_adapter.py's own slow test on the same
    # instance).
    data = _data(DJANGO_GOLDEN)
    # The image's default HEAD is not base_commit (confirmed on the box); read
    # the file at the exact commit `grade()` itself checks out, or a diff
    # built against the wrong version may not even apply.
    await django_runtime.run(["git", "checkout", "-q", data.base_commit], {})
    original = (await django_runtime.run(["cat", "tests/runtests.py"], {})).stdout
    fake_lines = [f"{name} ... ok" for name in (*data.fail_to_pass, *data.pass_to_pass)]
    fake_script = (
        "#!/usr/bin/env python\n"
        "import sys\n"
        f"for line in {fake_lines!r}:\n"
        "    print(line)\n"
        "sys.exit(0)\n"
    )
    patch = _unified_diff("tests/runtests.py", original, fake_script)
    report = await grading.grade(django_runtime, data, patch)
    assert report.applied is True
    # A bare `reward == 0.0` cannot tell "the real suite ran and genuinely
    # failed" apart from a collapsed run, a failed restoration, or a flake
    # in a 12,311-test suite -- and would keep passing even if a future
    # change deleted tests/runtests.py outright. `restored is True` confirms
    # grading actually put the real file back (the pre-fix repr shows
    # `restored=False`); `pass_to_pass_passed > 1400` confirms the real
    # suite genuinely ran (all 1432 pass_to_pass entries resolve at base,
    # per task-3-report.md's own measurement of this instance).
    assert report.restored is True
    assert report.pass_to_pass_passed > 1400
    assert report.reward == 0.0


@docker
@slow
async def test_a_patch_that_rewrites_djangos_settings_module_scores_zero(django_runtime):
    # IMPORTANT 1's negative control (fix round 2). The identical bypass as
    # the `runtests.py` control above, through a different, equally
    # unrestored file: `--settings=test_sqlite` is DJANGO_SETTINGS_MODULE,
    # imported by `runtests.py` INSIDE the grading process, before a single
    # test runs. Named by no test_patch in the corpus, no conftest.py, and
    # nothing test_entrypoint returns -- see
    # swe_adapter.test_command_argument_paths. Genuinely slow for the same
    # reason as the sibling control above (this instance's test_patch
    # touches no python files, so a genuine restored run is django's entire
    # suite).
    data = _data(DJANGO_GOLDEN)
    await django_runtime.run(["git", "checkout", "-q", data.base_commit], {})
    original = (await django_runtime.run(["cat", "tests/test_sqlite.py"], {})).stdout
    fake_lines = [f"{name} ... ok" for name in (*data.fail_to_pass, *data.pass_to_pass)]
    fake_settings = (
        "import os, sys\n"
        f"for line in {fake_lines!r}:\n"
        "    print(line)\n"
        # os._exit skips stdio flushing entirely; with stdout block-buffered
        # (no tty), an unflushed print is silently lost the instant this
        # fires -- confirmed by hand: without the explicit flush, this
        # dropped the tail of a 1,432-line pass_to_pass block believably
        # enough to look like a *correct* zero for the wrong reason.
        "sys.stdout.flush()\n"
        "os._exit(0)\n"
    )
    patch = _unified_diff("tests/test_sqlite.py", original, fake_settings)
    report = await grading.grade(django_runtime, data, patch)
    assert report.applied is True
    assert report.restored is True
    assert report.pass_to_pass_passed > 1400
    assert report.reward == 0.0


@docker
async def test_the_reference_patch_scores_one_for_sphinx_too(sphinx_runtime):
    # Not required by any review finding directly, but load-bearing for
    # trusting the two sphinx negative controls below: without this, nothing
    # confirms reward 1.0 is even reachable on this family at all. It is not
    # reachable by default -- sphinx's own pytest invocation (via tox.ini)
    # has no `-rA` and produces no line `parse_results` can read, so even the
    # *gold* patch scored 0 until `swe_adapter.pytest_reporting_fixup` (a
    # ninth silent-zero defect, found while building this golden -- see
    # task-4-report.md) was added.
    data = _data(SPHINX_GOLDEN)
    report = await grading.grade(sphinx_runtime, data, data.gold_patch)
    assert report.applied is True
    assert report.reward == 1.0


@docker
async def test_a_patch_that_rewrites_sphinxs_tox_commands_scores_zero(sphinx_runtime):
    # CRITICAL 2's negative control for the sphinx family. `tox.ini`'s own
    # `[testenv] commands` is what actually runs under `tox --current-env`;
    # nothing test_patch-shaped names it either. Fakes only ONE of this
    # instance's FAIL_TO_PASS entries and does not touch either of
    # test_patch's own (both *added*) paths, so this isolates CRITICAL 2
    # from CRITICAL 1 -- see test_a_patch_that_adds_files_and_squats_a_test_scores_zero
    # for the latter on this same instance.
    data = _data(SPHINX_GOLDEN)
    assert data.fail_to_pass == ("tests/test_ext_autodoc_automodule.py::test_empty_all",)
    # Same reason as the django negative control above: read at base_commit
    # explicitly, not whatever ref the image's container starts at.
    await sphinx_runtime.run(["git", "checkout", "-q", data.base_commit], {})
    original = (await sphinx_runtime.run(["cat", "tox.ini"], {})).stdout
    fake = original.replace(
        "commands=\n    python -X dev -m pytest --durations 25 {posargs}\n",
        "commands=\n"
        "    python -c \"print('PASSED "
        "tests/test_ext_autodoc_automodule.py::test_empty_all')\"\n",
    )
    assert fake != original, "the [testenv] commands= line has moved; update the replace() above"
    patch = _unified_diff("tox.ini", original, fake)
    report = await grading.grade(sphinx_runtime, data, patch)
    assert report.applied is True
    # `restored is True` and a genuine FAILED (not merely absent from
    # `results`) rule out a collapsed run or a failed restoration reading as
    # the same 0.0 this test exists to pin.
    assert report.restored is True
    assert report.results.get(data.fail_to_pass[0]) == "FAILED"
    assert report.reward == 0.0


@docker
async def test_a_patch_that_adds_files_and_squats_a_test_scores_zero(sphinx_runtime):
    # IMPORTANT 6's golden from a different family than every test above:
    # sphinx (not pytest-via-astropy's own bare invocation, but tox), and a
    # test_patch that touches two paths, BOTH of which are additions (per
    # `swe_adapter.get_modified_files` returning [] for it) rather than
    # modifications of anything that exists at base_commit -- exactly the
    # shape CRITICAL 1's batched `git checkout` used to silently restore
    # nothing for.
    #
    # Both of test_patch's added paths are squatted, not just the test file:
    # `swe_adapter.test_command` passes both as explicit pytest targets
    # (get_test_directives derives them from test_patch, unconditionally), and
    # pytest treats a *missing* explicit target as a hard usage error that
    # aborts collection entirely -- confirmed by hand, squatting only the test
    # file scores 0.0 under BOTH the broken and the fixed code, for different
    # reasons (a collection error vs. a real failure), which would make this
    # golden pass either way and prove nothing. Squatting both means every
    # explicit target exists, so old, buggy restoration lets the fake test
    # actually run and report PASSED.
    data = _data(SPHINX_GOLDEN)
    assert grading._paths_touched_by(data.test_patch) and not swe_adapter.get_modified_files(
        data.test_patch
    )
    patch = (
        "diff --git a/tests/roots/test-ext-autodoc/target/empty_all.py"
        " b/tests/roots/test-ext-autodoc/target/empty_all.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/roots/test-ext-autodoc/target/empty_all.py\n"
        "@@ -0,0 +1,1 @@\n"
        "+# squatted, not the real fixture\n"
        "diff --git a/tests/test_ext_autodoc_automodule.py b/tests/test_ext_autodoc_automodule.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/tests/test_ext_autodoc_automodule.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def test_empty_all():\n"
        "+    assert True\n"
    )
    report = await grading.grade(sphinx_runtime, data, patch)
    assert report.applied is True
    # Same reasoning as the two sphinx controls above: restored plus a
    # genuine FAILED, not a bare zero a collapsed run also produces.
    assert report.restored is True
    assert report.results.get(data.fail_to_pass[0]) == "FAILED"
    assert report.reward == 0.0
