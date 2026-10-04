"""Goldens for the R2E corpus, against real images.

Five tasks from five repositories, chosen to cover both test runners R2E
ships: coveragepy, pandas, numpy and orange3 run pytest; tornado runs its
own `r2e_tests/tornado_unittest_runner.py`, which prints a pytest-shaped
summary. The numpy and orange3 tasks expect FAILED and ERROR entries, so
their gold run checks that no expected map relied on upstream's
anywhere-in-the-line status match (see `grading.parse_log_pytest`).

All `slow`: each pulls a 0.8-1.5 GB image and CI's every-push job
deselects them (see .github/workflows/ci.yml); run them on a container host.
For each: the fix commit is unreachable once setup has run (every R2E image
carries the full upstream history, fix included), an empty patch scores 0,
and the gold patch scores 1 through the whole loop.
"""

from __future__ import annotations

import json

import pytest
from conftest import _trace, provisioned_runtime, run_gold_episode

from reliquary_swe import corpus, grading
from reliquary_swe.taskset import SweTask, task_for

docker = pytest.mark.docker
pytestmark = pytest.mark.slow

COVERAGEPY = "c1bfa7352368b63f3a9b30c02f242408d07a7ab2"
TORNADO = "b5ec807edc83c8e7d1d12553d635ebe765e5c614"
PANDAS = "fadb72cf5ef8489e409d4d33625bd16a76fa7a42"
# Expects 3 FAILED among 16.
NUMPY = "ebe2cfb68586208bb096a575a603d00da5ee3887"
# Expects 6 FAILED and 1 ERROR among 24.
ORANGE3 = "c0174f909e8fbde9564c462383137a72365ab3c1"
GOLDENS = [COVERAGEPY, TORNADO, PANDAS, NUMPY, ORANGE3]


def _r2e_task(fix_commit: str) -> SweTask:
    return task_for(corpus.r2e_row(fix_commit), 0, "r2e")


async def _fix_is_reachable(box, fix_commit: str) -> bool:
    found = await box.run(["git", "cat-file", "-t", fix_commit], {})
    return found.exit_code == 0 and found.stdout.strip() == "commit"


@docker
@pytest.mark.parametrize("fix_commit", GOLDENS)
async def test_setup_makes_the_fix_commit_unreachable_and_hides_the_tests(fix_commit):
    task = _r2e_task(fix_commit)
    async with provisioned_runtime(task) as box:
        assert await _fix_is_reachable(box, fix_commit), "the image no longer carries the fix"
        # Nothing the strip cannot vouch for (see taskset._R2E_LEAK_GUARD).
        borrowed = await box.run(
            [
                "sh",
                "-c",
                "test ! -s .git/objects/info/alternates && test ! -e .git/worktrees "
                "&& test ! -e .git/shallow && test ! -e .git/info/grafts",
            ],
            {},
        )
        assert borrowed.exit_code == 0
        # The content of every tracked file that exists on disk.
        tree = ["sh", "-c", "git ls-files -z | xargs -0 md5sum 2>/dev/null"]
        before = (await box.run(tree, {})).stdout
        await task.setup(_trace(task), box)
        # The image's own working tree is the base, untouched -- not HEAD's
        # (pandas: its image edits setup.cfg and deletes pyproject.toml, and
        # HEAD's versions stop pytest from starting).
        after = (await box.run(tree, {})).stdout
        assert after == before
        dirty = await box.run(["git", "status", "--porcelain", "--untracked-files=no"], {})
        assert dirty.stdout.strip() == ""
        assert not await _fix_is_reachable(box, fix_commit)
        refs = await box.run(["git", "for-each-ref"], {})
        assert refs.stdout.strip() == ""
        head = await box.run(["git", "rev-parse", "--verify", "HEAD"], {})
        assert head.exit_code == 0, head.stderr
        # Real history before the bug is context the agent may read.
        count = await box.run(["git", "rev-list", "--count", "HEAD"], {})
        assert int(count.stdout.strip()) > 1
        hidden = await box.run(
            ["sh", "-c", "ls -d /r2e_tests /testbed/run_tests.sh /testbed/r2e_tests"], {}
        )
        assert hidden.exit_code != 0 and hidden.stdout.strip() == ""


@docker
@pytest.mark.parametrize("fix_commit", GOLDENS)
async def test_an_empty_patch_scores_zero(fix_commit):
    task = _r2e_task(fix_commit)
    async with provisioned_runtime(task) as box:
        report = await grading.grade(box, task.data, "")
        # The grading box runs the agent's code: the fix must not be
        # recoverable from it either.
        assert not await _fix_is_reachable(box, fix_commit)
    assert report.restored is True
    # A real run with a real summary, not a runner that never started.
    assert report.results_parsed > 0, report.test_output_tail
    assert report.reward == 0.0


@docker
@pytest.mark.parametrize("fix_commit", GOLDENS)
async def test_the_gold_patch_scores_one_through_the_whole_loop(fix_commit):
    task = _r2e_task(fix_commit)
    assert task.data.gold_patch
    episode = await run_gold_episode(task)
    trace = episode.traces[0]
    report = trace.info["swe_report"]
    assert report["applied"] is True
    assert report["restored"] is True
    assert trace.reward == 1.0, report["test_output_tail"]


def _new_file(path: str, body: str) -> str:
    lines = body.splitlines()
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n" + "".join(f"+{line}\n" for line in lines)
    )


@docker
async def test_a_forged_runner_and_forged_tests_do_not_survive_into_grading():
    # The agent never sees the hidden tests, but it can guess where they go:
    # a `run_tests.sh` that prints a passing summary for every expected name,
    # and an `r2e_tests/` of its own.
    task = _r2e_task(COVERAGEPY)
    names = json.loads(task.data.expected_output_json)
    summary = "\n".join(
        ["echo '=== short test summary info ==='"]
        + [f"echo '{status} r2e_tests/test_1.py::{name.replace('.', '::')}'" for name, status in names.items()]
    )
    forged = _new_file("run_tests.sh", summary) + _new_file(
        "r2e_tests/test_1.py", "def test_nothing():\n    pass\n"
    )
    async with provisioned_runtime(task) as box:
        report = await grading.grade(box, task.data, forged)
    assert report.applied is True
    assert report.restored is True
    assert report.reward == 0.0


@docker
async def test_a_root_conftest_forcing_passes_does_not_survive_into_grading():
    # Measured by hand on this image, without the fix: with this conftest.py
    # left in place, all 7 hidden tests report PASSED -- this task's exact
    # expected map, a paid 1.0.
    task = _r2e_task(COVERAGEPY)
    conftest = (
        "import pytest\n"
        "\n"
        "@pytest.hookimpl(hookwrapper=True)\n"
        "def pytest_runtest_makereport(item, call):\n"
        "    outcome = yield\n"
        "    outcome.get_result().outcome = 'passed'\n"
    )
    async with provisioned_runtime(task) as box:
        report = await grading.grade(box, task.data, _new_file("conftest.py", conftest))
        left = await box.run(["test", "-e", "conftest.py"], {})
    assert report.applied is True
    assert report.restored is True
    assert left.exit_code != 0
    assert report.reward == 0.0


@docker
async def test_a_root_module_shadowing_pytest_does_not_survive_into_grading():
    # `python -m pytest` from the root puts the root first on sys.path:
    # measured by hand on this image, a root `pytest.py` replaces pytest
    # outright. This one runs the real thing and then prints every expected
    # name as PASSED.
    task = _r2e_task(COVERAGEPY)
    names = json.loads(task.data.expected_output_json)
    shim = "\n".join(
        ["import sys", "print('=== short test summary info ===')"]
        + [f"print('PASSED r2e_tests/test_1.py::{name.replace('.', '::')}')" for name in names]
        + ["sys.exit(0)"]
    )
    async with provisioned_runtime(task) as box:
        report = await grading.grade(box, task.data, _new_file("pytest.py", shim))
        left = await box.run(["test", "-e", "pytest.py"], {})
    assert report.applied is True
    assert report.restored is True
    assert left.exit_code != 0
    assert report.reward == 0.0


@docker
async def test_a_pth_file_shipped_into_the_venv_does_not_run():
    # `.venv` is ignored (uv writes `.venv/.gitignore` = `*`) and already in
    # the image, so deleting new root entries never reaches it, and
    # `run_tests.sh` runs `.venv/bin/python`: a `.pth` there executes at
    # interpreter start-up. An agent that un-ignores it can ship one. This
    # one prints a passing summary for every expected name and exits --
    # measured by hand on this image: written into site-packages, the same
    # line makes `bash run_tests.sh` print exactly that and exit 0.
    task = _r2e_task(COVERAGEPY)
    async with provisioned_runtime(task) as box:
        site = (
            await box.run(["sh", "-c", "ls -d .venv/lib/python*/site-packages"], {})
        ).stdout.strip()
    assert site.startswith(".venv/lib/python")
    names = json.loads(task.data.expected_output_json)
    lines = ["=== short test summary info ==="] + [
        f"PASSED r2e_tests/test_1.py::{name.replace('.', '::')}" for name in names
    ]
    pth = (
        "import os, sys; sys.stdout.write("
        + repr("\n".join(lines) + "\n")
        + "); sys.stdout.flush(); os._exit(0)\n"
    )
    forged = _new_file(f"{site}/zz.pth", pth)
    async with provisioned_runtime(task) as box:
        report = await grading.grade(box, task.data, forged)
        left = await box.run(["test", "-e", f"{site}/zz.pth"], {})
    assert report.applied is False
    assert report.reward == 0.0
    assert ".venv/" in report.test_output_tail
    assert left.exit_code != 0
