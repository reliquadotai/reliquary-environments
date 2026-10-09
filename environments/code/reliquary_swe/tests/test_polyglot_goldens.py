"""Goldens for the polyglot corpus, against real images.

Upstream publishes no fix for these tasks, so the reference patch a golden
needs is recovered from the image itself: a shape-one image (see
`SweTask.setup`) parks the complete upstream under `origin/<branch>`, and
the diff from the "task base" commit to it is a fix the hidden tests accept
-- checked by hand before this file existed, and checked again below.
"""

from __future__ import annotations

import pytest
import verifiers.v1 as vf
from conftest import _trace, provisioned_runtime, run_gold_episode

from reliquary_swe import grading
from reliquary_swe.taskset import BASE_REF, SweTask

docker = pytest.mark.docker

# Shape one, Go, 0.49 GB -- the smallest image in the corpus. HEAD is a
# parentless "task base" commit with spew's method-rendering support cut
# out; `origin/master` still carries it, and a test identifier that only the
# removed code defines (`DisableMethods`) makes a reliable fingerprint.
SHAPE_ONE = "format-code-task-002703"
SHAPE_ONE_FINGERPRINT = "DisableMethods"

# Shape two, Python, /testbed: HEAD is upstream's own tip (198 commits of
# real history, no descendants anywhere), so there is no reference fix to
# recover -- only the negative control.
SHAPE_TWO = "format-code-task-001647"


def _polyglot_task(instance_id: str) -> SweTask:
    config = vf.taskset_config_type("reliquary-swe")
    for task in vf.load_taskset(config(id="reliquary-swe", split="polyglot")):
        if task.data.instance_id == instance_id:
            return task
    raise AssertionError(f"{instance_id} is not in the polyglot taskset")


async def _reference_patch(task: SweTask) -> str:
    """The diff from a pristine box's HEAD to the upstream it still parks,
    computed in a box nobody else touches."""
    async with provisioned_runtime(task) as box:
        upstream = await box.run(
            ["sh", "-c", "git for-each-ref --format='%(refname)' refs/remotes | grep -v HEAD"],
            {},
        )
        ref = upstream.stdout.split()[0]
        diff = await box.run(["git", "diff", "--binary", "HEAD", ref], {})
        assert diff.exit_code == 0, diff.stderr
        return diff.stdout


@docker
async def test_setup_prunes_the_parked_upstream_and_keeps_head():
    task = _polyglot_task(SHAPE_ONE)
    async with provisioned_runtime(task) as box:
        before = await box.run(
            ["sh", "-c", f"git cat-file --batch-all-objects --batch | grep -c {SHAPE_ONE_FINGERPRINT}"],
            {},
        )
        assert int(before.stdout.strip() or 0) > 0, "the fingerprint is not in the image at all"
        await task.setup(_trace(task), box)
        after = await box.run(
            ["sh", "-c", f"git cat-file --batch-all-objects --batch | grep -c {SHAPE_ONE_FINGERPRINT}"],
            {},
        )
        assert after.stdout.strip() == "0"
        # The trap `--detach` exists for: HEAD must still name a commit.
        head = await box.run(["git", "rev-parse", "--verify", "HEAD"], {})
        assert head.exit_code == 0, head.stderr
        # The one ref setup leaves is its own record of the base, at HEAD.
        refs = await box.run(["git", "for-each-ref", "--format=%(refname) %(objectname)"], {})
        assert refs.stdout.split() == [BASE_REF, head.stdout.strip()]


@docker
async def test_the_reference_patch_scores_one_through_the_whole_loop():
    task = _polyglot_task(SHAPE_ONE)
    reference = await _reference_patch(task)
    assert SHAPE_ONE_FINGERPRINT in reference
    episode = await run_gold_episode(
        SweTask(task.data.model_copy(update={"gold_patch": reference}))
    )
    trace = episode.traces[0]
    report = trace.info["swe_report"]
    assert report["applied"] is True
    assert report["restored"] is True
    assert trace.reward == 1.0, report["test_output_tail"]


@docker
async def test_an_empty_patch_scores_zero():
    task = _polyglot_task(SHAPE_ONE)
    async with provisioned_runtime(task) as box:
        report = await grading.grade(box, task.data, "")
    assert report.restored is True
    assert report.reward == 0.0
    assert report.test_command_exit_code not in (0, None)


@docker
async def test_a_guessed_hidden_test_file_does_not_survive_into_grading():
    # An agent that writes a trivially passing test at the very path the
    # hidden one lands on must not have it graded instead of the real one.
    task = _polyglot_task(SHAPE_ONE)
    hidden = next(
        path
        for path in grading._paths_touched_by(task.data.test_patch)
        if path.endswith("_test.go")
    )
    fake = (
        f"diff --git a/{hidden} b/{hidden}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{hidden}\n"
        "@@ -0,0 +1,5 @@\n"
        "+package spew_test\n"
        "+\n"
        '+import "testing"\n'
        "+\n"
        "+func TestNothing(t *testing.T) {}\n"
    )
    async with provisioned_runtime(task) as box:
        report = await grading.grade(box, task.data, fake)
    assert report.applied is True
    assert report.reward == 0.0


@docker
async def test_a_rewritten_test_script_does_not_survive_into_grading():
    # `mimo_test_command.sh` IS the verdict. A patch that ships its own,
    # exiting 0, must be graded against the corpus's script instead.
    task = _polyglot_task(SHAPE_ONE)
    forged = (
        "diff --git a/mimo_test_command.sh b/mimo_test_command.sh\n"
        "new file mode 100755\n"
        "--- /dev/null\n"
        "+++ b/mimo_test_command.sh\n"
        "@@ -0,0 +1,2 @@\n"
        "+#!/usr/bin/env bash\n"
        "+exit 0\n"
    )
    async with provisioned_runtime(task) as box:
        report = await grading.grade(box, task.data, forged)
    assert report.applied is True
    assert report.reward == 0.0


@docker
async def test_a_go_mod_replace_does_not_survive_into_grading():
    # Rewriting the module path in `go.mod` is one line of source-confined
    # patch that changes what `go test ./...` compiles against.
    task = _polyglot_task(SHAPE_ONE)
    async with provisioned_runtime(task) as box:
        original = (await box.run(["cat", "go.mod"], {})).stdout
    lines = original.splitlines()
    tampered = "\n".join([*lines, "// tampered"]) + "\n"
    body = "".join(
        f"-{line}\n" for line in lines
    ) + "".join(f"+{line}\n" for line in tampered.splitlines())
    patch = (
        "diff --git a/go.mod b/go.mod\n"
        "--- a/go.mod\n"
        "+++ b/go.mod\n"
        f"@@ -1,{len(lines)} +1,{len(lines) + 1} @@\n"
        f"{body}"
    )
    async with provisioned_runtime(task) as box:
        report = await grading.grade(box, task.data, patch)
        after = (await box.run(["cat", "go.mod"], {})).stdout
    assert report.applied is True
    assert report.restored is True
    assert after == original


@docker
async def test_shape_two_keeps_its_history_and_an_empty_patch_scores_zero():
    task = _polyglot_task(SHAPE_TWO)
    async with provisioned_runtime(task) as box:
        await task.setup(_trace(task), box)
        count = await box.run(["git", "rev-list", "--count", "HEAD"], {})
        # Real upstream history is context an agent may legitimately read;
        # nothing after HEAD exists to leak, so nothing is re-rooted away.
        assert int(count.stdout.strip()) > 1
    async with provisioned_runtime(task) as box:
        report = await grading.grade(box, task.data, "")
    assert report.reward == 0.0
    assert report.test_command_exit_code not in (0, None)
