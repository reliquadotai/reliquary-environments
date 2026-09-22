"""Score a patch inside a box the agent never touched.

Order matters and is the whole design. The agent's patch is applied first, then
the instance's own test patch is applied on top of a checkout of the test files
as they exist at the base commit. So a `conftest.py` that skips everything, a
monkeypatched framework, or an edited test file is overwritten before a single
test runs -- it can never be rewarded, only recorded.

Receives a live `Runtime` and never provisions one: provisioning, retries and
timeouts belong to Task 5's `env.py`. That seam is what lets this module be
tested against any box and keeps the retry policy in one place.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import verifiers.v1 as vf

from reliquary_swe import swe_adapter
from reliquary_swe.corpus import SweRow
from reliquary_swe.taskset import SweData

_DIFF_TARGET = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Report:
    reward: float
    applied: bool
    fail_to_pass_passed: int
    pass_to_pass_passed: int
    results: dict[str, str] = field(default_factory=dict)


def _paths_touched_by(patch: str) -> list[str]:
    """The files a unified diff writes to (the `+++ b/...` side of each hunk)."""
    return [match.group(1) for match in _DIFF_TARGET.finditer(patch)]


def _row_for(data: SweData) -> SweRow:
    """Rebuild the corpus row `swe_adapter`'s calls need.

    `swe_adapter` is typed against `SweRow`, not the taskset's wire-shaped
    `SweData`; `problem_statement` is dropped because grading never reads it.
    """
    return SweRow(
        instance_id=data.instance_id,
        repo=data.repo,
        base_commit=data.base_commit,
        version=data.version,
        problem_statement="",
        fail_to_pass=data.fail_to_pass,
        pass_to_pass=data.pass_to_pass,
        gold_patch=data.gold_patch,
        test_patch=data.test_patch,
    )


RestoreTests = Callable[[vf.Runtime, SweData], Awaitable[None]]


def _conftest_ancestors(paths: list[str]) -> list[str]:
    """Every `conftest.py` pytest could load on the way to `paths`.

    pytest auto-loads a `conftest.py` from every directory between a
    collected file and the repository root, not only ones a path explicitly
    names. Restoring only `test_patch`'s own touched paths leaves that walk
    uncovered: confirmed by hand, a `conftest.py` added at the repo root --
    nowhere near `test_patch`'s own path list -- with a
    `pytest_runtest_makereport` hook forcing every outcome to "passed" earned
    full reward on an otherwise-untouched checkout. Scoped to `conftest.py`
    specifically (not `pytest.ini`/`setup.cfg`/etc.): it is the vector this
    was actually observed on, and the one a policy is most likely to reach
    for first.
    """
    seen: dict[str, None] = {}
    for path in paths:
        directory = PurePosixPath(path).parent
        while True:
            seen[str(directory / "conftest.py")] = None
            if str(directory) == ".":
                break
            directory = directory.parent
    return list(seen)


async def _restore_from_test_patch(runtime: vf.Runtime, data: SweData) -> None:
    """SWE-bench Verified's strategy: put back, at `base_commit`, exactly the
    files `test_patch` touches, then reapply it -- matching upstream's own
    `make_eval_script_list_py` (`git checkout {base_commit} {test_files}` then
    `git apply` the test patch) command for command, since comparability to
    published SWE-bench numbers is the only reason to use this corpus at all.
    Also restores every ancestor `conftest.py` (see `_conftest_ancestors`),
    which upstream's own script does not -- upstream never scores an
    adversarial patch, so nothing there needed this either.
    """
    paths = _paths_touched_by(data.test_patch)
    if paths:
        await runtime.run(["git", "checkout", data.base_commit, "--", *paths], {})
    for conftest in _conftest_ancestors(paths):
        restored = await runtime.run(["git", "checkout", data.base_commit, "--", conftest], {})
        if restored.exit_code != 0:
            # Not present at base_commit: whatever is at this path now was
            # added by the patch under test, with no baseline to compare
            # against, so it comes out rather than staying in place.
            await runtime.run(["rm", "-f", conftest], {})
    if data.test_patch.strip():
        await runtime.write("/tmp/tests.diff", data.test_patch.encode())
        await runtime.run(["git", "apply", "-v", "/tmp/tests.diff"], {})


def _restore_strategy_for(data: SweData) -> RestoreTests:
    """Choose how to make the tests be what they should be -- from the data
    itself, never a step `grade()` hardcodes.

    Every row today (SWE-bench Verified) carries a `test_patch`, so that is
    the only strategy this returns. The training corpus intended to follow it,
    SWE-smith, carries none: it injects its bug into the source and ships
    already-pristine tests in the image, so nothing needs restoring there --
    that branch lands here, keyed off the field that actually distinguishes
    the two corpora, without `grade()` changing at all.
    """
    if data.test_patch.strip():
        return _restore_from_test_patch

    async def _nothing_to_restore(runtime: vf.Runtime, data: SweData) -> None:
        return None

    return _nothing_to_restore


async def grade(runtime: vf.Runtime, data: SweData, patch: str) -> Report:
    """Apply `patch` in `runtime`, restore the tests, run them, and score.

    `runtime` must be a freshly provisioned box from this instance's image.
    Passing the agent's own box would defeat the entire point.

    Reward is binary: every fail-to-pass and every pass-to-pass entry must
    pass, or it's zero. A fractional reward would pay for a half-repair, and a
    half-repair is not a repair.
    """
    checkout = await runtime.run(["git", "checkout", "-q", data.base_commit], {})
    if checkout.exit_code != 0:
        # Every image ships its repo already at base_commit; failing to reach
        # it back is not a real "0 result", it's a box that isn't what it
        # claims to be -- raising here is what keeps that from reading as a
        # silent, wrong zero.
        raise RuntimeError(
            f"could not check out {data.base_commit} for {data.instance_id}: "
            f"{(checkout.stderr or checkout.stdout).strip()[-500:]}"
        )

    applied = True
    if patch.strip():
        await runtime.write("/tmp/agent.diff", patch.encode())
        result = await runtime.run(["git", "apply", "-v", "/tmp/agent.diff"], {})
        applied = result.exit_code == 0
        if not applied:
            # A patch that will not apply changed nothing, which is reward 0 --
            # a real outcome, not an infrastructure failure.
            return Report(0.0, False, 0, 0, {})

    await _restore_strategy_for(data)(runtime, data)

    row = _row_for(data)
    run = await runtime.run(swe_adapter.test_command(row), {})
    results = swe_adapter.parse_results(row, swe_adapter.wrap_test_output(run.stdout or ""))

    f2p = sum(1 for name in data.fail_to_pass if results.get(name) == "PASSED")
    p2p = sum(1 for name in data.pass_to_pass if results.get(name) == "PASSED")
    resolved = f2p == len(data.fail_to_pass) and p2p == len(data.pass_to_pass)
    return Report(1.0 if resolved else 0.0, applied, f2p, p2p, results)
