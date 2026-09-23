"""Score a patch inside a box the agent never touched.

Order matters: the agent's patch is applied first, then grading restores the
paths its chosen strategy names to their `base_commit` state (see
`_restore_strategy_for`, which picks between `_restore_from_test_patch` --
SWE-bench Verified, reapplies the instance's own `test_patch` -- and
`_restore_from_pristine_image` -- SWE-smith, which never touches tests at
all, so there is nothing to reapply, only fail-to-pass/pass-to-pass files to
check out), then tests run. What that actually guarantees for the
`test_patch` path: the specific paths a strategy restores -- `test_patch`'s
own touched files; the `conftest.py`
hierarchy above them, and, at every level of that same walk, pytest's own
five config filenames (`pytest.ini`, `.pytest.ini`, `pyproject.toml`,
`tox.ini`, `setup.cfg` -- pytest's own `locate_config` searches a file's
ancestor directories innermost-first, so a nested one would otherwise win
over a restored root copy rather than being shadowed by it); the test
runner's own entry-point script; and the repo-relative paths its own
arguments name -- end the run exactly as they are at `base_commit`,
regardless of what the patch under test did to them first. That list is not
closed by construction: it is whatever `_test_infrastructure_paths`
currently enumerates, and each of its entries (`swe_adapter.test_entrypoint`,
`swe_adapter.test_command_argument_paths`, `swe_adapter.pytest_reporting_fixup`)
was added after a real, distinct bypass was found running against it, not
derived from a general survey of every way a test command can be told what
to read.

What it does NOT guarantee, and cannot: a patch confined entirely to
*source* files -- not a test, not a conftest.py, not any of the config files
above, a file the fix under test legitimately needs to keep -- can still
forge a result, for example by reassigning
`_pytest.reports.TestReport.from_item_and_call` at import time so every
report claims PASSED regardless of what actually ran. No restoration scheme
can defend this specific shape, because restoring source is the one thing
the grader must never do -- it is the fix under test. Closing that vector
needs a check that does not depend on file identity at all (an injected
canary test, at a path the agent cannot predict, that must FAIL and one that
must PASS) -- deliberately out of scope here; see task-4-report.md for where
it would land.

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

from reliquary_swe import swe_adapter, swesmith_adapter
from reliquary_swe.corpus import SweRow
from reliquary_swe.taskset import SweData

_DIFF_TARGET = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)

# Every filename pytest's own `locate_config` treats as a config file
# (`_pytest.config.findpaths.locate_config`'s `config_names`, checked against
# the installed pytest), which a patch confined to source could still use --
# at ANY directory level, not only the root, since `locate_config` walks
# `(argpath, *argpath.parents)` innermost-first -- to redirect what "the
# tests" run: pytest reads `addopts`/plugin registration from any of these,
# and tox's own `commands` (read even under `--current-env`, which only skips
# venv creation) from `tox.ini`. None is named by `test_patch`, `conftest.py`,
# or FAIL_TO_PASS/PASS_TO_PASS.
_TEST_CONFIG_FILES = (
    "pytest.ini",
    ".pytest.ini",
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
)


@dataclass(frozen=True, slots=True)
class Report:
    reward: float
    applied: bool
    # Only meaningful when `applied` is True: whether every path the chosen
    # restoration strategy touched (see `_restore_from_test_patch`) ended up
    # confirmed at its `base_commit` content. False means a reward of 0 here
    # may be OUR failure, not the agent's -- a grader-side restoration
    # problem reading identically to "did not fix the bug" is exactly the
    # silent-zero shape this module exists to avoid. When the patch itself
    # never applied, restoration was never attempted and this is set True by
    # convention: that zero already has a complete, unambiguous explanation.
    restored: bool
    fail_to_pass_passed: int
    pass_to_pass_passed: int
    results: dict[str, str] = field(default_factory=dict)
    # Monitoring fields, not reward inputs -- IMPORTANT 3. All nine defects
    # found while building this package shared one shape: the test command
    # ran but produced nothing `parse_results` could read, and `results == {}`
    # scored identically to "the tests ran and genuinely failed" -- silent by
    # construction, since neither a syntax-broken patch nor a real adapter
    # break look different from here. These two fields don't fix that (a
    # patch with a syntax error legitimately produces no parseable output,
    # so this cannot become an unconditional raise); they make it visible.
    # `results_parsed == 0` is per-instance and inherently ambiguous;
    # `pass_to_pass_passed == 0` aggregated across a whole batch is the
    # actual monitoring canary, because a real adapter break zeroes p2p for
    # every instance in the batch at once, while a patch-quality zero does
    # not -- see `grade`'s own docstring for the four defects that shape
    # would have caught in one batch instead of four separate nights.
    results_parsed: int = 0
    test_command_exit_code: int | None = None


def _paths_touched_by(patch: str) -> list[str]:
    """The files a unified diff writes to (the `+++ b/...` side of each hunk).

    Includes paths the diff *adds* (no path exists at any earlier commit for
    those) alongside paths it modifies or deletes (which do). Restoring
    "every path this names" from `base_commit` is therefore only correct for
    the ones that existed there -- see `_restore_from_test_patch`, which
    never uses this list directly for that reason and instead splits it
    against `swe_adapter.get_modified_files`.
    """
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


RestoreTests = Callable[[vf.Runtime, SweData], Awaitable[bool]]


async def _checkout(runtime: vf.Runtime, base_commit: str, path: str) -> bool:
    """Restore one path to its `base_commit` content -- one path per call,
    never batched into a single `git checkout base -- <path1> <path2> ...`.

    A batched checkout fails, and restores NONE of its paths, the instant a
    single pathspec does not match -- confirmed by hand under git 2.48.1.
    Counted against the cached corpus: 16/500 instances have a `test_patch`
    that adds a file, and 13 of those also carry an *existing* test file in
    the same patch (astropy-7336, django-11141/11749/13516/15525/16256/16454,
    pylint-6528, sphinx-10614/10673/11510/8269/8548 -- not sphinx-8595, which
    is add-only, both of its paths new: see IMPORTANT 6 in task-4-report.md)
    -- on every one of those 13, restoring the existing file this way would
    have silently restored nothing at all, because the added path's missing
    pathspec poisoned the whole command. One path per call means one bad
    path fails only itself.
    """
    result = await runtime.run(["git", "checkout", base_commit, "--", path], {})
    return result.exit_code == 0


async def _remove(runtime: vf.Runtime, path: str) -> bool:
    result = await runtime.run(["rm", "-rf", path], {})
    return result.exit_code == 0


async def _restore_if_present(runtime: vf.Runtime, base_commit: str, path: str) -> bool:
    """Make `path` exactly what it is at `base_commit`, for a path nothing
    already says the existence of (unlike `test_patch`'s own touched paths,
    which `swe_adapter.get_modified_files` classifies from the diff text
    itself -- this is for the paths in `_test_infrastructure_paths`, which
    `test_patch` never names at all).

    Existence is tested directly (`git cat-file -e {base}:{path}`) rather
    than inferred from a checkout's exit code. A checkout can fail for
    reasons that have nothing to do with absence -- an index lock, a
    permissions error, a directory sitting where a file belongs -- and
    reading any of those as "the agent added this" answers a genuine
    failure with deletion: `rm -f` on a repository's real, working
    `conftest.py` breaks collection and scores 0, silently, for a reason
    that has nothing to do with the patch under test. Only a *confirmed*
    absence at base gets removed.
    """
    exists = await runtime.run(["git", "cat-file", "-e", f"{base_commit}:{path}"], {})
    if exists.exit_code == 0:
        return await _checkout(runtime, base_commit, path)
    return await _remove(runtime, path)


def _conftest_ancestors(paths: list[str]) -> list[str]:
    """Every `conftest.py` -- and every one of `_TEST_CONFIG_FILES` -- pytest
    could load on the way to `paths`.

    pytest auto-loads a `conftest.py` from every directory between a
    collected file and the repository root, not only ones a path explicitly
    names, and its own `locate_config` searches that identical walk for its
    five config filenames innermost-first. Restoring only `test_patch`'s own
    touched paths -- or only the repo root -- leaves both walks uncovered.
    Confirmed by hand, both vectors: a `conftest.py` added at the repo root,
    nowhere near `test_patch`'s own path list, with a
    `pytest_runtest_makereport` hook forcing every outcome to "passed" earned
    full reward on an otherwise-untouched checkout; and, before this walk
    also carried the five config names, an added
    `astropy/modeling/tests/pytest.ini` with `addopts = -p no:terminal`
    silently dropped every PASSED/FAILED line -- our own explicit `-rA`
    notwithstanding, since a disabled terminal reporter has nothing left to
    apply `-rA` to -- turning the *gold* patch's correct 1.0 into a wrong
    0.0 (see `test_a_nested_pytest_ini_does_not_survive_into_grading` in
    test_goldens.py). A root-only `_TEST_CONFIG_FILES` copy never reaches
    this file at all, because `locate_config`'s innermost-first search finds
    the nested one first regardless.

    Every name is emitted at every level regardless of whether anything is
    actually there to restore -- existence is decided per path by
    `_restore_if_present`, not here. Measured against the full 500-instance
    corpus before choosing all five names over a narrower one (just
    `pytest.ini`/`.pytest.ini`, which would also close the vector): walking
    all five at every level clobbers the exact same one gold patch the
    root-only scheme already did (pylint-4661's own `setup.cfg`), zero
    additional -- so closing the vector completely costs nothing beyond what
    shipping already accepted.
    """
    seen: dict[str, None] = {}
    for path in paths:
        directory = PurePosixPath(path).parent
        while True:
            for name in ("conftest.py", *_TEST_CONFIG_FILES):
                seen[str(directory / name)] = None
            if str(directory) == ".":
                break
            directory = directory.parent
    return list(seen)


def _test_infrastructure_paths(data: SweData) -> list[str]:
    """Every path -- beyond `test_patch`'s own touched files -- that
    `swe_adapter.test_command` depends on and that a patch confined entirely
    to source could still use to control what "the tests" report: the test
    runner's own entry-point script (django's `./tests/runtests.py`,
    sympy's `bin/test` -- see `swe_adapter.test_entrypoint`), a repo-relative
    path or module label the command's own *arguments* name (django's
    `--settings=test_sqlite` -- see `swe_adapter.test_command_argument_paths`),
    pytest/tox configuration -- at the repo root unconditionally, and at
    every directory `_conftest_ancestors` walks above every path
    `test_patch` touches -- alongside the `conftest.py` hierarchy there.

    On 350/500 corpus instances (231 django, 75 sympy, 44 sphinx),
    `test_command` executes a file or reads a config the diff carries and
    `test_patch` never names -- rewriting django's `tests/runtests.py` to
    print a fake "... ok" line for every test is a *total* bypass, and an
    easier one than the conftest.py vector above: it needs no knowledge of
    pytest internals, `parse_log_django` keys on exactly that string, and
    nothing about it looks like tampering with a test file. The identical
    bypass also works through django's settings module
    (`tests/test_sqlite.py`), which `runtests.py` imports before running
    anything -- a rewritten settings module can print the same fake lines
    and exit at import time, never reaching `runtests.py` at all.
    """
    touched = _paths_touched_by(data.test_patch)
    paths = list(_TEST_CONFIG_FILES)
    paths.extend(_conftest_ancestors(touched))
    row = _row_for(data)
    entrypoint = swe_adapter.test_entrypoint(row)
    if entrypoint:
        paths.append(entrypoint)
    paths.extend(swe_adapter.test_command_argument_paths(row))
    return paths


async def _restore_from_test_patch(runtime: vf.Runtime, data: SweData) -> bool:
    """SWE-bench Verified's strategy: put back, at `base_commit`, every path
    a patch under test could use to control what "the tests" report, then
    reapply `test_patch`. Returns whether every one of those paths ended up
    confirmed restored (see `Report.restored`).

    `test_patch`'s own touched paths are split by `swe_adapter.get_modified_files`
    (upstream's own diff-text classifier: everything whose pre-patch, "a/"
    side is not `/dev/null`) into ones that existed at `base_commit` --
    checked out there, one at a time (`_checkout`) -- and ones the patch
    itself *adds*, which do not exist at any earlier commit and are removed
    instead (`_remove`). Reading paths off `test_patch`'s "+++ b/" side
    directly, as an earlier version of this function did, cannot make that
    distinction and either tries to check out a path that was never there
    (see `_checkout`'s docstring for what that failure used to do to the
    *other*, real paths in the same call) or -- if made robust to that by
    batching per-path instead -- still has no reason to prefer `rm` over
    `checkout` for an added path without asking the diff which one it is.

    Everything `_test_infrastructure_paths` names is restored the same way
    the conftest.py vector already was: existence is unknown a priori for
    all of them, so `_restore_if_present` decides per path rather than
    guessing from a checkout's exit code (see its own docstring).

    Finally applies `swe_adapter.pytest_reporting_fixup` (a no-op for every
    repo but sphinx today): a restored `tox.ini` is still a `tox.ini` that
    cannot produce parseable output on its own, and that gap is upstream of
    restoration, not a case restoration itself needs to know about.
    """
    existing = swe_adapter.get_modified_files(data.test_patch)
    touched = set(_paths_touched_by(data.test_patch))
    added = touched - set(existing)

    ok = True
    for path in existing:
        if not await _checkout(runtime, data.base_commit, path):
            ok = False
    for path in added:
        if not await _remove(runtime, path):
            ok = False
    for path in _test_infrastructure_paths(data):
        if not await _restore_if_present(runtime, data.base_commit, path):
            ok = False

    fixup = swe_adapter.pytest_reporting_fixup(_row_for(data))
    if fixup is not None:
        result = await runtime.run(fixup, {})
        if result.exit_code != 0:
            ok = False

    if data.test_patch.strip():
        await runtime.write("/tmp/tests.diff", data.test_patch.encode())
        result = await runtime.run(["git", "apply", "-v", "/tmp/tests.diff"], {})
        if result.exit_code != 0:
            ok = False
    return ok


async def _restore_from_pristine_image(runtime: vf.Runtime, data: SweData) -> bool:
    """SWE-smith's strategy: there is no `test_patch` to reapply because
    SWE-smith never touches tests at all -- its bug is injected purely into
    source (spec section 8's correction). The property restoration exists
    for -- "make the tests be what they should be, regardless of what the
    agent did" -- still has to hold, and it is reached the same way
    SWE-smith's own evaluation harness reaches it
    (`swesmith.harness.utils.run_patch_in_container`'s own anti-tamper step,
    `git checkout -- {f2p+p2p files}`): every fail-to-pass and pass-to-pass
    test lives in a file that is pristine at `data.base_commit` by
    construction (the bug commit never touched a test file), so restoring
    is a plain per-file checkout against it, no patch involved.

    Also walks the same `conftest.py`-and-pytest-config ancestry
    `_conftest_ancestors` already walks for the SWE-bench Verified path:
    that vector (a source-confined patch adding a `conftest.py` or
    `pytest.ini` to redirect what "the tests" report) is not specific to
    either corpus's restoration data, it is specific to pytest, so closing
    it once, generically, for both paths costs nothing extra here.
    """
    row = _row_for(data)
    files = swesmith_adapter.test_files(row)
    ok = True
    for path in files:
        if not await _checkout(runtime, data.base_commit, path):
            ok = False
    for path in _conftest_ancestors(files):
        if not await _restore_if_present(runtime, data.base_commit, path):
            ok = False
    return ok


def _restore_strategy_for(data: SweData) -> RestoreTests:
    """Choose how to make the tests be what they should be -- from the data
    itself, never a step `grade()` hardcodes.

    A row with a `test_patch` (SWE-bench Verified) reapplies it. A row
    without one (SWE-smith, which never touches tests -- see
    `_restore_from_pristine_image`'s own docstring) restores its
    fail-to-pass/pass-to-pass test files from the pristine image instead.
    Both answer the same question; `grade()` never branches on which corpus
    it is looking at, only this one function does.
    """
    if data.test_patch.strip():
        return _restore_from_test_patch
    return _restore_from_pristine_image


async def grade(runtime: vf.Runtime, data: SweData, patch: str) -> Report:
    """Apply `patch` in `runtime`, restore the tests, run them, and score.

    `runtime` must be a freshly provisioned box from this instance's image.
    Passing the agent's own box would defeat the entire point.

    Reward is binary: every fail-to-pass and every pass-to-pass entry must
    pass, or it's zero. A fractional reward would pay for a half-repair, and a
    half-repair is not a repair.

    All nine implementation defects recorded in this project's progress log
    presented as exactly this function returning an all-`0` report with an
    empty `results` map, each caught by a human noticing a zero on real data,
    one at a time, on separate nights -- conda not being on `PATH`, the 128 KiB
    argv cap, django's runner writing to stderr, sphinx's own invocation
    missing `-rA`, among others. `Report.results_parsed` and
    `.test_command_exit_code` exist so the same shape shows up as a
    monitoring signal instead: `pass_to_pass_passed == 0` aggregated across a
    whole batch, not any single instance's zero, is what would have caught
    four of those nine (the adapter-level ones) in one batch rather than four
    separate nights, because a real adapter break zeroes p2p for every
    instance at once and a patch that is merely bad does not.
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
            # a real outcome, not an infrastructure failure. `restored=True`
            # by convention: see Report.restored.
            return Report(0.0, False, True, 0, 0, {})

    # Deliberately not raised on failure: an agent can provoke a restoration
    # or test_patch `git apply` failure on purpose (e.g. squatting a file at
    # a path `test_patch` adds), and raising would turn that into an infra
    # error attributed to us rather than a result attributed to the patch.
    # `restored` carries the signal instead -- see Report.restored.
    restored = await _restore_strategy_for(data)(runtime, data)

    # A canary check -- one injected test that must FAIL and one that must
    # PASS, at a path the agent cannot predict -- would slot in here, after
    # restoration and before the real test run, and is the only defense that
    # would touch the source-monkeypatch vector this module's own docstring
    # names. Deliberately not built now; see task-4-report.md.
    row = _row_for(data)
    # `split` is already on the wire and is literally "which corpus is
    # this" -- the honest discriminator for this decision, rather than
    # `version` (empty for every SWE-smith row, but a proxy for the real
    # question, not the question itself).
    if data.split == "train":
        run = await runtime.run(swesmith_adapter.test_command(row), {})
        results = swesmith_adapter.parse_results(row, run.stdout or "")
    else:
        run = await runtime.run(swe_adapter.test_command(row), {})
        results = swe_adapter.parse_results(
            row, swe_adapter.wrap_test_output(run.stdout or "")
        )

    f2p = sum(1 for name in data.fail_to_pass if results.get(name) == "PASSED")
    p2p = sum(1 for name in data.pass_to_pass if results.get(name) == "PASSED")
    resolved = f2p == len(data.fail_to_pass) and p2p == len(data.pass_to_pass)
    return Report(
        1.0 if resolved else 0.0,
        applied,
        restored,
        f2p,
        p2p,
        results,
        results_parsed=len(results),
        test_command_exit_code=run.exit_code,
    )
