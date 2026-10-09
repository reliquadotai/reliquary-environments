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

`grade` runs in two halves, `prepare_box` (the pristine box, before the patch
is applied: anything that fails there is about us or the image,
`PristineBoxError`) and `grade_prepared` (the patch). A signed-episode
sandbox runs `prepare_box` as the task's `grading_setup`, before the agent's
artifacts reach the box, and `grade_prepared` from the task's reward; `grade`
runs both, as before.

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

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import verifiers.v1 as vf

from reliquary_swe import swe_adapter, swesmith_adapter
from reliquary_swe.corpus import SweRow
from reliquary_swe.taskset import (
    _R2E_CHECKOUT,
    _R2E_LEAK_GUARD,
    _STRIP_AND_GC,
    _TRAIN_GUARD_AND_REROOT,
    SweData,
)

_FULL_SHA = re.compile(r"[0-9a-f]{40}")
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


# Polyglot only. The files each ecosystem's test runner reads before it runs
# a single test, and that a patch confined to source could rewrite to make
# `mimo_test_command.sh` exit 0 without the hidden tests passing: a `test`
# script or `testPathIgnorePatterns` in `package.json`, a jest/vitest/mocha
# config, a `replace` directive in `go.mod`, a Maven surefire `skip`, a
# `Rakefile` or `Makefile` target. The polyglot verdict is an exit status,
# not a parsed per-test report, so anything that controls what the runner
# does is as good as a forged result.
#
# Restoring these can also revert a legitimate edit -- a fix that really
# needs a new `exports` entry in `package.json` scores 0. Accepted for the
# same reason `_TEST_CONFIG_FILES` accepted clobbering pylint-4661's own
# `setup.cfg`: a false 0 on a rare fix is recoverable, a forgeable 1 is not.
_POLYGLOT_RUNNER_FILES = (
    # JavaScript / TypeScript
    "package.json",
    *(f"jest.config.{ext}" for ext in ("js", "cjs", "mjs", "ts", "json")),
    *(f"vitest.config.{ext}" for ext in ("js", "cjs", "mjs", "ts", "mts")),
    *(f"vite.config.{ext}" for ext in ("js", "mjs", "ts")),
    *(f".mocharc.{ext}" for ext in ("js", "cjs", "json", "yml", "yaml")),
    *(f"babel.config.{ext}" for ext in ("js", "cjs", "json")),
    ".babelrc",
    "tsconfig.json",
    # Go
    "go.mod",
    "go.sum",
    "go.work",
    # Rust
    "Cargo.toml",
    ".cargo/config.toml",
    # Java
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    # Ruby
    "Gemfile",
    ".rspec",
    "Rakefile",
    # PHP
    "composer.json",
    "phpunit.xml",
    "phpunit.xml.dist",
    # Any
    "Makefile",
)


# R2E only. Where a grading box keeps the image's hidden tests while the
# agent's patch is applied: outside the repository, so `git apply` can neither
# write into them nor refuse a patch over them (a patch that adds its own
# `run_tests.sh` would otherwise not apply onto the image's untracked one).
# R2E's own runtime uses `/root` for the same purpose.
_R2E_STASH = "/r2e_grading"

# R2E only: the deadline on `run_tests.sh`, R2E's own default
# (`DockerRuntime._calculate_reward_r2e(timeout=300)`). Measured on the three
# goldens' images: 0-3 s per run. A run cut short prints no complete summary
# and scores 0, which is the agent's result (a patch that hangs the tests),
# not ours.
_R2E_TEST_TIMEOUT_SECONDS = 300

# R2E's `DockerRuntime.run_tests` strips this from the test output before
# parsing, and `execution_log_parser.decolor_dict_keys` this narrower one
# from the expected map's keys. 369 rows of the pinned revision store
# expected names wrapped in ANSI bold, so both matter.
_ANSI_OUTPUT = re.compile(r"\x1b\[[0-9;]*m|\r")
_ANSI_KEY = re.compile(r"\x1b\[\d+m")

# The diff header lines that carry a file mode, as `git apply` reads them
# (apply.c's `gitdiff_*` handlers; `index <a>..<b>` carries one after a further
# space).
_MODE_HEADERS = (b"old mode ", b"new mode ", b"deleted file mode ", b"new file mode ")
# `strtoul(text, &end, 8)`'s reading: C blanks (a newline included), a sign, octal digits.
_STRTOUL_OCTAL = re.compile(rb"[ \t\n\v\f\r]*([+-]?)([0-7]*)")
_S_IFMT, _S_IFREG, _S_IFLNK = 0o170000, 0o100000, 0o120000


def _git_file_type(raw: bytes, pos: int) -> int | None:
    """The file type git gives the mode written at `raw[pos:]`: `strtoul(.., 8)`
    (leading blanks -- a newline too, so the digits may sit on the next line --
    a sign, any number of zeros; saturating at ULONG_MAX), kept in an
    `unsigned int`. None where git reads no digits and refuses the patch."""
    match = _STRTOUL_OCTAL.match(raw, pos)
    if not match.group(2):
        return None
    value = int(match.group(2), 8)
    if value >= 1 << 64:
        value = (1 << 64) - 1
    elif match.group(1) == b"-":
        value = -value % (1 << 64)
    return value & 0xFFFFFFFF & _S_IFMT


def patch_shape_violations(raw: bytes) -> list[str]:
    """What the env norm refuses in an agent's patch, read from its header
    lines only (a hunk's content lines start with ' ', '+' or '-', so they
    never match): a symlink (file type 120000), a submodule, a binary hunk.

    Modes are read as git reads them (`_git_file_type`), and only a regular
    file passes: git's `canon_mode` turns any type it does not know into a
    gitlink, so everything that is neither a regular file nor a link counts
    as a submodule. Every line of the patch is read as a possible header: a
    false refusal costs a malformed patch its 0, a missed one would let a link
    into the box the tests run in. A binary hunk is refused in both of git's
    spellings (`GIT binary patch`; `Binary files ... differ`, which git applies
    from the object store when the header carries full blob ids).

    The capture runs in the agent's box with git configuration the agent
    controls, so this is checked here, in the grading box, never trusted
    from the capture."""
    found: set[str] = set()
    pos = 0
    while pos < len(raw):
        newline = raw.find(b"\n", pos)
        end = len(raw) if newline < 0 else newline
        line = raw[pos:end]
        mode_at = None
        for header in _MODE_HEADERS:
            if line.startswith(header):
                mode_at = pos + len(header)
                break
        if mode_at is None and line.startswith(b"index "):
            # gitdiff_index: the first '.' must open '..'; a mode follows the
            # first space after it.
            dot = line.find(b".")
            space = line.find(b" ", dot + 2) if dot >= 0 and line[dot + 1:dot + 2] == b"." else -1
            if space >= 0:
                mode_at = pos + space + 1
        if mode_at is not None:
            kind = _git_file_type(raw, mode_at)
            if kind == _S_IFLNK:
                found.add("symlink")
            elif kind is not None and kind != _S_IFREG:
                found.add("submodule")
        if line.startswith(b"GIT binary patch") or (
            line.startswith((b"Binary files ", b"Files ")) and b" differ" in line
        ):
            found.add("binary")
        pos = end + 1
    return sorted(found)


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
    # Polyglot and R2E only: the only evidence of *why* a run failed -- a
    # real failing assertion, or a runner that never started -- is the end
    # of its output (for polyglot the verdict is an exit status; for R2E a
    # verdict map that may legitimately contain FAILED and ERROR entries).
    test_output_tail: str = ""


class PristineBoxError(RuntimeError):
    """A grading box that is not what the corpus says, found before the agent's patch
    touches it (checking out the base, stripping history, setting hidden tests aside,
    listing the untracked paths at base): an error about us or the image, never a 0.
    A RuntimeError like before; on a signed-episode sandbox `sandbox.grade` reports it as
    the sandbox's `EnvInfraError` (the episode is aborted, not graded)."""


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
    return _ancestor_paths(paths, ("conftest.py", *_TEST_CONFIG_FILES))


def _ancestor_paths(paths: list[str], names: tuple[str, ...]) -> list[str]:
    """Every one of `names`, at every directory from each of `paths`' own
    up to the repository root -- the walk `_conftest_ancestors` documents,
    over any set of file names."""
    seen: dict[str, None] = {}
    for path in paths:
        directory = PurePosixPath(path).parent
        while True:
            for name in names:
                seen[str(directory / name)] = None
            if str(directory) == ".":
                break
            directory = directory.parent
    return list(seen)


def _polyglot_infrastructure_paths(test_patch: str) -> list[str]:
    """Every runner-config path, of every ecosystem, at the repository root
    and above every file `test_patch` touches. Language-blind on purpose:
    a row carries no language, and restoring a `go.mod` in a JavaScript
    repository is a no-op, not a risk."""
    names = ("conftest.py", *_TEST_CONFIG_FILES, *_POLYGLOT_RUNNER_FILES)
    return _ancestor_paths(["_", *_paths_touched_by(test_patch)], names)


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


async def _restore_all_if_present(
    runtime: vf.Runtime, base_commit: str, paths: list[str]
) -> bool:
    """`_restore_if_present` over many paths in ONE command.

    The polyglot walk names ~50 files per directory level; at two runtime
    round trips per path that is hundreds of calls per grading, so the same
    per-path logic -- confirmed existence at base, then checkout, else
    remove -- runs as one shell loop instead. One path per `git checkout`
    still, for the reason `_checkout` documents.
    """
    script = (
        'ok=0; while IFS= read -r p; do '
        'if git cat-file -e "$BASE:$p" 2>/dev/null; then '
        'git checkout -q "$BASE" -- "$p" || ok=1; '
        'else rm -rf -- "$p" || ok=1; fi; '
        'done < /tmp/restore-paths; exit $ok'
    )
    await runtime.write("/tmp/restore-paths", ("\n".join(paths) + "\n").encode())
    result = await runtime.run(["sh", "-c", script], {"BASE": base_commit})
    return result.exit_code == 0


async def _restore_polyglot(runtime: vf.Runtime, data: SweData) -> bool:
    """The polyglot strategy: `_restore_from_test_patch`'s shape, minus
    everything that asks SWE-bench's per-repository registry a question.

    `test_patch`'s own paths are checked out at base when they existed there
    and removed when the patch adds them (the hidden test files, and
    `mimo_test_command.sh` itself), every runner config
    `_polyglot_infrastructure_paths` names is put back, and `test_patch` is
    reapplied. The agent never saw the hidden tests, but it may have guessed
    a path or rewritten a runner; neither survives this.
    """
    existing = swe_adapter.get_modified_files(data.test_patch)
    added = [p for p in _paths_touched_by(data.test_patch) if p not in set(existing)]
    ok = True
    for path in existing:
        if not await _checkout(runtime, data.base_commit, path):
            ok = False
    for path in added:
        if not await _remove(runtime, path):
            ok = False
    if not await _restore_all_if_present(
        runtime, data.base_commit, _polyglot_infrastructure_paths(data.test_patch)
    ):
        ok = False
    await runtime.write("/tmp/tests.diff", data.test_patch.encode())
    result = await runtime.run(["git", "apply", "-v", "/tmp/tests.diff"], {})
    if result.exit_code != 0:
        ok = False
    return ok


async def _restore_r2e(runtime: vf.Runtime, data: SweData) -> bool:
    """The R2E strategy: put the image's own hidden tests back where R2E's
    runner expects them, whatever the agent's patch did.

    `prepare_box()` moved `/r2e_tests` and `run_tests.sh` into `_R2E_STASH` of
    this fresh box before the patch was applied, so they are the pristine
    image's -- the `_restore_from_pristine_image` idea, with the image's
    file system as the source instead of a git commit, because neither is
    tracked by git (measured: `run_tests.sh` is untracked, `/r2e_tests` is
    outside the repository). Then, as R2E's own runtime does
    (`DockerRuntime.setup_env`): `r2e_tests` reachable as a symlink at
    `$WORKDIR/r2e_tests`, and `run_tests.sh` at `$WORKDIR/run_tests.sh` --
    replacing anything the patch put at either path.

    The repository root is where the patch can steer the run itself, since
    `run_tests.sh` is `python -m pytest ... r2e_tests` started there: `-m`
    puts the working directory first on `sys.path`, and pytest loads a root
    `conftest.py`. Both measured on the coveragepy golden, without the fix:
    a root `conftest.py` forcing every outcome to "passed" made all 7
    expected tests report PASSED (that task's exact expected map, so a
    paid 1.0), and a root `pytest.py` replaced pytest outright and printed
    whatever summary it liked. So every root entry that was not there before
    the patch (`_R2E_STASH/root-entries`, listed by `prepare_box()`) is deleted --
    shadowing `pytest`, `_pytest`, `pluggy` or any other module pytest
    imports after start-up takes a new root entry -- and `conftest.py` and
    pytest's five config files (`_TEST_CONFIG_FILES`) that the image does
    track are put back to the base, with the root `.gitignore`. The cost: a
    patch's new root-level files never reach the test run; an agent's
    `reproduce_issue.py` is collateral, a fix that needs a new top-level
    module is not graded as one (none of the five goldens' does). Nothing below `r2e_tests` needs
    any of this: that whole tree is the image's.
    """
    script = " && ".join(
        [
            f"test -s {_R2E_STASH}/root-entries",
            f'ls -A "$WORKDIR" | grep -vxF -f {_R2E_STASH}/root-entries'
            ' | while IFS= read -r p; do rm -rf -- "$WORKDIR/$p"; done',
            'rm -rf "$WORKDIR"/r2e_tests "$WORKDIR"/run_tests.sh',
            f'cp {_R2E_STASH}/run_tests.sh "$WORKDIR"/run_tests.sh',
            f'ln -s {_R2E_STASH}/r2e_tests "$WORKDIR"/r2e_tests',
        ]
    )
    placed = await runtime.run(["sh", "-c", script], {"WORKDIR": data.workdir})
    ok = placed.exit_code == 0
    if not await _restore_all_if_present(
        runtime, data.base_commit, ["conftest.py", ".gitignore", *_TEST_CONFIG_FILES]
    ):
        ok = False
    return ok


def _numstat_paths(numstat_z: str) -> list[str]:
    """Every path a patch touches, read from `git apply --numstat -z` --
    git's own parse, so a quoted or escaped path in the diff text cannot
    slip past a regex. Each record is `added\tdeleted\tpath`, or for a
    rename `added\tdeleted\t` followed by the old and the new path as two
    further NUL-separated fields."""
    fields = numstat_z.split("\0")
    paths: list[str] = []
    i = 0
    while i < len(fields):
        parts = fields[i].split("\t", 2)
        if len(parts) == 3 and parts[2]:
            paths.append(parts[2])
            i += 1
        elif len(parts) == 3:
            paths.extend(fields[i + 1 : i + 3])
            i += 3
        else:
            i += 1
    return paths


def _forbidden_patch_paths(
    paths: list[str], untracked: list[str], ignored: list[str]
) -> list[str]:
    """The patch paths grading refuses outright, for every split: any under
    `.venv/`, any that is -- or lies inside -- a path present but untracked
    at the base (`git ls-files --others --directory`, which lists ignored
    paths too and collapses a wholly untracked directory to `dir/`), and any
    git ignores at the base.

    Why: restoration only puts back tracked paths and the tests' own files,
    and never reaches paths the image already has outside the tracked tree --
    exactly where a patch can steer the run without touching a test. R2E's
    `run_tests.sh` runs `.venv/bin/python`, so a
    `.venv/lib/python3.X/site-packages/zz.pth` executes at interpreter
    start-up, and `python -m pytest` reads plugin entry points from the
    `<project>.egg-info` the root holds; a polyglot image's ignored
    `node_modules/.bin/jest` is the runner its test command calls, and one
    hunk can make it `exit 0`.

    The patch is the agent's to shape, not a faithful `git add -A`: the
    capture runs in the agent's box, against a base ref and with a git
    config the agent can rewrite (`diff.external`, a moved
    `refs/reliquary/base`, a `!` rule in `.gitignore` or
    `.git/info/exclude`), so any path can appear in it. The ignore check
    runs against the base before the patch is applied, so a `.gitignore` the
    patch edits has no say in it. An honest capture never produces such a
    path: it leaves out what was untracked at setup, and `git add -A` skips
    ignored files.

    `ignored` comes from `git check-ignore` WITH the index, not
    `--no-index`: a tracked file that happens to match an ignore pattern is
    the repository's own source, and a fix may need to edit it.
    """
    untracked_dirs = tuple(entry for entry in untracked if entry.endswith("/"))
    untracked_files = {entry for entry in untracked if not entry.endswith("/")}
    ignored_set = set(ignored)
    return [
        path
        for path in paths
        if path == ".venv"
        or path.startswith(".venv/")
        or path in untracked_files
        or path.startswith(untracked_dirs)
        or path + "/" in untracked_dirs
        or path in ignored_set
    ]


_CHECKING_PATCH = "Checking patch "
_C_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}
_OCTAL_ESCAPE = re.compile(r"[0-3][0-7]{2}")


def _unquote_c(name: str) -> str:
    """A name as git's `quote_c_style` printed it, unquoted (bytes that are not
    UTF-8 come back as U+FFFD, which `_patch_violations` refuses)."""
    if len(name) < 2 or name[0] != '"' or name[-1] != '"':
        return name
    body, out, i = name[1:-1], bytearray(), 0
    while i < len(body):
        if body[i] == "\\" and _OCTAL_ESCAPE.match(body, i + 1):
            out.append(int(body[i + 1 : i + 4], 8))
            i += 4
        elif body[i] == "\\" and body[i + 1 : i + 2] in _C_ESCAPES:
            out.append(_C_ESCAPES[body[i + 1]])
            i += 2
        else:
            out += body[i].encode()
            i += 1
    return out.decode("utf-8", "replace")


def _checked_patch_names(output: str) -> list[str]:
    """Every name git reads or writes for a patch, from the `Checking patch`
    lines of `git apply --check -v`: `old => new` for a rename, a copy, or a
    hunk whose two sides name different files (git reads the old one and
    keeps its mode), the bare name otherwise. `--numstat` names only the
    written side. An unquoted name may itself hold ` => `, so every split is
    kept: a superset, never a miss."""
    names: list[str] = []
    for line in output.split("\n"):
        if not line.startswith(_CHECKING_PATCH) or not line.endswith("..."):
            continue
        body = line[len(_CHECKING_PATCH) : -3]
        names.append(_unquote_c(body))
        at = body.find(" => ")
        while at >= 0:
            names += [_unquote_c(body[:at]), _unquote_c(body[at + 4 :])]
            at = body.find(" => ", at + 1)
    return names


# One shell over every path a patch names: a symlink in the box's worktree,
# then the index entry of each (git's own mode: 120000 link, 160000 gitlink).
_SPECIAL_PATHS_SH = (
    'for p; do if [ -L "$p" ]; then printf "symlink %s\\0" "$p"; fi; done; '
    'exec git --literal-pathspecs ls-files -s -z -- "$@"'
)


def _special_paths(output: str) -> list[str]:
    """`_SPECIAL_PATHS_SH`'s output, as refusals: every symlink and gitlink."""
    found: list[str] = []
    for record in output.split("\0"):
        if record.startswith("symlink "):
            found.append(f"{record[len('symlink '):]} (symlink)")
            continue
        meta, tab, path = record.partition("\t")
        mode = meta.split(" ", 1)[0]
        if tab and mode == "120000":
            found.append(f"{path} (symlink)")
        elif tab and mode == "160000":
            found.append(f"{path} (submodule)")
    return list(dict.fromkeys(found))


async def _patch_violations(runtime: vf.Runtime, data: SweData) -> list[str]:
    """`_forbidden_patch_paths` for the patch at /tmp/agent.diff, asked of
    this box before the patch is applied, after refusing any path it reads
    or writes that is not UTF-8, is a symlink (worktree or index) or a
    gitlink. A patch header need not say a mode: a modeless hunk on a
    tracked link, or a copy or rename from one, makes git read through the
    link and keep its mode -- `patch_shape_violations` cannot see those.
    A patch git cannot parse yields no paths here and fails to apply right
    after, scoring 0 there."""
    numstat = await runtime.run(["git", "apply", "--numstat", "-z", "/tmp/agent.diff"], {})
    if numstat.exit_code != 0:
        return []
    checked = await runtime.run(
        ["sh", "-c", "LC_ALL=C git apply --check -v /tmp/agent.diff 2>&1"], {}
    )
    paths = list(
        dict.fromkeys(
            [
                *_numstat_paths(numstat.stdout or ""),
                *_checked_patch_names((checked.stdout or "") + (checked.stderr or "")),
            ]
        )
    )
    if not paths:
        return []
    undecodable = [path for path in paths if "\ufffd" in path]
    if undecodable:
        return [f"{path} (not UTF-8)" for path in undecodable]
    special = await runtime.run(["sh", "-c", _SPECIAL_PATHS_SH, "sh", *paths], {})
    if special.exit_code != 0:
        detail = (special.stderr or special.stdout).strip()[-200:]
        return [f"{path} (mode check exit {special.exit_code}: {detail})" for path in paths]
    found = _special_paths(special.stdout or "")
    if found:
        return found
    others = await runtime.run(["git", "ls-files", "-z", "--others", "--directory"], {})
    if others.exit_code != 0:
        raise PristineBoxError(
            f"could not list untracked paths for {data.instance_id}: "
            f"{(others.stderr or others.stdout).strip()[-500:]}"
        )
    await runtime.write("/tmp/agent-paths", ("\0".join(paths) + "\0").encode())
    ignored = await runtime.run(
        ["sh", "-c", "git check-ignore -z --stdin < /tmp/agent-paths"], {}
    )
    # 0: some path is ignored, 1: none is. Anything else is git refusing a
    # path, and the patch is what named it -- a path past a tracked symlink
    # exits 128 ("beyond a symbolic link"). Raising would read as an infra
    # failure (provisioning retried, the episode dropped); every path is a
    # violation instead, scored 0.
    if ignored.exit_code not in (0, 1):
        detail = (ignored.stderr or ignored.stdout).strip()[-200:]
        return [f"{path} (git check-ignore exit {ignored.exit_code}: {detail})" for path in paths]
    return _forbidden_patch_paths(
        paths,
        [entry for entry in (others.stdout or "").split("\0") if entry],
        [entry for entry in (ignored.stdout or "").split("\0") if entry],
    )


def parse_log_pytest(log: str) -> dict[str, str]:
    """A port of R2E-Gym's `execution_log_parser.parse_log_pytest`, the parser
    R2E uses for every repository in this corpus (its `parse_log_fn` maps
    all ten to it) -- tornado included, whose own `tornado_unittest_runner.py`
    prints the same summary format (checked on the golden image).

    Only lines after "short test summary info" count, and of those only lines
    whose FIRST token is PASSED, FAILED or ERROR -- the one departure from
    upstream, which tests `"PASSED" in line` anywhere in the line, in that
    order. Upstream's check is an exploit: `FAILED r2e_tests/...::test_fix -
    RuntimeError: PASSED` parses as PASSED, so one `raise
    RuntimeError("PASSED")` on the buggy path turns a failing test into a
    pass. Both runners in this corpus start every status line with the
    status (pytest's `-rA` summary; tornado's runner prints
    `f"{outcome.upper()} {test}"`), checked on the goldens' images.

    A status line records the `::`-separated parts after the file, joined
    with "."; FAILED and ERROR names are cut at " - " (the message pytest
    appends). A line without `::` -- a collection error,
    `ERROR r2e_tests/test_1.py - ...` -- records the empty name, as upstream
    does; `r2e_reward` decides what that is worth.
    """
    if "short test summary info" not in log:
        return {}
    statuses: dict[str, str] = {}
    for line in log.split("short test summary info")[1].strip().split("\n"):
        status = line.split(" ", 1)[0]
        if status == "PASSED":
            statuses[".".join(line.split("::")[1:])] = "PASSED"
        elif status in ("FAILED", "ERROR"):
            statuses[".".join(line.split("::")[1:]).split(" - ")[0]] = status
    return statuses


def _r2e_normalised(statuses: dict[str, str]) -> dict[str, str]:
    """R2E's key normalisation, applied to both sides before comparing:
    ANSI codes stripped, then each name cut at its first " - "."""
    decolored = {_ANSI_KEY.sub("", name): status for name, status in statuses.items()}
    return {name.split(" - ")[0]: decolored[name] for name in sorted(decolored)}


def r2e_reward(parsed: dict[str, str], expected_output_json: str) -> float:
    """1.0 iff the parsed verdict map is exactly the expected one, after
    R2E's own normalisation (`_r2e_normalised`); 0.0 otherwise.

    R2E's `_calculate_reward_r2e` asks for the same size and every parsed
    name to carry its expected status. Two deliberate departures, both
    strictly harder to pay:

    - An empty parse is 0.0, never a vacuous pass -- a runner that never
      reached its summary must not read as success, whatever the expected
      map holds.
    - The empty name is compared like any other. Upstream skips it, which
      lets a collection error stand in for a missing test: on a task with
      one expected test (46 rows in the pinned revision), a patch that
      breaks the test module's import parses as `{"": "ERROR"}` -- same
      size, nothing compared, paid 1.0. The one row whose expected map
      itself holds `""` (orange3 f813020a9c0a) still pays when it is
      reproduced.

    Equal maps are equal sizes with every name matching, so for every input
    without those two shapes this is R2E's own verdict.
    """
    if not parsed:
        return 0.0
    expected = _r2e_normalised(json.loads(expected_output_json))
    return 1.0 if _r2e_normalised(parsed) == expected else 0.0


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
    if data.split == "polyglot":
        return _restore_polyglot
    if data.split == "r2e":
        return _restore_r2e
    if data.test_patch.strip():
        return _restore_from_test_patch
    return _restore_from_pristine_image


async def grade(runtime: vf.Runtime, data: SweData, patch: str | bytes) -> Report:
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
    return await grade_prepared(runtime, await prepare_box(runtime, data), patch)


async def prepare_box(runtime: vf.Runtime, data: SweData) -> SweData:
    """The pristine-box half of `grade`: everything before the agent's patch is
    written (raises `PristineBoxError`). Returns `data` with `base_commit`
    resolved where the strip changed what it names."""
    # `--detach` for the reason `taskset._CHECKOUT` documents: a polyglot
    # `base_commit` is the symbolic `HEAD`, which the strip below would
    # otherwise leave pointing at a deleted branch.
    checkout = await runtime.run(
        ["git", "checkout", "-q", "--detach", data.base_commit], {}
    )
    if checkout.exit_code != 0:
        # Every image ships its repo already at base_commit; failing to reach
        # it back is not a real "0 result", it's a box that isn't what it
        # claims to be -- raising here is what keeps that from reading as a
        # silent, wrong zero.
        raise PristineBoxError(
            f"could not check out {data.base_commit} for {data.instance_id}: "
            f"{(checkout.stderr or checkout.stdout).strip()[-500:]}"
        )

    if data.split == "train":
        # This box is freshly provisioned from the image and never put
        # through `SweTask.setup()`'s cleanup, so it still has the same
        # two-commit history Critical 1 closed there -- the fix one parent
        # away from HEAD, reachable with `git diff HEAD HEAD^`, no ref
        # needed. Closing it here matters even though the agent never sees
        # this box: its patched source *executes* during the test run
        # below, after restoration, so a patch that reads git objects at
        # import time to recover the fix would make the suite genuinely
        # pass without deriving anything -- a second way into the same
        # residual this module's own docstring names for a monkeypatched
        # `pytest`. Reuses `taskset`'s own guard-and-reroot and strip-and-gc
        # verbatim rather than a second implementation of either.
        sever = await runtime.run(
            ["sh", "-c", "set -e ; " + _TRAIN_GUARD_AND_REROOT + " ; " + _STRIP_AND_GC],
            {"BASE_COMMIT": data.base_commit, "WORKDIR": data.workdir},
        )
        if sever.exit_code != 0:
            raise PristineBoxError(
                f"could not sever pristine history for {data.instance_id}: "
                f"{(sever.stderr or sever.stdout).strip()[-500:]}"
            )
        # The ref `data.base_commit` names (e.g. "origin/<id>~1") no longer
        # exists after the strip above -- every restoration checkout from
        # here on must use the resolved SHA instead. The rebound `data` (a
        # copy, not the caller's object) is what this function returns, so
        # every restoration in `grade_prepared` reads the SHA.
        resolved = await runtime.run(["git", "rev-parse", "HEAD"], {})
        if resolved.exit_code != 0:
            # Unchecked, this was a real path to a wrongly-paid reward: a
            # falsy `(resolved.stdout or "").strip()` becomes `""`, every
            # restoration checkout below fails against that empty string,
            # `restored=False` gets recorded -- and the test run proceeds
            # anyway, on an unrestored box, exactly like the `checkout` and
            # `sever` steps either side of this one refuse to let happen.
            raise PristineBoxError(
                f"could not resolve HEAD after severing history for "
                f"{data.instance_id}: "
                f"{(resolved.stderr or resolved.stdout).strip()[-500:]}"
            )
        data = data.model_copy(update={"base_commit": resolved.stdout.strip()})
    elif data.split in ("polyglot", "r2e"):
        # A shape-one polyglot image parks the complete upstream --
        # implementation included -- under `origin/<branch>` (see
        # `SweTask.setup`). The agent's patched source executes during the
        # test run below, so it could read that out of the object store at
        # run time; strip it here exactly as setup() does for the agent's
        # box. `base_commit` is `HEAD`, which means something else once
        # the agent's patch is committed or applied, so it is pinned to the
        # resolved SHA for every restoration checkout below.
        #
        # An R2E image carries the complete upstream history, all branches,
        # the fix commit included (measured: `git cat-file -t <fix>` answers
        # "commit" on every image probed). Same cleanup, same reason --
        # after committing the image's own working tree as the base, exactly
        # as `setup()` did in the agent's box (see `taskset._R2E_CHECKOUT`):
        # the agent's patch is a diff against that state, and restoration
        # must put back the image's `setup.cfg`, not HEAD's.
        prefix = "set -e ; "
        if data.split == "r2e":
            prefix += _R2E_CHECKOUT + " ; " + _R2E_LEAK_GUARD + " ; "
        strip = await runtime.run(
            ["sh", "-c", prefix + _STRIP_AND_GC], {"WORKDIR": data.workdir}
        )
        if strip.exit_code != 0:
            raise PristineBoxError(
                f"could not strip history for {data.instance_id}: "
                f"{(strip.stderr or strip.stdout).strip()[-500:]}"
            )
        resolved = await runtime.run(["git", "rev-parse", "HEAD"], {})
        if resolved.exit_code != 0:
            raise PristineBoxError(
                f"could not resolve HEAD for {data.instance_id}: "
                f"{(resolved.stderr or resolved.stdout).strip()[-500:]}"
            )
        data = data.model_copy(update={"base_commit": resolved.stdout.strip()})
        if data.split == "r2e":
            # Out of the patch's reach before it is applied -- see
            # `_R2E_STASH`. Missing hidden tests mean the image is not what
            # the corpus says it is: an error about us, never a 0.
            stash = await runtime.run(
                [
                    "sh",
                    "-c",
                    f"set -e ; mkdir -p {_R2E_STASH} ; "
                    f"mv /r2e_tests {_R2E_STASH}/r2e_tests ; "
                    f'mv "$WORKDIR"/run_tests.sh {_R2E_STASH}/run_tests.sh ; '
                    f'ls -A "$WORKDIR" > {_R2E_STASH}/root-entries',
                ],
                {"WORKDIR": data.workdir},
            )
            if stash.exit_code != 0:
                raise PristineBoxError(
                    f"could not set aside the hidden tests for {data.instance_id}: "
                    f"{(stash.stderr or stash.stdout).strip()[-500:]}"
                )
    else:
        # SWE-bench Verified's own leak, closed the same way `setup()`
        # already closes it for the agent's box: the fix for this
        # instance's own bug is a *descendant* of `base_commit`, reachable
        # through whatever branch/tag ref still names it, and this
        # grading box is provisioned fresh from the image and never put
        # through that cleanup. No guard and no re-root needed here --
        # unlike a train row, `base_commit` is already a raw SHA, and a
        # descendant (unlike an ancestor) *is* pruned by `gc` once no ref
        # reaches it, so stripping refs alone is enough. Same vector as
        # the train case: the agent's patched source executes during the
        # test run below, after restoration, so a patch reading git
        # objects at import time to recover the fix would make the suite
        # genuinely pass without deriving anything -- on the one number
        # anyone actually compares against a published result.
        strip = await runtime.run(
            ["sh", "-c", "set -e ; " + _STRIP_AND_GC], {"WORKDIR": data.workdir}
        )
        if strip.exit_code != 0:
            raise PristineBoxError(
                f"could not strip history for {data.instance_id}: "
                f"{(strip.stderr or strip.stdout).strip()[-500:]}"
            )
    return data


async def prepared_data(runtime: vf.Runtime, data: SweData) -> SweData:
    """`data` as `prepare_box` returned it, re-read from a box it already prepared:
    the strip replaced what `base_commit` names by the box's HEAD for every split
    but SWE-bench Verified. Nothing the agent sent can move HEAD here (only its
    patch file reached the box, outside the repository)."""
    if data.split not in ("train", "polyglot", "r2e"):
        return data
    resolved = await runtime.run(["git", "rev-parse", "HEAD"], {})
    if resolved.exit_code != 0:
        raise PristineBoxError(
            f"could not resolve HEAD for {data.instance_id} in the prepared box: "
            f"{(resolved.stderr or resolved.stdout).strip()[-500:]}"
        )
    return data.model_copy(update={"base_commit": resolved.stdout.strip()})


async def grade_prepared(runtime: vf.Runtime, data: SweData, patch: str | bytes) -> Report:
    """`grade` once `prepare_box` ran in this box (`data` as it returned it).

    A `bytes` patch is written to the box unchanged: an honest patch of a
    non-UTF-8 source file must apply as captured. A patch whose shape the env
    norm refuses (`patch_shape_violations`) scores 0 and is never applied.
    """
    if data.split in ("train", "polyglot", "r2e") and not _FULL_SHA.fullmatch(data.base_commit):
        # Restoration checks paths out of `base_commit`: only the commit
        # `prepare_box` resolved names the box's base, never a ref.
        raise PristineBoxError(
            f"base_commit {data.base_commit!r} for {data.instance_id} is not the full "
            "commit prepare_box resolved"
        )
    raw = patch if isinstance(patch, bytes) else patch.encode()
    applied = True
    if raw.strip():
        shape = patch_shape_violations(raw)
        if shape:
            return Report(
                0.0,
                False,
                True,
                0,
                0,
                {},
                test_output_tail="patch refused (env norm: no binary, symlink or submodule): "
                + ", ".join(shape),
            )
        await runtime.write("/tmp/agent.diff", raw)
        violations = await _patch_violations(runtime, data)
        if violations:
            # Not applied, on purpose, whatever the split: see
            # `_forbidden_patch_paths`.
            return Report(
                0.0,
                False,
                True,
                0,
                0,
                {},
                test_output_tail="patch refused (env norm: no path outside the tracked tree, "
                "no symlink, submodule or non-UTF-8 name): " + ", ".join(violations)[:1900],
            )
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
    if not restored:
        # Any restoration failure leaves a box whose verdict could be the
        # patch's, on every split: score it 0 and run nothing.
        return Report(
            0.0,
            applied,
            False,
            0,
            0,
            {},
            test_output_tail="restoration failed; nothing was run",
        )

    # A canary check -- one injected test that must FAIL and one that must
    # PASS, at a path the agent cannot predict -- would slot in here, after
    # restoration and before the real test run, and is the only defense that
    # would touch the source-monkeypatch vector this module's own docstring
    # names. Deliberately not built now; see task-4-report.md.
    if data.split == "polyglot":
        # The corpus's own verdict, unchanged: `mimo_test_command.sh` exits
        # 0 or it does not. No test lists exist to count against.
        run = await runtime.run(["bash", "-c", data.test_command], {})
        return Report(
            1.0 if run.exit_code == 0 else 0.0,
            applied,
            restored,
            0,
            0,
            {},
            results_parsed=0,
            test_command_exit_code=run.exit_code,
            test_output_tail=((run.stdout or "") + (run.stderr or ""))[-2000:],
        )

    if data.split == "r2e":
        # Run nothing unless the hidden tests and their runner are exactly
        # where `_restore_r2e` put them: whatever else sits at run_tests.sh
        # is the patch's, and its verdict is worth nothing.
        placed = await runtime.run(
            [
                "sh",
                "-c",
                f'test -L "$WORKDIR"/r2e_tests && '
                f'test "$(readlink "$WORKDIR"/r2e_tests)" = {_R2E_STASH}/r2e_tests && '
                f'cmp -s {_R2E_STASH}/run_tests.sh "$WORKDIR"/run_tests.sh',
            ],
            {"WORKDIR": data.workdir},
        )
        if placed.exit_code != 0:
            return Report(
                0.0,
                applied,
                False,
                0,
                0,
                {},
                test_output_tail="hidden tests not in place; nothing was run",
            )
        # R2E's own procedure (`_calculate_reward_r2e`): run the image's
        # script from the repository root, parse, compare to the expected
        # map. Only stdout is parsed: both runners print their summary
        # there, and tornado's logs `ERROR:tornado...` lines to stderr that
        # would parse as spurious entries if appended after it.
        run = await runtime.run(
            [
                "sh",
                "-c",
                f'cd "$WORKDIR" && timeout -k 10 {_R2E_TEST_TIMEOUT_SECONDS} bash run_tests.sh',
            ],
            {"WORKDIR": data.workdir},
        )
        output = _ANSI_OUTPUT.sub("", run.stdout or "")
        results = parse_log_pytest(output)
        return Report(
            r2e_reward(results, data.expected_output_json),
            applied,
            restored,
            0,
            0,
            results,
            results_parsed=len(results),
            test_command_exit_code=run.exit_code,
            test_output_tail=(output + (run.stderr or ""))[-2000:],
        )

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
