"""SWE-bench as one Verifiers taskset.

The agent gets a repository at its base commit, a bug report, and whatever
shell its harness provides. It gets nothing else: no network, no history past
the base commit, no build output that could name the fix.

This module deliberately defines no tools. The harness supplies them, and
`verifiers` already ships several (`bash`, `mini_swe_agent`, `codex`,
`claude_code`, `terminus_2`). Training across more than one is how task-solving
strategy stops being welded to a single harness's quirks.
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Literal

import verifiers.v1 as vf
from pydantic import field_validator
from verifiers.v1.utils.git import PATCH_CAP_BYTES

from reliquary_swe import corpus, swe_adapter, swesmith_adapter

WORKDIR = "/testbed"
PATCH_PATH = f"{vf.ARTIFACTS_DIR}/patch.diff"
BASE_REF = "refs/reliquary/base"
"""Where setup records the base the agent's diff is taken against: in the box, so no host
memory outlives an episode (a gateway restarts; one Task serves concurrent episodes). The
agent can move it: grading recomputes everything from the image and refuses what a forged
diff could name, so the capture is never trusted."""
UNTRACKED_FILE = ".git/reliquary-untracked"
"""The image's untracked files at setup (NUL-separated, relative to the workdir), which
finalize leaves out of the diff."""
MAX_UNTRACKED_BYTES = 1024 * 1024
MAX_PATCH_BYTES = PATCH_CAP_BYTES
_RECORD_BASE = f' && git -C "$WORKDIR" update-ref {BASE_REF} HEAD'
"""Appended with `&&`, never `;`: R2E's cleanup ends on a `test` list `set -e` does not
exit on, so after a `;` a later command's success would hide a hidden-test leak."""

PROMPT = """\
You are working in a checked-out repository at {workdir}.

{problem_statement}

Fix the issue by editing the repository's source. Do not edit the tests.
"""

# Everything an agent could read the published fix out of. Build logs and
# verifier output are produced while the image is built with the reference
# patch applied, so they can name the very lines under test. Third-party
# dependencies are kept: the repository must still build offline.
#
# Split into three parts, assembled by `setup()`, because SWE-smith needs an
# extra step *between* checkout and the generic ref-stripping below (see
# `_TRAIN_GUARD_AND_REROOT`'s own docstring) -- order matters: the guard and
# re-root must run before `reflog expire`+`gc`, or the ancestor they sever
# survives (git will not let `gc` drop an ancestor of HEAD).
#
# Every step addresses the repository through `$WORKDIR` rather than a
# literal path: the polyglot corpus ships images whose repository lives at
# `/workspace/repo`, not `/testbed`.
#
# `--detach` matters only for the polyglot corpus, whose `base_commit` is the
# symbolic `HEAD`: checking out `HEAD` leaves HEAD attached to its branch, the
# ref stripping below then deletes that branch, and HEAD is left pointing at
# an unborn one -- `git rev-parse HEAD` fails, and `gc` no longer counts the
# base commit as reachable. Confirmed on a real polyglot image. For a SHA or
# an `origin/<id>~1` expression it changes nothing: those already detach.
_CHECKOUT = " ; ".join(
    [
        "set -e",
        'git -C "$WORKDIR" reset --hard "$BASE_COMMIT"',
        'git -C "$WORKDIR" checkout -q --detach "$BASE_COMMIT"',
    ]
)

# SWE-smith-only (spec section 8's corpus has no analogue of this: a
# SWE-bench Verified `base_commit` is an ordinary point in real upstream
# history, and the fix for its own bug is a *descendant*, which the ref
# stripping below already handles). Confirmed by hand, on `setup()` before
# this fix existed (see the implementation report):
# upstream builds every SWE-smith image's `main` from the pristine,
# already-fixed tree as a single commit, then builds each instance's own
# branch as exactly that commit plus one child, "Bug Patch". So
# `origin/<instance_id>~1` -- this package's `base_commit` for a train row
# -- is *the pristine tree's own child*, and after checkout the box holds
# precisely two commits with the fix sitting one parent away:
#
#   git diff HEAD HEAD^      # == the exact gold patch, verbatim
#   git show HEAD^:<path>    # == the fixed file, verbatim
#
# Stripping refs (the step below, shared with the SWE-bench Verified path)
# does not touch this: HEAD's own parent pointer keeps the pristine commit
# reachable with no ref needed at all, and `git gc` will never prune an
# ancestor of HEAD. Two steps close it: assert the branch is shaped as
# documented (fail loud rather than silently pay a mis-shaped one -- see
# also the FAIL_TO_PASS-emptiness guard in `corpus.load_swesmith_rows`),
# then re-root HEAD onto a fresh, parentless commit carrying the identical
# tree, which severs the ancestry link outright so the ref-stripping and
# `gc` immediately below actually make the old chain unreachable and prune
# it -- verified directly: after this, `git rev-list --count HEAD` is 1 and
# the gold patch's added lines are absent from a full
# `git cat-file --batch-all-objects --batch` scan.
_TRAIN_GUARD_AND_REROOT = " ; ".join(
    [
        'test "$(git -C "$WORKDIR" log -1 --format=%s "$BASE_COMMIT")" = "Bug Patch"',
        'NEW_ROOT=$(git -C "$WORKDIR" -c user.name=reliquary-swe '
        "-c user.email=reliquary-swe@localhost "
        "commit-tree HEAD^{tree} -m base)",
        'git -C "$WORKDIR" reset --hard "$NEW_ROOT"',
    ]
)

_STRIP_AND_GC = " ; ".join(
    [
        # Drop every ref that could reach a later commit, then expire the
        # reflog so neither `git log --all` nor `git fsck` finds one.
        """git -C "$WORKDIR" for-each-ref --format='%(refname)' """
        '| xargs -r -n1 git -C "$WORKDIR" update-ref -d',
        # `for-each-ref` skips a *broken* symbolic ref (one naming a branch
        # that does not exist) with only a warning, so the loop above never
        # deletes it -- and `gc` then dies on it ("failed to run repack").
        # Real polyglot images ship exactly that: `refs/remotes/origin/HEAD`
        # pointing at a remote branch the image never fetched. Any loose ref
        # file left after the loop is one of these; remove it directly.
        'find "$WORKDIR"/.git/refs -mindepth 1 -type f -delete',
        'git -C "$WORKDIR" reflog expire --expire=now --all',
        'git -C "$WORKDIR" gc --prune=now --quiet',
        'rm -rf "$WORKDIR"/.git/logs',
        'rm -f "$WORKDIR"/*.orig "$WORKDIR"/*.rej',
        "rm -f /*.patch /home/*.patch /tmp/*.patch /root/*.patch",
        """find "$WORKDIR" -name '__pycache__' -type d -prune -exec rm -rf {} + || true""",
        "rm -rf /root/.cache/pip /tmp/build",
    ]
)


# R2E only, in place of `_CHECKOUT`. An R2E image's working tree is NOT its
# HEAD: measured on the pandas golden, the image builder staged edits to
# `pandas/__init__.py`, `_version.py` and `versioneer.py`, edited `setup.cfg`
# (dropping an `addopts = --strict-data-files` that the hidden tests cannot
# run under) and deleted `pyproject.toml` -- and `_CHECKOUT`'s
# `git reset --hard` puts all of it back, after which pytest refuses to start
# and every rollout scores 0, gold patch included. So the image's own state
# is committed instead, as a child of HEAD (the pre-fix commit), and that
# commit is the base the agent's diff is taken against and the base grading
# restores from (`grade()` runs the same step). On a clean image (coveragepy,
# tornado) it is an empty commit. Untracked files (`run_tests.sh`,
# `install.sh`, build output) stay untracked, as setup's untracked list
# expects.
_R2E_CHECKOUT = " ; ".join(
    [
        "set -e",
        'git -C "$WORKDIR" checkout -q --detach HEAD',
        'git -C "$WORKDIR" -c user.name=reliquary-swe '
        "-c user.email=reliquary-swe@localhost "
        "commit -q -a --allow-empty --no-verify -m base",
    ]
)

# R2E only, run before `_STRIP_AND_GC`: refuse a repository whose objects
# the strip cannot vouch for. `gc` prunes what no ref reaches in THIS object
# store; objects borrowed through `objects/info/alternates`, a linked
# worktree's own HEAD (`.git/worktrees`), and the history rewrites of a
# shallow file or grafts all sit outside what the strip-and-gc reasoning
# covers -- a fix commit could survive behind any of them. None was present
# on the images checked (the three goldens and the numpy and orange3 tasks
# re-run on 2026-10-04); this makes a future image that has one fail setup
# loudly instead of leaking quietly.
_R2E_LEAK_GUARD = " ; ".join(
    [
        'test ! -s "$WORKDIR"/.git/objects/info/alternates',
        'test ! -e "$WORKDIR"/.git/worktrees',
        'test ! -e "$WORKDIR"/.git/shallow',
        'test ! -e "$WORKDIR"/.git/info/grafts',
    ]
)

# R2E only, run after `_STRIP_AND_GC`. Every R2E image carries its hidden
# tests inside the very box the agent works in -- `/r2e_tests` (the test
# files) and `/testbed/run_tests.sh` (the command that runs them), measured
# on three images -- and R2E's own runtime hides both before its agent starts
# (`SKIP_FILES_NEW` moved under `/root`). Deleting them is enough here: the
# agent's box is never graded, and grading takes both from a fresh box of the
# same image (see `grading._restore_r2e`). `$WORKDIR/r2e_tests` is not in the
# image; it is removed in case a future image ships R2E's symlink already.
_R2E_HIDE_TESTS = " ; ".join(
    [
        'rm -rf /r2e_tests "$WORKDIR"/r2e_tests "$WORKDIR"/run_tests.sh',
        'test ! -e /r2e_tests && test ! -e "$WORKDIR"/run_tests.sh',
    ]
)


def cleanup_script(split: str) -> str:
    """The shell cleanup `SweTask.setup` runs before the agent starts (and the sandbox's
    `prepare`, reliquary_swe.sandbox): checkout, SWE-smith's guard and re-root, R2E's
    leak guard, ref stripping and gc, R2E's hidden-test removal, in that order."""
    steps = [_R2E_CHECKOUT if split == "r2e" else _CHECKOUT]
    # `split` -- already on the wire, and the honest discriminator for this
    # decision (it is literally "which corpus is this") -- decides whether the
    # SWE-smith-only guard-and-reroot step runs. See `_TRAIN_GUARD_AND_REROOT`'s
    # own docstring for why "train" needs it and "eval" must not: a SWE-bench
    # Verified `base_commit` has no pristine-tree ancestor to sever, and
    # asserting a "Bug Patch" commit message on it would just fail every eval
    # task.
    #
    # "polyglot" needs no re-root either. Its images come in two shapes, both
    # checked on real images: HEAD is a parentless "task base" commit whose
    # tree has the requested behaviour cut out, with the complete upstream --
    # implementation and tests -- still parked under `origin/<branch>`; or HEAD
    # is upstream's own tip, with nothing after it. Either way nothing an agent
    # could use sits in HEAD's ancestry, and the ref stripping below prunes the
    # parked upstream: after it, a full object-store scan no longer finds the
    # removed code.
    if split == "train":
        steps.append(_TRAIN_GUARD_AND_REROOT)
    if split == "r2e":
        steps.append(_R2E_LEAK_GUARD)
    steps.append(_STRIP_AND_GC)
    # "r2e" needs no re-root either: HEAD is detached at the pre-fix commit,
    # and the fix commit is a *descendant* reachable only through the full
    # upstream history's branch refs, which the strip deletes and `gc` then
    # prunes (asserted on real images in tests/test_r2e_goldens.py). It does
    # need its hidden tests out of the box.
    if split == "r2e":
        steps.append(_R2E_HIDE_TESTS)
    return " ; ".join(steps)


_SANDBOX = "reliquary-sandbox"
"""The runtime type of a signed-episode sandbox: the only runtime the reward grades on
(the sandbox calls it only in its grading box)."""
_CAPTURE_DIR = "/tmp/reliquary-capture"
"""Prefix of finalize's scratch repository (a host nonce follows: the agent cannot
pre-create it)."""
_NO_HOME = "/tmp/reliquary-no-home"
"""Prefix of the HOME git runs with outside the scratch repository: a host nonce
follows and nothing creates it, so no git before 2.32 finds a config there."""
_EXCLUDE = (
    'x="$2/.git/info/exclude"; { [ -f "$x" ] && [ ! -L "$x" ]; } || x=/dev/null; '
)
"""The one excludes file both the untracked list and the capture read: the repository's
`info/exclude` when it is a regular file (a FIFO would hang git), else none."""
_PREPARE_CAPTURE = (
    _EXCLUDE
    + 'mkdir -m 700 -- "$1" && git init -q --bare --template= "$1"'
    ' && git --git-dir="$1" config core.bare false'
    ' && git --git-dir="$1" config core.excludesFile "$x"'
    ' && base=$(git --git-dir="$1" rev-parse --verify -q "$3^{commit}")'
    ' && git --git-dir="$1" update-ref --no-deref HEAD "$base"'
    ' && git --git-dir="$1" read-tree HEAD'
)
"""The scratch repository finalize captures in: empty (no hook, no config, no remote of
the agent's), the agent's objects reached read-only as an alternate, HEAD and index at
the base, peeled here (a missing object fails, it is never fetched). The agent's own
exclude file is kept: it only hides paths, as `.gitignore` does."""
_LIST_UNTRACKED = (
    _EXCLUDE + 'exec git -C "$2" -c core.excludesFile="$x" ls-files --others --exclude-standard -z'
)
"""setup's untracked list, under the capture's ignore rules (`.gitignore` files and the
same excludes file, no global or system one)."""


def _hardened_git_env(home: str) -> dict[str, str]:
    """No system, global or XDG git config, no system attributes, no replace refs, no
    lazy fetch from a promisor remote."""
    return {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "HOME": home,  # git before 2.32 ignores GIT_CONFIG_GLOBAL
        "XDG_CONFIG_HOME": home,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
    }


def _no_home() -> str:
    return f"{_NO_HOME}-{uuid.uuid4().hex}"


def _capture_env(workdir: str, scratch: str) -> dict[str, str]:
    """The capture's git: the scratch repository over the agent's work tree. Every git
    setting the agent can write (its `.git/config` and what it includes, hooks,
    `.git/info/attributes`, fsmonitor, a redirected `core.worktree`, replace refs, a
    `.git` file pointing elsewhere) lives in a repository this one never reads; the
    in-tree `.gitattributes` it still reads names drivers only config defines, and
    none is defined. What reaches it is plain data: the work tree and the objects."""
    return {
        **_hardened_git_env(scratch),
        "GIT_DIR": scratch,
        "GIT_WORK_TREE": workdir,
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": f"{workdir}/.git/objects",
    }


async def _resolve_base(runtime: vf.Runtime, workdir: str) -> str:
    """The base ref's object name, else HEAD's, else "": read in the agent's repository
    without peeling, so no object is read (a missing one in a partial clone would be
    fetched through the agent's remote); its local config applies, but this `rev-parse`
    runs no hook, filter, fsmonitor or transport. The scratch repository peels it."""
    env = {**_hardened_git_env(_no_home()), "GIT_DIR": f"{workdir}/.git"}
    for ref in (BASE_REF, "HEAD"):
        result = await runtime.run(["git", "rev-parse", "--verify", "-q", ref], env)
        sha = (result.stdout or "").strip()
        if result.exit_code == 0 and sha:
            return sha
    return ""


async def _capture(
    sink: vf.Trace | SimpleNamespace,
    runtime: vf.Runtime,
    workdir: str,
    ignore: list[str],
    write_path: str | None = None,
) -> None:
    """verifiers' `capture_patch` against the base, in a scratch repository of ours
    (`_capture_env`), removed after. Sets `sink.info["patch"]` or `["patch_error"]`."""
    scratch = f"{_CAPTURE_DIR}-{uuid.uuid4().hex}"
    base = await _resolve_base(runtime, workdir)
    env = _capture_env(workdir, scratch)
    if not base:
        sink.info["patch_error"] = "no base: neither the base ref nor HEAD resolves"
    else:
        prepared = await runtime.run(
            ["sh", "-c", _PREPARE_CAPTURE, "reliquary-capture", scratch, workdir, base],
            {k: v for k, v in env.items() if k not in ("GIT_DIR", "GIT_WORK_TREE")},
        )
        if prepared.exit_code != 0:
            if (await runtime.run(["true"], {})).exit_code != 0:
                raise vf.SandboxError("patch capture failed and the box stopped answering")
            sink.info["patch_error"] = (
                f"exit={prepared.exit_code} {(prepared.stderr or '').strip()[-500:]}"
            )
        else:
            await vf.capture_patch(
                sink, runtime, base_commit=base, env=env, ignore=ignore,  # type: ignore[arg-type]
                write_path=write_path,
            )
    await runtime.run(["rm", "-rf", "--", scratch], {})


async def _remove_patch(runtime: vf.Runtime) -> None:
    result = await runtime.run(["rm", "-f", "--", PATCH_PATH], {})
    if result.exit_code != 0:
        # Kept, it would travel as the agent's patch: the extract fails instead (0).
        raise RuntimeError(
            f"could not remove {PATCH_PATH} (exit {result.exit_code}): "
            f"{(result.stderr or '').strip()[-300:]}"
        )


class SweData(vf.TaskData):
    instance_id: str
    repo: str
    base_commit: str
    # Grading keys `MAP_REPO_VERSION_TO_SPECS[repo][version]` to find the
    # instance's test command (see `swe_adapter.test_command`); without this
    # field on the wire data, a grading task would have to guess a version
    # rather than read the one the corpus row already carries.
    version: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    gold_patch: str
    test_patch: str
    split: str
    # Polyglot only (see `corpus.SweRow.test_command`): the corpus's own
    # command, whose exit status is the verdict. Defaulted so the wire shape
    # of the other two corpora is unchanged.
    test_command: str = ""
    # R2E only (see `corpus.SweRow.expected_output_json`): the exact verdict
    # map grading must reproduce. Defaulted so the other corpora's wire
    # shape is unchanged.
    expected_output_json: str = ""


class SweTask(vf.Task[SweData]):
    NEEDS_CONTAINER = True
    _graded_elsewhere: bool = False

    @property
    def key(self) -> str:
        return self.data.instance_id

    def graded_elsewhere(self) -> SweTask:
        """A copy whose reward records nothing here: `SweEnv` grades the patch in a box of
        its own (verifiers' Harbor pattern)."""
        clone = copy.copy(self)
        clone._graded_elsewhere = True
        return clone

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """The cleanup (`cleanup_script`), then the base ref and the image's untracked
        list, both kept in the box: a Task keeps no per-episode state of its own."""
        script = cleanup_script(self.data.split) + _RECORD_BASE
        result = await runtime.run(
            ["sh", "-c", script],
            {"BASE_COMMIT": self.data.base_commit, "WORKDIR": self.data.workdir},
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"environment preparation failed for {self.data.instance_id} "
                f"(exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )
        workdir = self.data.workdir
        listing = await runtime.run(
            ["sh", "-c", _LIST_UNTRACKED, "reliquary-untracked", workdir],
            _hardened_git_env(_no_home()),
        )
        if listing.exit_code != 0:
            raise RuntimeError(
                f"environment preparation failed for {self.data.instance_id}: listing the "
                f"image's untracked files (exit {listing.exit_code}): "
                f"{(listing.stderr or '').strip()[-500:]}"
            )
        untracked = [p for p in (listing.stdout or "").split("\0") if p]
        listed = "\0".join(untracked).encode()
        if len(listed) > MAX_UNTRACKED_BYTES:
            # finalize reads at most this much back; a longer list would silently stop
            # ignoring the image's files. An image this shape needs a closer look first.
            raise RuntimeError(
                f"environment preparation failed for {self.data.instance_id}: the image's "
                f"untracked list is {len(listed)} bytes, over {MAX_UNTRACKED_BYTES}"
            )
        await runtime.write(f"{workdir}/{UNTRACKED_FILE}", listed)
        # The capture reads none of the image's git config: an image its own git reads
        # as clean but the capture does not (core.filemode, core.autocrlf, a filter
        # defined outside the repository) would put that difference in every agent's
        # patch, graded 0. Found here, on the untouched box, it is ours.
        check = SimpleNamespace(info={})
        await _capture(check, runtime, workdir, untracked)
        if check.info.get("patch") != "":
            detail = check.info.get("patch_error") or check.info.get("patch", "")[:300]
            raise RuntimeError(
                f"environment preparation failed for {self.data.instance_id}: the "
                f"untouched box does not capture as an empty patch: {detail}"
            )

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """The agent's diff against the base setup recorded in the box, without the
        image's untracked files, written to PATCH_PATH (verifiers' artifact convention).

        Diffed against that base rather than bare HEAD, so commits the agent made are
        inside it. Edits to tests are captured on purpose: grading discards them, and
        keeping them in the trace is what makes reward hacking visible afterwards. Only
        box commands and file writes happen here (env norm): grading parses the patch,
        never this. A base ref or an untracked list the agent removed or broke is its
        own outcome: the diff is wider and fails to apply in the pristine box.

        The capture runs in a scratch repository of ours (`_capture_env`), so no git
        setting or hook the agent left runs; whenever it writes no patch, PATCH_PATH is
        removed again and checked gone: a patch planted there never travels. A box that
        stops answering raises (ours, not the agent's).
        """
        await _remove_patch(runtime)  # a planted patch never travels
        workdir = self.data.workdir
        try:
            listed = await runtime.read(
                f"{workdir}/{UNTRACKED_FILE}", max_bytes=MAX_UNTRACKED_BYTES
            )
            ignore = [p for p in listed.decode("utf-8", "replace").split("\0") if p]
        except TimeoutError:
            raise  # the step deadline (an OSError subclass)
        except (OSError, vf.SandboxError):
            # Missing, not a regular file or inflated by the agent (verifiers' capped
            # read raises SandboxError for all three): its diff carries more.
            ignore = []
        await _capture(trace, runtime, workdir, ignore, write_path=PATCH_PATH)
        if "patch" not in trace.info:
            await _remove_patch(runtime)  # whatever ran during a failed capture planted

    async def grading_setup(self, runtime: vf.Runtime) -> None:
        """Env norm: the pristine grading box, before the agent's patch reaches it (the
        history strip, R2E's hidden tests set aside). A failure is about us or the image."""
        from reliquary_swe import grading

        await grading.prepare_box(runtime, self.data)

    @vf.reward(weight=1.0)
    async def patch_passes_tests(
        self, runtime: vf.Runtime, trace: vf.Trace
    ) -> float | dict[str, float]:
        """Env norm: `runtime` is a pristine box of the image that `grading_setup`
        prepared and that received only the declared artifacts. The runtime type tells a
        signed-episode sandbox from any other runtime; a sandbox calls the reward only in
        its grading box. On any other runtime it records nothing: `SweEnv` grades in a
        box of its own, and verifiers' default env would call it in the agent's box."""
        from reliquary_swe import grading

        box_type = getattr(getattr(runtime, "config", None), "type", None)
        if self._graded_elsewhere or box_type != _SANDBOX:
            return {}
        try:
            raw = await runtime.read(PATCH_PATH, max_bytes=MAX_PATCH_BYTES)
        except FileNotFoundError:
            raw = b""  # finalize wrote none (git refused): graded as an empty patch
        except TimeoutError:
            raise
        except OSError:  # over the cap, or not a regular file: nothing to apply
            trace.record_metrics({"patch_unreadable": 1.0})
            return 0.0
        data = await grading.prepared_data(runtime, self.data)
        report = await grading.grade_prepared(runtime, data, raw)
        metrics = {
            "applied": float(report.applied),
            "restored": float(report.restored),
            "fail_to_pass_passed": float(report.fail_to_pass_passed),
            "fail_to_pass_total": float(len(self.data.fail_to_pass)),
            "pass_to_pass_passed": float(report.pass_to_pass_passed),
            "pass_to_pass_total": float(len(self.data.pass_to_pass)),
            "results_parsed": float(report.results_parsed),
        }
        if report.test_command_exit_code is not None:
            metrics["test_command_exit_code"] = float(report.test_command_exit_code)
        trace.record_metrics(metrics)
        return report.reward


class SweTasksetConfig(vf.TasksetConfig):
    # `vf.TasksetConfig` carries no `split`; every sibling package that loads
    # more than one split (see `reliquary_code.taskset.CodeConfig`) defines
    # its own. "eval" is SWE-bench Verified (spec section 8: an evaluation
    # set, never trained on); "train" is SWE-smith, this package's training
    # corpus.
    #
    # A real risk, worth naming: now that "train" exists, the likeliest
    # operator error is wiring `[[orchestrator.train.source]]` and
    # forgetting to say which split, which -- with a plain string default --
    # would train on the evaluation set silently, at full speed. A
    # *required* `Literal["eval", "train"]` was tried first, to force every
    # caller to declare intent, and reverted: it breaks the CLI outright,
    # verified directly. `resolve.py`'s `narrow_taskset_config` (behind
    # every taskset-first CLI -- `validate`, `debug`) constructs
    # `SweTasksetConfig(id="reliquary-swe")` with no other fields as a
    # bootstrapping step *before* CLI overrides are applied, and a required
    # field makes that step itself raise `pydantic.ValidationError`,
    # regardless of what `--taskset.split` the real invocation passes.
    #
    # `| None = None` is the shape that satisfies both constraints at once:
    # `taskset_type(id=...)` stays constructible (`None` is a valid default,
    # unlike a bare required field), and `SweTaskset.load()` below raises
    # loudly the moment `None` is actually used to load rows -- so an
    # operator who wires a train source and forgets `env.taskset.split`
    # gets a `ValueError` naming the problem, not a silent fallback to
    # `"eval"`. `rl.toml` shipping with no `[[orchestrator.train.source]]`
    # at all (`TrainSource` itself raises with none) remains the first line
    # of defense; this is the second, for the moment a real source exists.
    #
    # "polyglot" is a second training corpus: MiMo-V2.6-RL-oss's `code`
    # config, 2,698 tasks in eight languages, graded by the exit status of a
    # hidden test script rather than by test lists (see
    # `corpus.load_polyglot_rows`).
    #
    # "r2e" is a third: R2E-Gym-Subset, 4,578 tasks from the real commit
    # history of 10 Python repositories, graded by reproducing the exact
    # verdict map of hidden tests (see `corpus.load_r2e_rows`).
    split: Literal["eval", "train", "polyglot", "r2e"] | None = None
    # Only read when split="train". Together with `max_test_count`, this
    # determines the exact task set (spec section 8: "declared, never
    # auto-detected"); see `corpus.load_swesmith_rows` for why a fixed pair
    # is what lets two machines agree on what task #400 is.
    num_images: int = corpus.DEFAULT_SWESMITH_IMAGES

    @field_validator("num_images")
    @classmethod
    def _num_images_within_the_pins(cls, value: int) -> int:
        # Refused at config time rather than as a KeyError deep in loading:
        # only the top images carry a pinned digest.
        corpus._check_swesmith_num_images(value)
        return value
    # Only read when split="train". Caps fail-to-pass + pass-to-pass test
    # count per instance -- SWE-smith's own per-instance cost is heavily
    # right-skewed, and this is the declared, deterministic bound on it.
    # `None` (the default) disables the cap: this package's own real
    # container measurements found the shipped corpus's cost a non-problem
    # (see `corpus.DEFAULT_SWESMITH_MAX_TEST_COUNT`'s own docstring), so
    # discarding real training rows by default to bound a cost that is not
    # observed here would be the wrong trade. Pass
    # `corpus.DEFAULT_SWESMITH_MAX_TEST_COUNT` (its measured p95) explicitly
    # if a repository this package has not measured ever needs it.
    max_test_count: int | None = None
    # Only read when split="polyglot" or "r2e": the first N tasks of the
    # pinned corpus, `None` for all of them (2,698 and 4,578). One image per
    # task in both, so this is the disk budget -- see
    # `corpus.load_polyglot_rows` and `corpus.load_r2e_rows`.
    num_tasks: int | None = None


# Every phase of a rollout defaults to no limit at all (`TimeoutConfig`'s and
# `TaskTimeout`'s own fields are all `None`), and this is the one environment
# that hands a policy a general shell with `max_turns` bounding turns, not
# wall clock. Three of the four below are sized past a real measurement,
# never guessed; `setup` is the exception -- see its own comment -- and is
# sized by analogy with the other three instead. The measurements
# themselves, and the headroom reasoning, live in the package README's
# "Timeouts" section rather than here.

# `setup()` (`_CHECKOUT` + `_STRIP_AND_GC`, and, for a train row,
# `_TRAIN_GUARD_AND_REROOT` in between) plus the harness's own setup --
# `rollout.py` computes one setup-stage deadline and wraps both
# `task.setup` and `harness.setup(runtime)` in it ("Task setup and harness
# provisioning share one setup-stage deadline"), so this budget has to
# cover both terms, and only the first one is actually measured. Our own
# cleanup: on the container host, the heaviest sampled repository
# (matplotlib, 296 MB `.git`, the largest of six families checked) runs
# the whole cleanup script, `git gc --prune=now` included, in 3.6s -- a
# SWE-bench Verified measurement, predating `_TRAIN_GUARD_AND_REROOT`; a
# train row's extra `git log`, `commit-tree` and `reset --hard` are cheap,
# single-object git operations on top of that, not re-measured separately
# because the harness-install term below dominates this budget by roughly
# two orders of magnitude either way. The harness's setup, for
# the `bash` harness this branch's example pins, is a genuine per-rollout
# network install inside the fresh container: `pip install -q -U --user
# uv` (falling back to `apt-get install curl` plus a curl-fetched
# installer) followed by `uv sync --script`, which fetches a managed
# CPython and the script's dependencies -- egress is still open at this
# point, it closes only after setup, and there is no cross-rollout cache
# (`_uv_interpreters` lives on the per-rollout Runtime). That second term
# is the one that dominates this budget, and it has not been measured --
# the budget is sized by analogy with the other three phases below, not
# derived from it. If a pilot hits this ceiling anyway, `--env.agent.timeout.setup`
# overrides the task value at run time without a code change.
# Capped by signed-episode sandboxes: setup at most 600 s, each grading step at most
# 810 s (their verification window).
_SETUP_TIMEOUT_SECONDS = 600.0

# The agent's solve attempt -- the phase Important 1 is actually about.
# Running the repository's own tests is the most natural thing a repair
# agent does, and the slowest real suite run measured (django, no test
# directives -> its entire 12,311-test suite) took ~225s wall clock; the
# worst *projected* one (matplotlib's largest instance, unrunnable here) is
# ~570s. An hour gives room for several such runs plus editing, while still
# turning "holds the slot for hours" into a bounded, finite failure.
_AGENT_TIMEOUT_SECONDS = 3600.0

# `finalize()`'s `git add -A` + `git diff --cached --binary` against
# whatever the agent's box holds. Measured on the container host: a single
# 573 MB novel file costs 35.2s combined -- roughly 60s/GB. 600s covers
# ~10 GB of agent-authored content (the disk the task declares), far past any
# legitimate edit, while still bounding a disk-filling pathology to a fixed
# ceiling.
# Capped by signed-episode sandboxes: setup at most 600 s, each grading step at most
# 810 s (their verification window).
_FINALIZE_TIMEOUT_SECONDS = 600.0

# `env.py`'s `_grade` wraps provisioning-through-grading in
# `asyncio.timeout(task.data.timeout.scoring)`; left at `TaskData`'s default
# (`None`) this is unbounded, so a reachable-but-HANGING box would never
# raise and never score. 810s (13.5 minutes) is sized past the measured
# tail with headroom, not guessed -- full reasoning, the measured times it is checked against, and the p90/max
# corpus figures (independently re-measured, not just quoted) live in the
# package README's "Timeouts" section rather than here.
# Capped by signed-episode sandboxes: setup at most 600 s, each grading step at most
# 810 s (their verification window).
_SCORING_TIMEOUT_SECONDS = 810.0


class SweTaskset(vf.Taskset[SweTask, SweTasksetConfig]):
    def _split(self) -> str:
        if self.config.split is None:
            raise ValueError(
                "reliquary-swe: --taskset.split is required (\"eval\" for "
                "SWE-bench Verified, \"train\" for SWE-smith, \"polyglot\" "
                "for MiMo-V2.6-RL-oss's code tasks, \"r2e\" for "
                "R2E-Gym-Subset) -- it has no "
                "default so that an operator wiring a real training source "
                "cannot silently fall back to the evaluation set"
            )
        return self.config.split

    def __len__(self) -> int:
        split = self._split()
        if split == "train":
            return len(corpus.swesmith_order(self.config.num_images, self.config.max_test_count))
        if split == "r2e":
            size = len(corpus.r2e_instance_ids())
            return size if self.config.num_tasks is None else min(size, self.config.num_tasks)
        if split == "polyglot":
            return len(corpus.load_polyglot_rows(self.config.num_tasks))
        return len(corpus.load_rows(split))

    def task_at(self, index: int) -> SweTask:
        """Task `index` of the split, built alone (`load` builds them all: SWE-smith's
        default split weighs about 4 GB as rows). Equal to the index-th task `load`
        yields."""
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        split = self._split()
        if split == "train":
            row = corpus.swesmith_row_at(
                self.config.num_images, index, self.config.max_test_count
            )
            swesmith_adapter.ensure_python_profile(row.repo)
        elif split == "r2e":
            row = corpus.r2e_row_at(index)
        elif split == "polyglot":
            row = corpus.load_polyglot_rows(self.config.num_tasks)[index]
        else:
            row = corpus.load_rows(split)[index]
        return task_for(row, index, split, self.config.task)

    def load(self) -> Iterator[SweTask]:
        self._split()
        if self.config.split == "train":
            rows = corpus.load_swesmith_rows(
                self.config.num_images, self.config.max_test_count
            )
            # Once per distinct repo, not once per row (registry.get_from_inst
            # is cheap, but there is no reason to repeat it thousands of
            # times for one repo's worth of instances) -- fails loudly at
            # load time if `num_images` reaches a non-Python image, rather
            # than mis-scoring every rollout for that repo silently.
            for repo in dict.fromkeys(row.repo for row in rows):
                swesmith_adapter.ensure_python_profile(repo)
        elif self.config.split == "polyglot":
            rows = corpus.load_polyglot_rows(self.config.num_tasks)
        elif self.config.split == "r2e":
            rows = corpus.load_r2e_rows(self.config.num_tasks)
        else:
            rows = corpus.load_rows(self.config.split)
        for index, row in enumerate(rows):
            yield task_for(row, index, self.config.split, self.config.task)


def task_for(row: corpus.SweRow, index: int, split: str, task_config=None) -> SweTask:
    """One corpus row as a task -- the construction `SweTaskset.load` uses
    for every row, exposed so a caller needing ONE task (a golden test) can
    build it without materializing the whole split: SWE-smith's default 20
    images cost more than 5 GB of memory as tasks, more than a CI runner has.
    """
    return SweTask(
        SweData(
            idx=index,
            name=row.instance_id,
            prompt=PROMPT.format(workdir=row.workdir, problem_statement=row.problem_statement),
            # Given directly for SWE-smith, polyglot and R2E (row.image); derived
            # for SWE-bench Verified, whose rows carry none (spec section 8:
            # "the image is given, not derived" -- Verified's own derivation
            # is the fallback, not the rule).
            image=row.image if row.image is not None else swe_adapter.image_for(row),
            workdir=row.workdir,
            network_allow=[],
            timeout=vf.TaskTimeout(
                setup=_SETUP_TIMEOUT_SECONDS,
                agent=_AGENT_TIMEOUT_SECONDS,
                finalize=_FINALIZE_TIMEOUT_SECONDS,
                scoring=_SCORING_TIMEOUT_SECONDS,
            ),
            # Provisional: no SWE box declared any before (Docker's default is
            # unlimited); a signed-episode sandbox enforces both.
            resources=vf.TaskResources(memory=4.0, disk=10.0),
            instance_id=row.instance_id,
            repo=row.repo,
            base_commit=row.base_commit,
            version=row.version,
            fail_to_pass=row.fail_to_pass,
            pass_to_pass=row.pass_to_pass,
            gold_patch=row.gold_patch,
            test_patch=row.test_patch,
            split=split,
            test_command=row.test_command,
            expected_output_json=row.expected_output_json,
        ),
        task_config,
    )

__all__ = [
    "BASE_REF",
    "PATCH_PATH",
    "UNTRACKED_FILE",
    "SweData",
    "SweTask",
    "SweTasksetConfig",
    "SweTaskset",
    "cleanup_script",
    "task_for",
]
