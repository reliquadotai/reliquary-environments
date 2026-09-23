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

from collections.abc import Iterator
from typing import ClassVar, Literal

import verifiers.v1 as vf
from verifiers.v1.utils.git import snapshot_untracked

from reliquary_swe import corpus, swe_adapter, swesmith_adapter

WORKDIR = "/testbed"
PATCH_PATH = f"{vf.ARTIFACTS_DIR}/patch.diff"

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
_CHECKOUT = " ; ".join(
    [
        "set -e",
        'git -C /testbed reset --hard "$BASE_COMMIT"',
        'git -C /testbed checkout -q "$BASE_COMMIT"',
    ]
)

# SWE-smith-only (spec section 8's corpus has no analogue of this: a
# SWE-bench Verified `base_commit` is an ordinary point in real upstream
# history, and the fix for its own bug is a *descendant*, which the ref
# stripping below already handles). Confirmed by hand, on the shipped
# `_CLEANUP` before this fix existed (see the implementation report):
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
        'test "$(git -C /testbed log -1 --format=%s "$BASE_COMMIT")" = "Bug Patch"',
        "NEW_ROOT=$(git -C /testbed -c user.name=reliquary-swe "
        "-c user.email=reliquary-swe@localhost "
        "commit-tree HEAD^{tree} -m base)",
        'git -C /testbed reset --hard "$NEW_ROOT"',
    ]
)

_STRIP_AND_GC = " ; ".join(
    [
        # Drop every ref that could reach a later commit, then expire the
        # reflog so neither `git log --all` nor `git fsck` finds one.
        "git -C /testbed for-each-ref --format='%(refname)' "
        "| xargs -r -n1 git -C /testbed update-ref -d",
        "git -C /testbed reflog expire --expire=now --all",
        "git -C /testbed gc --prune=now --quiet",
        "rm -rf /testbed/.git/logs",
        "rm -f /testbed/*.orig /testbed/*.rej",
        "rm -f /*.patch /home/*.patch /tmp/*.patch /root/*.patch",
        "find /testbed -name '__pycache__' -type d -prune -exec rm -rf {} + || true",
        "rm -rf /root/.cache/pip /tmp/build",
    ]
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


class SweTask(vf.Task[SweData]):
    NEEDS_CONTAINER = True

    # Keyed by id(runtime); set in setup, read in finalize. Host memory only:
    # a base commit kept inside the box is one the agent can rewrite.
    _heads: ClassVar[dict[int, str]] = {}
    _untracked: ClassVar[dict[int, list[str]]] = {}

    @property
    def key(self) -> str:
        return self.data.instance_id

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # `split` -- already on the wire, and the honest discriminator for
        # this decision (it is literally "which corpus is this") -- decides
        # whether the SWE-smith-only guard-and-reroot step runs. See
        # `_TRAIN_GUARD_AND_REROOT`'s own docstring for why "train" needs it
        # and "eval" must not: a SWE-bench Verified `base_commit` has no
        # pristine-tree ancestor to sever, and asserting a "Bug Patch"
        # commit message on it would just fail every eval task.
        steps = [_CHECKOUT]
        if self.data.split == "train":
            steps.append(_TRAIN_GUARD_AND_REROOT)
        steps.append(_STRIP_AND_GC)
        cleanup = " ; ".join(steps)
        result = await runtime.run(
            ["sh", "-c", cleanup], {"BASE_COMMIT": self.data.base_commit}
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"environment preparation failed for {self.data.instance_id} "
                f"(exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )
        self._heads[id(runtime)] = await vf.resolve_head(runtime)
        self._untracked[id(runtime)] = await snapshot_untracked(runtime)

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """Snapshot the agent's diff while its box is still alive.

        Diffed against the SHA recorded at setup rather than bare HEAD, so
        commits the agent made are inside it. Edits to test files are captured
        on purpose: grading discards them, and keeping them in the trace is
        what makes reward hacking visible afterwards.
        """
        try:
            head = self._heads.pop(id(runtime))
        except KeyError:
            # No `"" ` fallback: `capture_patch` turns a falsy `base_commit`
            # into bare `HEAD` (see its own docstring), which misses any
            # commit the agent made -- a silent, wrong zero, not a real one.
            # `setup()` always records an entry before the agent runs; a
            # missing one means it never ran for this runtime, which is a
            # bug worth raising loudly, not papering over.
            raise RuntimeError(
                f"no base commit recorded for {self.data.instance_id}'s "
                "runtime -- setup() must run before finalize()"
            ) from None
        await vf.capture_patch(
            trace,
            runtime,
            base_commit=head,
            ignore=self._untracked.pop(id(runtime), []),
            write_path=PATCH_PATH,
        )


class SweTasksetConfig(vf.TasksetConfig):
    # `vf.TasksetConfig` carries no `split`; every sibling package that loads
    # more than one split (see `reliquary_code.taskset.CodeConfig`) defines
    # its own. "eval" is SWE-bench Verified (spec section 8: an evaluation
    # set, never trained on); "train" is SWE-smith, this package's training
    # corpus.
    #
    # A real risk, and worth naming even though the fix below is not what
    # was first tried: now that "train" exists, the likeliest operator error
    # is wiring `[[orchestrator.train.source]]` and forgetting to say which
    # split, which would default to "eval" and train on the evaluation set
    # silently, at full speed. A field with NO default was tried first, to
    # force every caller to declare intent -- and reverted, because it does
    # not just add friction, it breaks the CLI outright: verified directly,
    # `uv run validate reliquary-swe ...` (any taskset-first CLI --
    # `resolve.py`'s `narrow_taskset_config`) constructs
    # `SweTasksetConfig(id="reliquary-swe")` with no other fields as a
    # bootstrapping step *before* CLI overrides are applied, and a required
    # field makes that step itself raise `pydantic.ValidationError`,
    # regardless of what `--taskset.split` the actual invocation passes.
    # Every `TasksetConfig` subclass in this repository needs a workable
    # default on every field for exactly this reason -- confirmed by the
    # identical pattern already in `reliquary_code.taskset.CodeConfig.split:
    # Literal["train"] = "train"`, which still carries a default despite
    # having only one legal value. The real protection against the
    # named risk is structural and already in place: `rl.toml` ships with
    # no `[[orchestrator.train.source]]` at all (`TrainSource` itself
    # raises with none), so nothing trains by default regardless of this
    # field's value, and wiring a real source is a single, deliberate,
    # documented action (see `examples/prime_rl/rl.toml`'s own comment) --
    # not something this field can fail-safe on its own without breaking
    # every taskset-first CLI command for this package.
    split: Literal["eval", "train"] = "eval"
    # Only read when split="train". Together with `max_test_count`, this
    # determines the exact task set (spec section 8: "declared, never
    # auto-detected"); see `corpus.load_swesmith_rows` for why a fixed pair
    # is what lets two machines agree on what task #400 is.
    num_images: int = corpus.DEFAULT_SWESMITH_IMAGES
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


# Every phase of a rollout defaults to no limit at all (`TimeoutConfig`'s and
# `TaskTimeout`'s own fields are all `None`), and this is the one environment
# that hands a policy a general shell with `max_turns` bounding turns, not
# wall clock. Three of the four below are sized past a real measurement,
# never guessed; `setup` is the exception -- see its own comment -- and is
# sized by analogy with the other three instead. The measurements
# themselves, and the headroom reasoning, live in the package README's
# "Timeouts" section rather than here.

# `setup()` (`_CLEANUP` above) plus the harness's own setup -- `rollout.py`
# computes one setup-stage deadline and wraps both `task.setup` and
# `harness.setup(runtime)` in it ("Task setup and harness provisioning
# share one setup-stage deadline"), so this budget has to cover both terms,
# and only the first one is actually measured. `_CLEANUP` itself: on the
# container host, the heaviest sampled repository (matplotlib, 296 MB
# `.git`, the largest of six families checked) runs the whole cleanup
# script, `git gc --prune=now` included, in 3.6s. The harness's setup, for
# the `bash` harness this branch's example pins, is a genuine per-rollout
# network install inside the fresh container: `pip install -q -U --user
# uv` (falling back to `apt-get install curl` plus a curl-fetched
# installer) followed by `uv sync --script`, which fetches a managed
# CPython and the script's dependencies -- egress is still open at this
# point, it closes only after setup, and there is no cross-rollout cache
# (`_uv_interpreters` lives on the per-rollout Runtime). That second term
# is the one that dominates this budget, and it has not been measured --
# 900s is sized by analogy with the other three phases below, not derived
# from it. If a pilot hits this ceiling anyway, `--env.agent.timeout.setup`
# overrides the task value at run time without a code change.
_SETUP_TIMEOUT_SECONDS = 900.0

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
# 573 MB novel file costs 35.2s combined -- roughly 60s/GB. 900s covers
# ~15 GB of agent-authored content, far past any legitimate edit, while
# still bounding a disk-filling pathology to a fixed ceiling.
_FINALIZE_TIMEOUT_SECONDS = 900.0

# `env.py`'s `_grade` wraps provisioning-through-grading in
# `asyncio.timeout(task.data.timeout.scoring)`; left at `TaskData`'s default
# (`None`) this is unbounded, so a reachable-but-HANGING box would never
# raise and never score (see task-5-report.md's Important finding). 1800s
# (30 minutes) is sized past the measured tail with headroom, not guessed --
# full reasoning, the measured times it is checked against, and the p90/max
# corpus figures (independently re-measured, not just quoted) live in the
# package README's "Timeouts" section rather than here.
_SCORING_TIMEOUT_SECONDS = 1800.0


class SweTaskset(vf.Taskset[SweTask, SweTasksetConfig]):
    def load(self) -> Iterator[SweTask]:
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
        else:
            rows = corpus.load_rows(self.config.split)
        for index, row in enumerate(rows):
            yield SweTask(
                SweData(
                    idx=index,
                    name=row.instance_id,
                    prompt=PROMPT.format(
                        workdir=WORKDIR, problem_statement=row.problem_statement
                    ),
                    # Given directly for SWE-smith (row.image); derived for
                    # SWE-bench Verified, whose rows carry none (spec section
                    # 8: "the image is given, not derived" -- Verified's own
                    # derivation is the fallback, not the rule).
                    image=row.image if row.image is not None else swe_adapter.image_for(row),
                    workdir=WORKDIR,
                    network_allow=[],
                    timeout=vf.TaskTimeout(
                        setup=_SETUP_TIMEOUT_SECONDS,
                        agent=_AGENT_TIMEOUT_SECONDS,
                        finalize=_FINALIZE_TIMEOUT_SECONDS,
                        scoring=_SCORING_TIMEOUT_SECONDS,
                    ),
                    instance_id=row.instance_id,
                    repo=row.repo,
                    base_commit=row.base_commit,
                    version=row.version,
                    fail_to_pass=row.fail_to_pass,
                    pass_to_pass=row.pass_to_pass,
                    gold_patch=row.gold_patch,
                    test_patch=row.test_patch,
                    split=self.config.split,
                ),
                self.config.task,
            )


__all__ = ["SweData", "SweTask", "SweTasksetConfig", "SweTaskset"]
