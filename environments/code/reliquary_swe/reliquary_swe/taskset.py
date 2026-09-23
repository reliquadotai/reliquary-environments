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

from reliquary_swe import corpus, swe_adapter

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
_CLEANUP = " ; ".join(
    [
        "set -e",
        'git -C /testbed reset --hard "$BASE_COMMIT"',
        'git -C /testbed checkout -q "$BASE_COMMIT"',
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
        result = await runtime.run(
            ["sh", "-c", _CLEANUP], {"BASE_COMMIT": self.data.base_commit}
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
        await vf.capture_patch(
            trace,
            runtime,
            base_commit=self._heads.pop(id(runtime), ""),
            ignore=self._untracked.pop(id(runtime), []),
            write_path=PATCH_PATH,
        )


class SweTasksetConfig(vf.TasksetConfig):
    # `vf.TasksetConfig` carries no `split`; every sibling package that loads
    # more than one split (see `reliquary_code.taskset.CodeConfig`) defines
    # its own. SWE-bench Verified currently ships only "eval"
    # (`corpus.SPLITS`), but the field stays user-configurable rather than
    # baked in, matching that convention.
    split: Literal["eval"] = "eval"


# `env.py`'s `_grade` wraps provisioning-through-grading in
# `asyncio.timeout(task.data.timeout.scoring)`; left at `TaskData`'s default
# (`None`) this is unbounded, so a reachable-but-HANGING box would never
# raise and never score (see task-5-report.md's Important finding). 1800s
# (30 minutes) is sized past the measured tail with headroom, not guessed --
# full reasoning, the measured times it is checked against, and the p90/max
# corpus figures (independently re-measured, not just quoted) live in the
# package README's "Grading timeout" section rather than here.
_SCORING_TIMEOUT_SECONDS = 1800.0


class SweTaskset(vf.Taskset[SweTask, SweTasksetConfig]):
    def load(self) -> Iterator[SweTask]:
        for index, row in enumerate(corpus.load_rows(self.config.split)):
            yield SweTask(
                SweData(
                    idx=index,
                    name=row.instance_id,
                    prompt=PROMPT.format(
                        workdir=WORKDIR, problem_statement=row.problem_statement
                    ),
                    image=swe_adapter.image_for(row),
                    workdir=WORKDIR,
                    network_allow=[],
                    timeout=vf.TaskTimeout(scoring=_SCORING_TIMEOUT_SECONDS),
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
