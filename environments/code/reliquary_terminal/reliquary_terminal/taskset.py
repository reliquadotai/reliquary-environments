"""Terminal tasks as one Verifiers taskset, in two splits.

`eval` is Terminal-Bench 2.1 exactly as it ships: graded in the box the agent
worked in, so the number is comparable to every published result. `train` is
MiMo-V2.6-RL-oss's 64 Terminal-Bench-format tasks, graded in a fresh box that
receives only the agent's `/app` -- the discipline those tasks were written
for, and the one a training reward needs. See the design spec
(docs/superpowers/specs/2026-09-23-reliquary-terminal-env-design.md,
section 4, option C) for why the two splits grade differently.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from datasets import load_dataset
from verifiers.v1.task import TaskResources, TaskTimeout
from verifiers.v1.taskset import Taskset
from verifiers.v1.tasksets.harbor.taskset import (
    HarborConfig,
    HarborData,
    VerifierConfig,
    dataset_dir,
    parse_task,
)
from verifiers.v1.utils.artifacts import Artifact

from reliquary_terminal.grading import TerminalTask

# Pinned by content digest: `@latest` is revision 6 today and can move.
EVAL_DATASET = (
    "terminal-bench/terminal-bench-2-1"
    "@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
)

_TRAIN_SOURCE = (
    "XiaomiMiMo/MiMo-V2.6-RL-oss",
    "general",
    "639865fd3374018d6cb29b9fb82dd531406fcf5f",
)
TRAIN_IMAGE_REPOSITORY = "xiaomimimo/mimo-v2.6-rl-oss"
TRAIN_CACHE = Path.home() / ".cache" / "reliquary-terminal" / _TRAIN_SOURCE[2]

# The row declares 120 s for its verifier, but a separate verifier's scoring
# deadline also covers provisioning its box (see `HarborEnv._grade`), which the
# task author's number never included.
_TRAIN_SCORING_TIMEOUT_SECONDS = 600.0

_WORKDIR = re.compile(r"^\s*WORKDIR\s+(\S+)\s*$", re.MULTILINE | re.IGNORECASE)


class TerminalConfig(HarborConfig):
    # No default, for the reason `reliquary-swe`'s own `split` has none: a
    # training source that forgets to say which split must not fall back to
    # the evaluation set.
    split: Literal["eval", "train"] | None = None
    dataset: str = EVAL_DATASET


def image_workdir(task_dir: Path) -> str | None:
    """The last `WORKDIR` of a task's own Dockerfile, when it ships one.

    verifiers' `DockerConfig` defaults `workdir` to `/app`, and that default
    overrides the image's own `WORKDIR` whenever `task.toml` declares none --
    none of Terminal-Bench 2.1's 89 tasks do. For the three whose image works
    elsewhere (`fix-git`, `sanitize-git-repo`, `prove-plus-comm`) even the
    reference solution then scores 0. The Dockerfile is what built the
    published image, so its final `WORKDIR` is the image's.
    """
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return None
    found = _WORKDIR.findall(dockerfile.read_text())
    return found[-1] if found else None


@functools.cache
def load_train_rows() -> tuple[dict, ...]:
    """Every Terminal-Bench-format row of the pinned `general` config (64)."""
    name, config, revision = _TRAIN_SOURCE
    dataset = load_dataset(name, config, split="train", revision=revision)
    rows = []
    for row in dataset:
        instance = json.loads(row["extra_info"]["instance_json"])
        if instance["dataset_type"] == "terminal_bench":
            rows.append(instance)
    return tuple(rows)


GUARD = "anti_hack_guard.py"
GUARD_SHA256 = "336149b10b45d03b7774ba589d984e9663135cfc660f3a133fc482a8716d0e3a"
"""The guard all 64 rows ship, at the pinned revision."""
GUARD_PATCHED_SHA256 = "76b45434a9479164c2577f6a3da1601ad655453aeec1503c6c222f91a92bac9c"
"""The same guard after `_guarded`."""
_GUARD_SKIP = (
    "        if not path.is_file() or path.is_symlink():\n"
    "            continue\n"
)
_GUARD_CHECK = (
    "        # reliquary-terminal: a symlink with a dangerous name is refused like a\n"
    "        # planted file, whatever it points to (the shipped guard skipped links).\n"
    "        if path.is_dir() and not path.is_symlink():\n"
    "            continue\n"
)


def _guarded(source: bytes) -> bytes:
    """The rows' `anti_hack_guard.py` (one file, the same in all 64), refusing a
    planted symlink with a dangerous name as it refuses a planted file: it skipped
    every symlink, so `conftest.py -> hooks.py` loaded a pytest hook unseen."""
    if hashlib.sha256(source).hexdigest() != GUARD_SHA256:
        raise ValueError("anti_hack_guard.py does not have the pinned sha256: not the guard "
                         "this package patches")
    patched = source.decode().replace(_GUARD_SKIP, _GUARD_CHECK).encode()
    if hashlib.sha256(patched).hexdigest() != GUARD_PATCHED_SHA256:
        raise ValueError("the patched anti_hack_guard.py does not have the pinned sha256")
    return patched


def tests_files(row: dict) -> dict[str, bytes]:
    """A row's `tests/` as `{relative path: bytes}`, the guard patched (`_guarded`)."""
    files = row["tests_files"]
    if isinstance(files, str):
        files = json.loads(files)
    out = {}
    for rel, encoded in files.items():
        data = base64.b64decode(encoded)
        out[rel] = _guarded(data) if rel == GUARD else data
    return out


def _matches(task_dir: Path, expected: dict[str, bytes]) -> bool:
    """Whether `task_dir` holds exactly `tests/` with `expected` in it: nothing
    more, nothing less, byte for byte, no links."""
    tests = task_dir / "tests"
    try:
        if [p.name for p in task_dir.iterdir()] != ["tests"]:
            return False
        found = {}
        for path in tests.rglob("*"):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                return False
            if path.is_file():
                found[path.relative_to(tests).as_posix()] = path.read_bytes()
    except OSError:
        return False
    return found == expected


def materialize_tests(row: dict, root: Path = TRAIN_CACHE) -> Path:
    """Write a row's `tests/` to a task directory `HarborTask` can stage from.

    The row carries its tests base64-encoded, so they never live in the
    image the agent works in. An existing directory is used only when it holds
    exactly the row's files (`_matches`); anything else -- an older guard, a
    damaged or tampered cache -- is replaced. Safe across processes: each
    writer stages in its own directory and renames it into place; a writer that
    finds the place taken by a matching directory keeps that one.
    """
    task_dir = root / row["instance_id"]
    expected = tests_files(row)
    if _matches(task_dir, expected):
        return task_dir
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{task_dir.name}.partial-", dir=root))
    try:
        for rel, data in expected.items():
            path = staging / "tests" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for _ in range(8):
            try:
                os.rename(staging, task_dir)
                return task_dir
            except OSError:  # taken (a non-empty directory)
                if _matches(task_dir, expected):
                    return task_dir
                stale = root / f"{task_dir.name}.stale-{uuid.uuid4().hex}"
                try:
                    os.rename(task_dir, stale)
                except FileNotFoundError:
                    continue
                shutil.rmtree(stale, ignore_errors=True)
        raise RuntimeError(f"could not materialize {task_dir}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def train_data(row: dict, idx: int, config: TerminalConfig) -> HarborData:
    image_tag = row["docker_image"].split(":", 1)[0]
    agent_timeout = None if config.ignore_timeouts else row["agent_timeout_sec"]
    resources = TaskResources(
        cpu=row["cpus"] * config.resource_multiplier,
        memory=row["memory_mb"] / 1024 * config.resource_multiplier,
        disk=row["storage_mb"] / 1024 * config.resource_multiplier,
    )
    return HarborData(
        idx=idx,
        name=row["instance_id"],
        description=row.get("display_description"),
        prompt=row["problem_statement"].strip(),
        image=f"{TRAIN_IMAGE_REPOSITORY}:{image_tag}",
        workdir=row["cwd"],
        network_allow=[] if not row.get("allow_internet") else ["*"],
        timeout=TaskTimeout(agent=agent_timeout, scoring=_TRAIN_SCORING_TIMEOUT_SECONDS),
        resources=resources,
        category=row.get("category"),
        tags=list(row.get("tags") or []),
        task_dir=str(materialize_tests(row)),
        # Only the agent's work travels to the grading box. `/app` is the
        # whole of it for these tasks (about 2 MB on the image checked), and
        # the tests' own guard compares it file by file against a manifest of
        # the pristine tree.
        artifacts=[Artifact(source=row["cwd"])],
        # Declared rather than a fresh copy of the agent's box for one reason:
        # the grading box must NOT work in `/app`. `restore()` first deletes
        # every artifact root, then writes each archive through `docker exec
        # --workdir <workdir>` -- and with `/app` both the root just deleted
        # and the workdir, that exec fails (with an empty message) on every
        # rollout. Checked on a real image. These tests address `/app` and
        # `/tests` by absolute path only, so `/` changes nothing they read.
        verifier=VerifierConfig(
            workdir="/",
            resources=resources,
            network_allow=[],
        ),
    )


# Parameterized directly rather than subclassing `HarborTaskset`: verifiers
# resolves a taskset's config type from its generic parameters, so a plain
# subclass would still hand the CLI `HarborConfig`, without `split`.
class TerminalTaskset(Taskset[TerminalTask, TerminalConfig]):
    def load(self) -> Iterator[TerminalTask]:
        if self.config.split is None:
            raise ValueError(
                'reliquary-terminal: --taskset.split is required ("eval" for '
                'Terminal-Bench 2.1, "train" for MiMo-V2.6\'s terminal tasks) -- '
                "it has no default so a training source cannot silently fall "
                "back to the evaluation set"
            )
        if self.config.split == "train":
            for idx, row in enumerate(load_train_rows()):
                if self.config.tasks is None or row["instance_id"] in self.config.tasks:
                    yield TerminalTask(train_data(row, idx, self.config), self.config.task)
            return
        root = dataset_dir(self.config)
        task_dirs = [
            toml.parent
            for toml in sorted(root.rglob("task.toml"))
            if (toml.parent / "instruction.md").is_file()
            and (self.config.tasks is None or toml.parent.name in self.config.tasks)
        ]
        if not task_dirs:
            raise ValueError(f"no terminal-bench tasks found in {root}")
        for idx, task_dir in enumerate(task_dirs):
            data = parse_task(task_dir, idx, self.config)
            if data.workdir is None:
                data = data.model_copy(update={"workdir": image_workdir(task_dir)})
            yield TerminalTask(data, self.config.task)


__all__ = ["EVAL_DATASET", "TerminalConfig", "TerminalTaskset", "image_workdir"]
