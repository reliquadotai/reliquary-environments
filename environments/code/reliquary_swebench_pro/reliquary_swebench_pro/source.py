"""The vendored SWE-bench Pro V2 tasks, read only through their pinned digests.

`tasks/<id>/` holds upstream's Harbor files at `_source.COMMIT`, unchanged, plus
`instance.json` (the dataset row's ids and base commit at `_source.DATASET_REVISION`). A
bridged env's identity digests the package's `.py` files only, so every vendored file is
bound to it through `_source.FILES` (its sha256): a file is read only when its bytes
match, and a task with a file missing or changed is refused. Task ids come from that
table, not from the disk, so the task list is the same on every machine.
"""

from __future__ import annotations

import functools
import hashlib
import json
import tomllib
from pathlib import Path

from reliquary_swebench_pro import _source

TASKS = Path(__file__).resolve().parent / "tasks"
UPSTREAM = _source.UPSTREAM
COMMIT = _source.COMMIT
DATASET_REVISION = _source.DATASET_REVISION
TESTS = ("config.json", "parser.py", "run_script.sh", "test.sh", "test_patch.patch")
"""What grading stages as `/tests` (upstream's `tests/`, whole)."""


class TaskFileMismatch(ValueError):
    """A vendored file is missing, or its bytes are not the pinned ones."""


@functools.cache
def task_ids() -> tuple[str, ...]:
    """Every vendored task id (the dataset's `instance_id`), sorted (642)."""
    return tuple(sorted({rel.split("/", 1)[0] for rel in _source.FILES}))


def files_of(task_id: str) -> tuple[str, ...]:
    """The task's vendored files, relative to its directory, sorted."""
    prefix = f"{task_id}/"
    return tuple(sorted(rel[len(prefix):] for rel in _source.FILES if rel.startswith(prefix)))


@functools.lru_cache(maxsize=4096)
def read(task_id: str, rel: str) -> bytes:
    """The bytes of `tasks/<task_id>/<rel>`, checked against the pinned sha256."""
    key = f"{task_id}/{rel}"
    expected = _source.FILES.get(key)
    if expected is None:
        raise TaskFileMismatch(f"{key} is not a vendored file")
    try:
        data = (TASKS / key).read_bytes()
    except OSError as exc:
        raise TaskFileMismatch(f"{key}: {exc}") from exc
    if hashlib.sha256(data).hexdigest() != expected:
        raise TaskFileMismatch(f"{key} does not have its pinned sha256")
    return data


@functools.cache
def task_toml(task_id: str) -> dict:
    return tomllib.loads(read(task_id, "task.toml").decode())


@functools.cache
def instance(task_id: str) -> dict:
    """The dataset row's `instance_id`, `repo`, `base_commit`, `repo_language`, `hard`,
    `docker_image`."""
    return json.loads(read(task_id, "instance.json"))


def instruction(task_id: str) -> str:
    return read(task_id, "instruction.md").decode().strip()


@functools.cache
def config(task_id: str) -> dict:
    """`tests/config.json`: `fail_to_pass`, `pass_to_pass`, `selected_test_files_to_run`."""
    return json.loads(read(task_id, "tests/config.json"))


def gold_patch(task_id: str) -> bytes:
    """The reference solution (`solution/gold_patch.diff`): never used for grading."""
    return read(task_id, "solution/gold_patch.diff")


def tests_files(task_id: str) -> dict[str, bytes]:
    return {name: read(task_id, f"tests/{name}") for name in TESTS}


__all__ = ["COMMIT", "DATASET_REVISION", "TASKS", "TESTS", "UPSTREAM", "TaskFileMismatch",
           "config", "files_of", "gold_patch", "instance", "instruction", "read", "task_ids",
           "task_toml", "tests_files"]
