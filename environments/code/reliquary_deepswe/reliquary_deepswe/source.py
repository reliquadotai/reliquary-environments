"""The vendored DeepSWE v1.1 tasks, read only through their pinned digests.

`tasks/<id>/` holds upstream's files at `_source.COMMIT`, unchanged. A bridged env's
identity digests the package's `.py` files only, so every vendored file is bound to it
through `_source.FILES` (its sha256): a file is read only when its bytes match, and a
task with a file missing or changed is refused. Task ids come from that table, not from
the disk, so the task list is the same on every machine.
"""

from __future__ import annotations

import functools
import hashlib
import tomllib
from pathlib import Path

from reliquary_deepswe import _source

TASKS = Path(__file__).resolve().parent / "tasks"
UPSTREAM = _source.UPSTREAM
COMMIT = _source.COMMIT


class TaskFileMismatch(ValueError):
    """A vendored file is missing, or its bytes are not the pinned ones."""


@functools.cache
def task_ids() -> tuple[str, ...]:
    """Every vendored task id, sorted (113)."""
    return tuple(sorted({rel.split("/", 1)[0] for rel in _source.FILES}))


def files_of(task_id: str) -> tuple[str, ...]:
    """The task's vendored files, relative to its directory, sorted."""
    prefix = f"{task_id}/"
    return tuple(sorted(rel[len(prefix):] for rel in _source.FILES if rel.startswith(prefix)))


@functools.lru_cache(maxsize=1024)
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


def instruction(task_id: str) -> str:
    return read(task_id, "instruction.md").decode().strip()


def gold_patch(task_id: str) -> bytes:
    """The reference solution (`solution/solution.patch`): never used for grading."""
    return read(task_id, "solution/solution.patch")


__all__ = ["COMMIT", "TASKS", "UPSTREAM", "TaskFileMismatch", "files_of", "gold_patch",
           "instruction", "read", "task_ids", "task_toml"]
