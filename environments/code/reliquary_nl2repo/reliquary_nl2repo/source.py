"""NL2Repo-Bench's 104 tasks, read only through their pinned digests.

`tasks/<id>/` holds upstream's four files per task at `_source.COMMIT`, fetched by
`scripts/vendor_tasks.py` (upstream has no license: they are never committed). A bridged
env's identity digests the package's `.py` files only, so every file is bound to it
through `_source.FILES`: a file is read only when its bytes match, and a task with a file
missing or changed is refused. Task ids come from that table, not from the disk.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

from reliquary_nl2repo import _source

TASKS = Path(__file__).resolve().parent / "tasks"
UPSTREAM = _source.UPSTREAM
COMMIT = _source.COMMIT
IMAGE_REPOSITORY = "ghcr.io/multimodal-art-projection/nl2repobench"


class TaskFileMismatch(ValueError):
    """A task file is missing, or its bytes are not the pinned ones."""


@functools.cache
def task_ids() -> tuple[str, ...]:
    """Every task id (upstream's directory name), sorted (104)."""
    return tuple(sorted({rel.split("/", 1)[0] for rel in _source.FILES}))


@functools.lru_cache(maxsize=512)
def read(task_id: str, name: str) -> bytes:
    """The bytes of `tasks/<task_id>/<name>`, checked against the pinned sha256."""
    key = f"{task_id}/{name}"
    expected = _source.FILES.get(key)
    if expected is None:
        raise TaskFileMismatch(f"{key} is not a pinned file")
    try:
        data = (TASKS / key).read_bytes()
    except OSError as exc:
        raise TaskFileMismatch(f"{key}: {exc} (run scripts/vendor_tasks.py)") from exc
    if hashlib.sha256(data).hexdigest() != expected:
        raise TaskFileMismatch(f"{key} does not have its pinned sha256")
    return data


def spec(task_id: str) -> bytes:
    """`start.md`: the specification the agent finds in its workspace."""
    return read(task_id, "start.md")


def spec_sha256(task_id: str) -> str:
    return _source.FILES[f"{task_id}/start.md"]


def test_commands(task_id: str) -> list[str]:
    commands = json.loads(read(task_id, "test_commands.json"))
    if not (isinstance(commands, list) and commands and all(isinstance(c, str) for c in commands)):
        raise TaskFileMismatch(f"{task_id}: test_commands.json is not a list of strings")
    return commands


def test_files(task_id: str) -> list[str]:
    paths = json.loads(read(task_id, "test_files.json"))
    if not (isinstance(paths, list) and all(isinstance(p, str) and p for p in paths)):
        raise TaskFileMismatch(f"{task_id}: test_files.json is not a list of paths")
    return paths


def test_case_count(task_id: str) -> int:
    count = int(read(task_id, "test_case_count.txt").decode().strip())
    if count <= 0:
        raise TaskFileMismatch(f"{task_id}: test_case_count.txt is not positive")
    return count


def image(task_id: str) -> str:
    """Upstream's image tag (lower-cased: `more-Itertools` is `more-itertools`)."""
    return f"{IMAGE_REPOSITORY}/{task_id.lower()}:1.0"


__all__ = ["COMMIT", "IMAGE_REPOSITORY", "TASKS", "UPSTREAM", "TaskFileMismatch", "image",
           "read", "spec", "spec_sha256", "task_ids", "test_case_count", "test_commands",
           "test_files"]
