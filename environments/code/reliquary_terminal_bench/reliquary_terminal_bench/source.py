"""The vendored Terminal-Bench 2.1 tasks, read only through their pinned digests.

`tasks/<id>/` holds Harbor's files for the pinned dataset digest, unchanged (task.toml,
instruction.md, environment/Dockerfile, tests/, solution/). A bridged env's identity
digests the package's `.py` files only, so every vendored file is bound to it through
`_source.FILES`: a file is read only when its bytes match. Task ids come from that table,
not from the disk, so the task list is the same on every machine.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import io
import re
import tarfile
import tomllib
from pathlib import Path

from reliquary_terminal_bench import _source

TASKS = Path(__file__).resolve().parent / "tasks"
DATASET = _source.DATASET
DIGEST = _source.DIGEST
_WORKDIR = re.compile(r"^\s*WORKDIR\s+(\S+)\s*$", re.MULTILINE | re.IGNORECASE)


class TaskFileMismatch(ValueError):
    """A vendored file is missing, or its bytes are not the pinned ones."""


@functools.cache
def task_ids() -> tuple[str, ...]:
    """Every vendored task id, sorted (89)."""
    return tuple(sorted({rel.split("/", 1)[0] for rel in _source.FILES}))


def files_of(task_id: str, prefix: str = "") -> tuple[str, ...]:
    """The task's vendored files under `prefix`, relative to its directory, sorted."""
    head = f"{task_id}/"
    return tuple(sorted(rel[len(head):] for rel in _source.FILES
                        if rel.startswith(head + prefix)))


@functools.lru_cache(maxsize=2048)
def read(task_id: str, rel: str) -> bytes:
    """The bytes of `tasks/<task_id>/<rel>`, checked against the pinned sha256."""
    key = f"{task_id}/{rel}"
    expected = _source.FILES.get(key)
    if expected is None:
        raise TaskFileMismatch(f"{key} is not a vendored file")
    try:
        if key in _source.GZIPPED:
            # Vendored compressed (fake credentials secret scanners refuse in plain text).
            data = gzip.decompress((TASKS / (key + ".gz")).read_bytes())
        else:
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
    return read(task_id, "instruction.md").decode()


@functools.cache
def workdir(task_id: str) -> str:
    """The image's own WORKDIR: the last one of the Dockerfile that built it. None of the
    89 `task.toml` declares one, and three tasks work outside `/app` (fix-git,
    sanitize-git-repo, prove-plus-comm): their reference scores 0 elsewhere."""
    found = _WORKDIR.findall(read(task_id, "environment/Dockerfile").decode())
    if not found:
        raise TaskFileMismatch(f"{task_id}: its Dockerfile sets no WORKDIR")
    return found[-1]


def _tar(task_id: str, prefix: str) -> bytes:
    """A deterministic gzip tar of the task's `prefix` files, relative to `prefix` (root
    owner, fixed mtime; modes 0755/0644 as vendored)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT,
                      compresslevel=6) as tar:
        for rel in files_of(task_id, prefix):
            data = read(task_id, rel)
            info = tarfile.TarInfo(rel[len(prefix):])
            info.size = len(data)
            path = TASKS / task_id / rel
            if f"{task_id}/{rel}" in _source.GZIPPED:
                path = path.with_name(path.name + ".gz")
            info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    raw = buffer.getvalue()
    # gzip's header carries an mtime: zero it so the bytes are the same on every machine.
    return raw[:4] + b"\0\0\0\0" + raw[8:]


@functools.cache
def tests_archive(task_id: str) -> bytes:
    """`tests/`, staged at `/tests` in the grading box."""
    return _tar(task_id, "tests/")


@functools.cache
def solution_archive(task_id: str) -> bytes:
    """`solution/` (the oracle, `solve.sh`): references and the gold check only."""
    return _tar(task_id, "solution/")


def tests_sha256(task_id: str) -> str:
    return hashlib.sha256(b"".join(
        f"{rel}\0{_source.FILES[f'{task_id}/{rel}']}\n".encode()
        for rel in files_of(task_id, "tests/"))).hexdigest()


__all__ = ["DATASET", "DIGEST", "TASKS", "TaskFileMismatch", "files_of", "instruction",
           "read", "solution_archive", "task_ids", "task_toml", "tests_archive",
           "tests_sha256", "workdir"]
