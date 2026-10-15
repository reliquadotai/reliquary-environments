"""Reading pinned Hub datasets without keeping them: column reads over HTTP
for small columns, and one file downloaded, read and deleted for the large
ones (open-r1's generated tests, Code-Contests-Plus parts), so a build never
holds more than one such file on disk."""

from __future__ import annotations

import contextlib
import fnmatch
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

import pyarrow.parquet as pq

T = TypeVar("T")
RETRIES = 4
BLOCK_SIZE = 16 << 20


def retry(action: Callable[[], T], what: str) -> T:
    for attempt in range(RETRIES):
        try:
            return action()
        except Exception as error:  # network errors come in many types
            if attempt == RETRIES - 1:
                raise RuntimeError(f"{what}: {error}") from error
            time.sleep(5 * 2**attempt)
    raise AssertionError("unreachable")


def list_files(repository: str, revision: str, pattern: str) -> list[str]:
    from huggingface_hub import HfApi

    names = retry(lambda: HfApi().list_repo_files(repository, repo_type="dataset", revision=revision), repository)
    return sorted(name for name in names if fnmatch.fnmatch(name, pattern))


def remote_parquet(repository: str, revision: str, path: str) -> pq.ParquetFile:
    """A ParquetFile read by HTTP ranges: only the row groups and columns asked
    for travel."""
    from huggingface_hub import HfFileSystem

    handle = HfFileSystem().open(f"datasets/{repository}@{revision}/{path}", "rb", block_size=BLOCK_SIZE)
    return pq.ParquetFile(handle)


@contextlib.contextmanager
def downloaded(repository: str, revision: str, path: str, scratch: Path) -> Iterator[Path]:
    """One file of the dataset on local disk for the time of the block, then
    deleted."""
    from huggingface_hub import hf_hub_download

    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    local = Path(retry(
        lambda: hf_hub_download(repository, path, repo_type="dataset", revision=revision, local_dir=scratch),
        f"{repository}/{path}",
    ))
    try:
        yield local
    finally:
        local.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            for meta in (scratch / ".cache").rglob("*"):
                if meta.is_file() and meta.name.startswith(Path(path).name):
                    meta.unlink()
