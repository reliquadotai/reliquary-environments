"""The agent's workspace archive, made safe in the scoring process, and the test count.

`finalize` archives the agent's `/workspace` in the agent's box, with that box's `tar`:
the bytes are the agent's. `sanitize` reads them with `tarfile` and writes a new archive
of regular files only, every path relative and inside the workspace, so extracting it in
the grading box can create nothing but plain files under `/workspace`. On the way it drops:

* upstream's `remove_package_files` list (the image's packaging wins, as upstream);
* the task's `test_files.json` paths (the image's hidden tests win, as upstream);
* toolchain and cache directories (`.git`, `__pycache__`, virtualenvs, `node_modules`,
  `*.egg-info`, top-level `build/` and `dist/`);
* what could change how the hidden tests run rather than what they test: every
  `conftest.py`, `sitecustomize.py`, `usercustomize.py` and `*.pth`, pytest config
  files, modules shadowing pytest or its plugins (the workspace is on `PYTHONPATH`), and
  files pytest would collect as tests (`test_*.py`, `*_test.py`): upstream's commands
  sometimes run a bare `pytest` over the whole workspace, which would count the agent's
  own tests.

The grading box extracts the clean archive with `--skip-old-files`: a file the image
already has under `/workspace` (hidden tests, upstream packaging) is never replaced.

`count_passed` reads the JUnit reports pytest wrote (strictly: no DTD, size-capped): a
test case passes when it has no failure, error or skipped child.
"""

from __future__ import annotations

import io
import tarfile
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath

MAX_ARCHIVE_BYTES = 24 * 1024 * 1024
"""The archive `finalize` leaves (the bridge carries at most 32 MiB of artifacts)."""
MAX_UNPACKED_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 50_000
MAX_REPORT_BYTES = 32 * 1024 * 1024

# Upstream's `post_processor.remove_package_files` list, plus pytest's other config files.
PACKAGE_FILES = frozenset({
    "setup.py", "pyproject.toml", "setup.cfg", "requirements.txt", "requirements-dev.txt",
    "requirements-test.txt", "tox.ini", "pytest.ini", "poetry.lock", "Pipfile",
    "Pipfile.lock", "environment.yml", "conda-env.yaml", "manifest.in", "MANIFEST.in",
    ".pytest.ini", "pytest.toml", ".pytest.toml",
})
FORCING_FILES = frozenset({"conftest.py", "sitecustomize.py", "usercustomize.py"})
TOOLCHAIN_DIRS = frozenset({
    ".git", "__pycache__", ".venv", "venv", ".env", "node_modules", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".nox", ".eggs", ".hypothesis", ".cache",
})
TOP_LEVEL_DIRS = frozenset({"build", "dist"})
# Import names that would shadow the test runner from /workspace or /workspace/src.
SHADOWED = frozenset({"pytest", "_pytest", "pluggy", "py", "xdist", "iniconfig", "_hypothesis_pytestplugin",
                      "coverage", "pytest_cov", "pytest_asyncio", "pytest_mock", "pytest_randomly",
                      "pytest_timeout", "typeguard", "junitparser"})


def _module_name(part: str) -> str:
    return part[:-3] if part.endswith(".py") else part.split(".", 1)[0]


def drop_reason(path: str, test_paths: tuple[str, ...],
                own_modules: frozenset[str] = frozenset()) -> str | None:
    """Why `path` (relative, normalised) is left out of the graded workspace, or None."""
    parts = PurePosixPath(path).parts
    name = parts[-1]
    for test_path in test_paths:
        if path == test_path or path.startswith(test_path.rstrip("/") + "/"):
            return "test_path"
    if any(p in TOOLCHAIN_DIRS or p.endswith(".egg-info") for p in parts[:-1]):
        return "toolchain_dir"
    if len(parts) > 1 and parts[0] in TOP_LEVEL_DIRS:
        return "toolchain_dir"
    if name in PACKAGE_FILES:
        return "package_file"
    if name in FORCING_FILES or name.endswith(".pth"):
        return "runner_hook"
    module = parts[1:] if parts[0] == "src" and len(parts) > 1 else parts
    top = _module_name(module[0])
    if top not in own_modules and (top in SHADOWED or top.startswith("pytest_")):
        return "shadows_runner"
    if name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py")):
        return "test_file"
    return None


class ArchiveRefused(ValueError):
    """The archive cannot be read as a workspace at all (graded 0)."""


def _normalise(name: str) -> str | None:
    if not name or name.startswith("/") or "\\" in name or "\0" in name:
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


def sanitize(raw: bytes, test_paths: list[str] | tuple[str, ...],
             own_modules: list[str] | tuple[str, ...] = ()) -> tuple[bytes, dict]:
    """A clean, uncompressed tar of the workspace's kept regular files, and what was
    dropped (counts per reason). Raises ArchiveRefused for unreadable or oversized
    input."""
    tests = tuple(p for p in (_normalise(t) for t in test_paths) if p)
    own = frozenset(own_modules)
    dropped: dict[str, int] = {}
    kept: dict[str, tuple[bytes, int]] = {}
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as source:
            for index, member in enumerate(source):
                if index >= MAX_MEMBERS:
                    raise ArchiveRefused(f"more than {MAX_MEMBERS} members")
                if member.isdir():
                    continue
                path = _normalise(member.name)
                reason = None
                if path is None:
                    reason = "unsafe_path"
                elif not member.isreg():
                    reason = "not_a_regular_file"
                else:
                    reason = drop_reason(path, tests, own)
                if reason is not None:
                    dropped[reason] = dropped.get(reason, 0) + 1
                    continue
                total += member.size
                if total > MAX_UNPACKED_BYTES:
                    raise ArchiveRefused(f"more than {MAX_UNPACKED_BYTES} bytes unpacked")
                handle = source.extractfile(member)
                data = handle.read() if handle is not None else b""
                kept[path] = (data, 0o755 if member.mode & 0o111 else 0o644)
    except (tarfile.TarError, EOFError, OSError, ValueError) as exc:
        if isinstance(exc, ArchiveRefused):
            raise
        raise ArchiveRefused(f"{type(exc).__name__}: {str(exc)[:200]}") from None
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.GNU_FORMAT) as clean:
        for path in sorted(kept):
            data, mode = kept[path]
            info = tarfile.TarInfo(path)
            info.size, info.mode, info.mtime = len(data), mode, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            clean.addfile(info, io.BytesIO(data))
    return out.getvalue(), {"kept": len(kept), "dropped": dropped}


def empty_archive() -> bytes:
    """An empty workspace, as `finalize` captures one."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz"):
        pass
    return out.getvalue()


def count_passed(report: bytes) -> tuple[int, int]:
    """(passed, total) test cases of one JUnit report. Raises ValueError when the report
    is not JUnit XML or declares a DTD or entities."""
    if b"<!DOCTYPE" in report[:4096].upper() or b"<!ENTITY" in report.upper():
        raise ValueError("a JUnit report with a DTD or entities is not read")
    root = ET.fromstring(report)
    if root.tag not in ("testsuites", "testsuite"):
        raise ValueError(f"not a JUnit report: <{root.tag}>")
    passed = total = 0
    for case in root.iter("testcase"):
        total += 1
        if not any(child.tag in ("failure", "error", "skipped") for child in case):
            passed += 1
    return passed, total


__all__ = ["ArchiveRefused", "FORCING_FILES", "MAX_ARCHIVE_BYTES", "MAX_REPORT_BYTES",
           "PACKAGE_FILES", "SHADOWED", "TOOLCHAIN_DIRS", "count_passed", "drop_reason",
           "empty_archive", "sanitize"]
