"""In-box helper for TMax tasks (docs/tmax.md, decision 3).

This module is copied into each box and run by `python3` from the base image
(Ubuntu 22.04's Python 3.10), so it uses only the standard library and syntax
3.10 accepts. It has three commands:

- `setup agent <inputs.json>`: deletes the task's *hidden* inputs (answers
  and oracles the instruction never names), so the agent cannot read or copy
  them.
- `setup grade <inputs.json>`: stashes the task's *protected* inputs, as its
  own setup just generated them, under `/var/lib/reliquary-tmax`. That is
  outside both artifact roots, and only the grading box has it.
- `relay`: run by `test.sh` after the agent's `/app` and `/home/user` have
  been restored. It puts every stashed input back. A symlink, file or
  directory in the way, at the path or at any parent, is replaced rather than
  followed, so a symlinked input cannot redirect the restore into the
  agent's own output. It exits non-zero if the stash is missing.

`root` lets the tests run all of this under a temporary directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys

STASH = "var/lib/reliquary-tmax"


def _under(root: str, path: str) -> str:
    return os.path.join(root, path.lstrip("/"))


def _inputs(path: str) -> dict:
    with open(path) as handle:
        data = json.load(handle)
    return {"protected": list(data.get("protected", [])), "hidden": list(data.get("hidden", []))}


def hide(hidden: list, root: str = "/") -> None:
    for path in hidden:
        target = _under(root, path)
        if os.path.islink(target) or os.path.isfile(target):
            os.unlink(target)
        elif os.path.isdir(target):
            shutil.rmtree(target)


def stash(protected: list, root: str = "/") -> None:
    base = _under(root, STASH)
    if os.path.lexists(base):
        shutil.rmtree(base)
    os.makedirs(os.path.join(base, "files"), mode=0o700)
    entries = []
    for i, path in enumerate(sorted(protected)):
        source = _under(root, path)
        if os.path.islink(source) or not os.path.isfile(source):
            raise SystemExit(f"reliquary-tmax: protected input {path} is not a regular file after setup")
        with open(source, "rb") as handle:
            content = handle.read()
        blob = os.path.join(base, "files", str(i))
        with open(blob, "wb") as handle:
            handle.write(content)
        entries.append(
            {
                "path": path,
                "blob": str(i),
                "mode": stat.S_IMODE(os.stat(source).st_mode),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    with open(os.path.join(base, "manifest.json"), "w") as handle:
        json.dump(entries, handle)
    # Written last: its presence means the stash is complete.
    with open(os.path.join(base, "done"), "w") as handle:
        handle.write("ok\n")


def _real_directory(path: str) -> None:
    """Make `path` a real directory, replacing whatever non-directory or
    symlink is there."""
    if os.path.islink(path) or (os.path.lexists(path) and not os.path.isdir(path)):
        os.unlink(path)
    if not os.path.isdir(path):
        os.mkdir(path)


def relay(root: str = "/") -> None:
    base = _under(root, STASH)
    if not os.path.isfile(os.path.join(base, "done")):
        raise SystemExit("reliquary-tmax: no stash of protected inputs in this box")
    with open(os.path.join(base, "manifest.json")) as handle:
        entries = json.load(handle)
    for entry in entries:
        parts = [p for p in entry["path"].split("/") if p]
        current = root
        for part in parts[:-1]:
            current = os.path.join(current, part)
            _real_directory(current)
        target = os.path.join(current, parts[-1])
        if os.path.islink(target) or (os.path.lexists(target) and not os.path.isdir(target)):
            os.unlink(target)
        elif os.path.isdir(target):
            shutil.rmtree(target)
        with open(os.path.join(base, "files", entry["blob"]), "rb") as handle:
            content = handle.read()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise SystemExit(f"reliquary-tmax: stash of {entry['path']} is corrupt")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(target, flags, entry["mode"])
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.chmod(target, entry["mode"])


def main(argv: list) -> None:
    if len(argv) >= 3 and argv[0] == "setup" and argv[1] in ("agent", "grade"):
        inputs = _inputs(argv[2])
        if argv[1] == "agent":
            hide(inputs["hidden"])
        else:
            stash(inputs["protected"])
        return
    if argv == ["relay"]:
        relay()
        return
    raise SystemExit("usage: tmax_box.py setup agent|grade <inputs.json> | relay")


if __name__ == "__main__":
    main(sys.argv[1:])
