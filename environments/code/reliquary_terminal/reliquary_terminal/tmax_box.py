"""In-box helper for TMax tasks (docs/tmax.md, decision 3).

This module is copied into each box and run by `python3` from the base image
(Ubuntu 22.04's Python 3.10), so it uses only the standard library and syntax
3.10 accepts. It has four commands:

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
- `run-tests`: run by `test.sh` after `relay`, as root. Hands the agent's
  restored trees to an unprivileged uid (`TEST_UID`), runs the final-state
  test as that uid under `python3 -I`, with a HOME of its own outside the
  roots and PYTHONNOUSERSITE, kills every process of that uid, then writes
  the verdict as root: 1 only for a clean exit with every process gone and a
  report. Nothing the agent's code starts during the test can write
  `/logs/verifier`, which stays root's.

`root` lets the tests run all of this under a temporary directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import sys
import tempfile
import time

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


TEST_UID = 61000
ROOTS = ("/app", "/home/user")
VERIFIER_DIR = "logs/verifier"
MAX_CTRF_BYTES = 8 * 1024 * 1024
FINAL_TEST = "test_final_state.py"
PYTHON = "/usr/bin/python3"
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def hand_over(roots: list, uid: int, root: str = "/", lchown=os.lchown, chmod=os.chmod) -> None:
    """Give every entry under each existing root (the root included) to `uid`: the link
    itself, never its target, and a linked root is not entered. Directories get u+rwx
    (before they are listed), other files u+rw, and u+x where any x bit was set; set-id
    bits are dropped. A device node goes to root with mode 0. A file with more links than
    the roots hold shares its inode with the rest of the box and is left alone."""
    found = []
    seen = {}
    for top in roots:
        pending = [_under(root, top)]
        if not os.path.lexists(pending[0]):
            continue
        while pending:
            path = pending.pop()
            info = os.lstat(path)
            if stat.S_ISDIR(info.st_mode):
                lchown(path, uid, uid)
                chmod(path, stat.S_IMODE(info.st_mode) | stat.S_IRWXU)
                pending += [entry.path for entry in os.scandir(path)]
                continue
            found.append((path, info))
            key = (info.st_dev, info.st_ino)
            seen[key] = seen.get(key, 0) + 1
    for path, info in found:
        if info.st_nlink > seen[(info.st_dev, info.st_ino)]:
            continue
        if stat.S_ISCHR(info.st_mode) or stat.S_ISBLK(info.st_mode):
            lchown(path, 0, 0)
            chmod(path, 0)
            continue
        lchown(path, uid, uid)
        if stat.S_ISLNK(info.st_mode):
            continue
        mode = stat.S_IMODE(info.st_mode) & ~(stat.S_ISUID | stat.S_ISGID)
        extra = stat.S_IRUSR | stat.S_IWUSR | (stat.S_IXUSR if mode & 0o111 else 0)
        chmod(path, mode | extra)


def test_argv(report: str) -> list:
    return [PYTHON, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c",
            "/tests/pytest.ini", "--rootdir=/tests", "--confcutdir=/tests", "--ctrf", report,
            "-rA", "/tests/" + FINAL_TEST]


def test_env(environ, home: str) -> dict:
    """The environment the grading command was given (its PATH fixed, its loader
    variables cleared), with a HOME of the test's own outside the roots, and no user site
    for any Python the test starts."""
    env = dict(environ)
    env["HOME"] = home
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _drop(uid: int) -> None:
    os.setgroups([])
    os.setgid(uid)
    os.setuid(uid)


def run_as(uid: int, argv: list, env: dict, cwd: str = "/tests") -> int:
    pid = os.fork()
    if pid == 0:
        try:
            _drop(uid)
            os.chdir(cwd)
            os.execve(argv[0], argv, env)
        finally:
            os._exit(127)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


def processes_of(uid: int, proc: str = "/proc") -> list:
    """The live processes whose real or effective uid is `uid` (zombies cannot run)."""
    found = []
    for name in os.listdir(proc):
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(proc, name, "status")) as handle:
                fields = dict(line.split(":", 1) for line in handle if ":" in line)
        except OSError:
            continue
        if fields.get("State", "").split()[:1] in (["Z"], ["X"]):
            continue
        if str(uid) in fields.get("Uid", "").split()[0:2]:
            found.append(int(name))
    return sorted(found)


def reap(uid: int, proc: str = "/proc", rounds: int = 50) -> bool:
    """Kill every process of `uid`, from a child dropped to `uid` (`kill(-1)` reaches
    exactly them), until none is left. False if some survived."""
    for _ in range(rounds):
        if not processes_of(uid, proc):
            return True
        pid = os.fork()
        if pid == 0:
            try:
                _drop(uid)
                os.kill(-1, signal.SIGKILL)
            finally:
                os._exit(0)
        os.waitpid(pid, 0)
        time.sleep(0.05)
    return not processes_of(uid, proc)


def _remove(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.unlink(path)


def _verifier_dir(path: str) -> None:
    """`path` as a real directory of this process's user, mode 0755."""
    if os.path.lexists(path):
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
            _remove(path)
    os.makedirs(path, exist_ok=True)
    os.chmod(path, 0o755)


def _write(path: str, data: bytes) -> None:
    partial = path + ".partial"
    _remove(partial)
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(partial, path)


def _report(path: str):
    """The report's bytes: a regular file, not a link, at most MAX_CTRF_BYTES; else None.
    Opened non-blocking, so a FIFO left in its place cannot hang the grader."""
    try:
        fd = os.open(path, os.O_RDONLY | _NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return None
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CTRF_BYTES:
            return None
        data = handle.read(MAX_CTRF_BYTES + 1)
    return data if len(data) <= MAX_CTRF_BYTES else None


def finish(rc: int, report: str, verifier_dir: str, reaped: bool) -> int:
    """Write the verdict: `reward.txt` is 1 only for `rc == 0`, every process reaped and a
    report copied to `ctrf.json`. Returns `rc`, or 1 if a process survived."""
    _verifier_dir(verifier_dir)
    for name in ("reward.json", "ctrf.json"):
        _remove(os.path.join(verifier_dir, name))
    data = _report(report) if reaped else None
    if data is not None:
        _write(os.path.join(verifier_dir, "ctrf.json"), data)
    passed = rc == 0 and reaped and data is not None
    _write(os.path.join(verifier_dir, "reward.txt"), b"1\n" if passed else b"0\n")
    return rc if reaped else 1


def _tests_ready(tests: str, uid: int):
    """Make `/tests` readable by `uid` and writable by its owner only. Returns why it
    cannot be used: not a real directory, or an entry `uid` owns (it could rewrite it)."""
    if not os.path.isdir(tests) or os.path.islink(tests):
        return "/tests is not a directory"
    entries = [tests]
    for dirpath, dirnames, filenames in os.walk(tests):
        entries += [os.path.join(dirpath, name) for name in dirnames + filenames]
    for path in entries:
        if os.lstat(path).st_uid == uid:
            return f"{path} is owned by the test uid {uid}"
    for path in entries:
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            continue
        mode = stat.S_IMODE(info.st_mode) & ~(0o022 | stat.S_ISUID | stat.S_ISGID)
        os.chmod(path, mode | (0o555 if stat.S_ISDIR(info.st_mode) else 0o444))
    return None


def run_tests(uid: int = TEST_UID, root: str = "/") -> int:
    verifier = _under(root, VERIFIER_DIR)
    _verifier_dir(verifier)
    for name in ("reward.txt", "reward.json", "ctrf.json"):
        _remove(os.path.join(verifier, name))
    _write(os.path.join(verifier, "reward.txt"), b"0\n")
    tests = _under(root, "tests")
    refused = _tests_ready(tests, uid)
    if refused is not None:
        print(f"reliquary-tmax: {refused}; scoring 0", file=sys.stderr)
        return 1
    hand_over(list(ROOTS), uid, root)
    out = tempfile.mkdtemp(prefix="reliquary-tests-", dir=_under(root, "tmp"))
    try:
        # `out` stays root's, so the test uid cannot swap `run` or `home` for a link.
        os.chmod(out, 0o711)
        run, home = os.path.join(out, "run"), os.path.join(out, "home")
        for path in (run, home):
            os.mkdir(path, 0o700)
            os.chown(path, uid, uid)
        report = os.path.join(run, "ctrf.json")
        sys.stdout.flush()
        sys.stderr.flush()
        rc = run_as(uid, test_argv(report), test_env(os.environ, home), cwd=tests)
        reaped = reap(uid, _under(root, "proc"))
        return finish(rc, report, verifier, reaped)
    finally:
        shutil.rmtree(out, ignore_errors=True)


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
    if argv == ["run-tests"]:
        raise SystemExit(run_tests())
    raise SystemExit("usage: tmax_box.py setup agent|grade <inputs.json> | relay | run-tests")


if __name__ == "__main__":
    main(sys.argv[1:])
