"""Every Docker container an episode creates is removed, even when the process
that created it is killed.

verifiers already removes a rollout's container in its `finally` and from an
`atexit` hook, which covers a clean exit and a first Ctrl-C or SIGTERM. It
cannot cover a process that dies without running Python: the eval's worker
pool SIGKILLs any worker still tearing down 10 s after an interrupt
(`EnvServerPool._shutdown`), and SIGKILL, an OOM kill or a lost host session
run nothing at all. Its containers are named after the rollout's trace id
(32 hex; a grading box `vf-<12 hex>`) and carry no label, so what one leaves
behind cannot be told apart from anyone else's.

So each process that provisions a box for this package keeps a ledger -- one
file under `LEDGER_DIR` naming the process (host, boot, pid, start time) and
every container it set up -- and starts one guardian: a detached process, in
its own session (outside the process group `timeout` and Ctrl-C signal), that
waits for the owner to die and then removes what the ledger names. A ledger
whose owner and guardian both died (a reboot, a cgroup-wide kill) is reaped
the next time this package's environment starts on that host, or by hand:

    python -m reliquary_terminal.containers list
    python -m reliquary_terminal.containers reap

Only names a ledger of this package recorded, shaped like a verifiers
container name, are ever removed; only ledgers written by this user, on this
host, in this pid namespace are ever reaped. Liveness errs towards "alive":
a process that cannot read its own start time or pid namespace (no `/proc`,
e.g. off Linux) records nothing and starts no guardian, rather than one that
would judge every owner dead.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

LEDGER_DIR = Path(
    os.environ.get(
        "RELIQUARY_TERMINAL_LEDGER_DIR",
        Path.home() / ".cache" / "reliquary-terminal" / "containers",
    )
)
_SUFFIX = ".ledger"
_POLL_SECONDS = 1.0
# Read by `docker` to pick its daemon: a guardian or a later reaper must talk
# to the daemon that created the containers, and only to it.
_DOCKER_ENV = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")
# A rollout's box is named after its trace id; a grading box is verifiers'
# default runtime name. Nothing else is ever removed.
_NAME = re.compile(r"^[0-9a-f]{32}$|^vf-[0-9a-f]{12}$")


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def _pid_namespace() -> str | None:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def _start_time(pid: int) -> str | None:
    """The process's start time in clock ticks since boot, which a reused pid
    does not share; None when it cannot be read (no such process, or no
    `/proc`)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name (field 2) may hold spaces and parentheses; fields after
    # its closing parenthesis are fixed. starttime is field 22.
    return stat.rsplit(")", 1)[1].split()[19]


@dataclass(frozen=True)
class Owner:
    host: str
    boot: str
    pidns: str
    pid: int
    start: str
    docker_env: dict

    @classmethod
    def current(cls) -> Owner | None:
        """This process, or None when it cannot be identified well enough for
        anyone to tell later whether it is alive."""
        pid = os.getpid()
        boot, pidns, start = _boot_id(), _pid_namespace(), _start_time(pid)
        if boot is None or pidns is None or start is None:
            return None
        return cls(
            host=socket.gethostname(),
            boot=boot,
            pidns=pidns,
            pid=pid,
            start=start,
            docker_env={k: os.environ[k] for k in _DOCKER_ENV if k in os.environ},
        )

    def local(self) -> bool:
        """Whether this process can judge the owner: same host, same pid
        namespace (pids mean nothing across one)."""
        return self.host == socket.gethostname() and self.pidns == _pid_namespace()

    def alive(self) -> bool:
        """Dead only when that is certain: another boot, or no process with
        this pid and start time in our own pid namespace. Anything this
        process cannot check reads as alive."""
        boot = _boot_id()
        if boot is not None and boot != self.boot:
            return False
        if not self.local() or _start_time(os.getpid()) is None:
            return True
        return _start_time(self.pid) == self.start


@dataclass(frozen=True)
class Ledger:
    path: Path
    owner: Owner
    names: tuple[str, ...]

    @classmethod
    def read(cls, path: Path) -> Ledger | None:
        try:
            header, *names = path.read_text().splitlines()
            owner = Owner(**json.loads(header))
        except (OSError, ValueError, TypeError):
            return None
        return cls(path, owner, tuple(dict.fromkeys(n for n in names if _NAME.match(n))))

    def ours(self) -> bool:
        """Written by this user, on this host, in this pid namespace."""
        try:
            uid = self.path.stat().st_uid
        except OSError:
            return False
        return uid == os.geteuid() and self.owner.local()


_mine: dict[int, Path | None] = {}  # pid -> this process's ledger (a fork gets its own)


def ledger_path(owner: Owner, root: Path | None = None) -> Path:
    return (root or LEDGER_DIR) / f"{owner.host}-{owner.pid}-{owner.start}{_SUFFIX}"


def record(name: str, root: Path | None = None, guard: bool = True) -> Path | None:
    """Note that this process set up container `name`; on the first call,
    create the ledger and start its guardian. None, with one warning per
    process, when this process cannot be identified (see `Owner.current`)."""
    pid = os.getpid()
    if pid in _mine and _mine[pid] is None:
        return None
    path = _mine.get(pid)
    if path is None:
        owner = Owner.current()
        if owner is None:
            _mine[pid] = None
            logger.warning(
                "reliquary-terminal: cannot read this process's start time or pid "
                "namespace (no /proc?); its containers are not guarded against "
                "it being killed"
            )
            return None
        root = root or LEDGER_DIR
        root.mkdir(parents=True, exist_ok=True)
        path = ledger_path(owner, root)
        path.write_text(json.dumps(owner.__dict__) + "\n")
        _mine[pid] = path
        if guard:
            start_guardian(path)
    with path.open("a") as ledger:
        ledger.write(name + "\n")
    return path


def start_guardian(path: Path) -> subprocess.Popen:
    # This file run as a script, not `-m reliquary_terminal.containers`: that
    # would import the package, and with it verifiers and harbor -- about 4 s
    # of CPU per guardian, measured on sandbox-dev-01. This module needs only
    # the standard library; `-I` keeps the environment and the script's own
    # directory out of its imports.
    return subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).resolve()), "guard", str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def _docker_env(owner: Owner) -> dict:
    """This environment with its own Docker selection replaced by the owner's."""
    base = {k: v for k, v in os.environ.items() if k not in _DOCKER_ENV}
    return {**base, **owner.docker_env}


def remove(ledger: Ledger) -> list[str] | None:
    """`docker rm --force` every container the ledger names that still exists,
    then delete the ledger. The removed names, or None when Docker could not
    be asked (the ledger is then kept for a later attempt)."""
    env = _docker_env(ledger.owner)
    if ledger.names:
        try:
            listed = subprocess.run(
                ["docker", "ps", "--all", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=60, env=env, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if listed.returncode != 0:
            return None
        present = [n for n in ledger.names if n in set(listed.stdout.split())]
    else:
        present = []
    removed: list[str] = []
    if present:
        try:
            done = subprocess.run(
                ["docker", "rm", "--force", *present],
                capture_output=True, text=True, timeout=300, env=env, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        removed = [n for n in done.stdout.split() if n in present]
        if done.returncode != 0 and len(removed) < len(present):
            return None
    ledger.path.unlink(missing_ok=True)
    return removed


def guard(path: Path) -> None:
    """Wait for the ledger's owner to die, then remove its containers."""
    ledger = Ledger.read(path)
    if ledger is None or not ledger.ours():
        return
    while ledger.owner.alive():
        time.sleep(_POLL_SECONDS)
    # Re-read: the owner kept appending until it died.
    ledger = Ledger.read(path)
    if ledger is not None:
        removed = remove(ledger)
        if removed:
            print(f"removed {len(removed)} orphan container(s): {' '.join(removed)}")


def ledgers(root: Path | None = None) -> list[Ledger]:
    """This user's ledgers on this host and in this pid namespace; any other
    is never read further, let alone acted on."""
    found = (Ledger.read(p) for p in sorted((root or LEDGER_DIR).glob(f"*{_SUFFIX}")))
    return [ledger for ledger in found if ledger is not None and ledger.ours()]


def reap(root: Path | None = None) -> list[str]:
    """Remove the containers of every ledger of ours whose owner is dead.
    Best-effort: never raises; returns the names it removed."""
    removed: list[str] = []
    try:
        for ledger in ledgers(root):
            if not ledger.owner.alive():
                removed += remove(ledger) or []
    except Exception:  # noqa: BLE001 - cleanup must never stop an environment from starting
        logger.warning("reliquary-terminal: reaping orphan containers failed", exc_info=True)
    if removed:
        logger.warning(
            "reliquary-terminal: removed %d container(s) left by dead processes: %s",
            len(removed), " ".join(removed),
        )
    return removed


def _main(argv: list[str]) -> int:
    command = argv[0] if argv else "list"
    if command == "guard" and len(argv) == 2:
        guard(Path(argv[1]))
        return 0
    if command == "reap":
        for name in reap():
            print(name)
        return 0
    if command == "list":
        for ledger in ledgers():
            state = "alive" if ledger.owner.alive() else "dead"
            print(f"{ledger.path}  owner pid {ledger.owner.pid} ({state})")
            for name in ledger.names:
                print(f"  {name}")
        return 0
    print("usage: python -m reliquary_terminal.containers [list | reap | guard <ledger>]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
