"""Container ledgers and their guardian, without Docker: a fake `docker` on
PATH keeps a list of "containers" in a file."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from reliquary_terminal import containers

FAKE_DOCKER = r"""#!/bin/sh
echo "$*" >> "$FAKE_DOCKER_LOG"
case "$1" in
  ps) cat "$FAKE_DOCKER_STATE" ;;
  rm) shift 2
      for n in "$@"; do
        if grep -qx "$n" "$FAKE_DOCKER_STATE"; then
          echo "$n"
          grep -vx "$n" "$FAKE_DOCKER_STATE" > "$FAKE_DOCKER_STATE.tmp"
          mv "$FAKE_DOCKER_STATE.tmp" "$FAKE_DOCKER_STATE"
        fi
      done ;;
esac
"""


@pytest.fixture
def docker(tmp_path, monkeypatch):
    """A fake `docker` whose daemon holds `state`'s lines as containers."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(FAKE_DOCKER)
    (bin_dir / "docker").chmod(0o755)
    state = tmp_path / "state"
    state.write_text("")
    log = tmp_path / "docker.log"
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state))
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    return state, log


def _dead_pid() -> int:
    child = subprocess.Popen(["true"])
    child.wait()
    return child.pid


def test_record_writes_one_ledger_per_process(_ledger_dir):
    first = containers.record("vf-00000000000a", guard=False)
    second = containers.record("vf-00000000000b", guard=False)
    assert first == second and first.parent == _ledger_dir
    ledger = containers.Ledger.read(first)
    assert ledger.names == ("vf-00000000000a", "vf-00000000000b")
    assert ledger.owner.pid == os.getpid()
    assert ledger.owner.alive()


def test_a_dead_or_reused_pid_is_not_the_owner():
    me = containers.Owner.current()
    assert me.alive()
    assert not containers.Owner(**{**me.__dict__, "pid": _dead_pid()}).alive()
    # Same pid, other start time: a reused pid.
    assert not containers.Owner(**{**me.__dict__, "start": "1"}).alive()
    # Same pid and start time, another boot.
    assert not containers.Owner(**{**me.__dict__, "boot": "other"}).alive()


def _ledger_of(owner: containers.Owner, root: Path, names: list[str]) -> Path:
    import json

    root.mkdir(parents=True, exist_ok=True)
    path = containers.ledger_path(owner, root)
    path.write_text(json.dumps(owner.__dict__) + "\n" + "".join(n + "\n" for n in names))
    return path


def test_reap_removes_only_what_dead_owners_recorded(_ledger_dir, docker):
    state, _ = docker
    state.write_text("vf-0000000dead1\nvf-0000000dead2\nvf-0000000000ee\ncccccccccccccccccccccccccccccccc\n")
    me = containers.Owner.current()
    dead = _ledger_of(containers.Owner(**{**me.__dict__, "pid": _dead_pid(), "start": "7"}), _ledger_dir, ["vf-0000000dead1", "vf-00000000ffff", "vf-0000000dead2"])
    live = _ledger_of(me, _ledger_dir, ["vf-0000000000ee"])
    elsewhere = _ledger_of(containers.Owner(**{**me.__dict__, "host": "other-host", "pid": 1, "start": "1"}), _ledger_dir, ["cccccccccccccccccccccccccccccccc"])
    assert sorted(containers.reap()) == ["vf-0000000dead1", "vf-0000000dead2"]
    assert state.read_text().split() == ["vf-0000000000ee", "cccccccccccccccccccccccccccccccc"]
    assert not dead.exists()
    assert live.exists() and elsewhere.exists()


def test_a_ledger_is_kept_when_docker_cannot_be_asked(_ledger_dir, tmp_path, monkeypatch):
    bin_dir = tmp_path / "broken"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text("#!/bin/sh\necho 'Cannot connect to the Docker daemon' >&2\nexit 1\n")
    (bin_dir / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    me = containers.Owner.current()
    dead = _ledger_of(containers.Owner(**{**me.__dict__, "pid": _dead_pid(), "start": "7"}), _ledger_dir, ["vf-000000000001"])
    assert containers.reap() == []
    assert dead.exists()


def test_the_guardian_removes_a_killed_owners_containers(_ledger_dir, docker):
    # The owner records two containers, starting its guardian, then dies by
    # SIGKILL -- which runs no finally, no atexit, nothing of Python's.
    state, log = docker
    state.write_text("11111111111111111111111111111111\n22222222222222222222222222222222\nunrelated\n")
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import sys, time
                from pathlib import Path
                from reliquary_terminal import containers
                containers.record("11111111111111111111111111111111", Path({str(_ledger_dir)!r}))
                containers.record("22222222222222222222222222222222", Path({str(_ledger_dir)!r}))
                print("ready", flush=True)
                time.sleep(600)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert owner.stdout.readline().strip() == "ready"
    assert state.read_text().split() == ["11111111111111111111111111111111", "22222222222222222222222222222222", "unrelated"]
    owner.send_signal(signal.SIGKILL)
    owner.wait()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and state.read_text().split() != ["unrelated"]:
        time.sleep(0.2)
    assert state.read_text().split() == ["unrelated"]
    assert "rm --force 11111111111111111111111111111111 22222222222222222222222222222222" in log.read_text()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and list(_ledger_dir.iterdir()):
        time.sleep(0.1)
    assert list(_ledger_dir.iterdir()) == []


def test_the_guardian_imports_nothing_but_the_standard_library():
    # It runs once per process for as long as that process lives; importing
    # verifiers cost about 4 s of CPU each on sandbox-dev-01.
    probe = (
        "import runpy, sys; sys.argv = ['containers.py', 'list']; "
        f"runpy.run_path({containers.__file__!r}, run_name='__main__')"
    )
    out = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", probe], capture_output=True, text=True, check=False
    )
    imported = out.stderr
    assert "verifiers" not in imported and "reliquary_terminal" not in imported


def _me(**changes) -> containers.Owner:
    return containers.Owner(**{**containers.Owner.current().__dict__, **changes})


def test_names_not_shaped_like_verifiers_boxes_are_never_removed(_ledger_dir, docker):
    state, log = docker
    state.write_text("postgres\nreliquary-gradebox-1\n" + "a" * 32 + "\n")
    _ledger_of(_me(pid=_dead_pid(), start="7"), _ledger_dir, ["postgres", "reliquary-gradebox-1", "a" * 32, "vf-XYZ"])
    assert containers.reap() == ["a" * 32]
    assert state.read_text().split() == ["postgres", "reliquary-gradebox-1"]


def test_ledgers_of_another_pid_namespace_are_never_touched(_ledger_dir, docker):
    # Same host, same boot: a container sharing ~/.cache and the docker
    # socket. Its pids mean nothing here, so its owner must not be judged.
    state, log = docker
    name = "d" * 32
    state.write_text(name + "\n")
    other = _ledger_of(_me(pidns="pid:[1]", pid=_dead_pid(), start="7"), _ledger_dir, [name])
    assert containers.ledgers() == []
    assert containers.reap() == []
    assert other.exists() and state.read_text().split() == [name]
    assert not log.exists()  # docker never even asked
    # Judged from here, such an owner can only read as alive.
    assert containers.Ledger.read(other).owner.alive()


def test_ledgers_of_another_user_are_never_touched(_ledger_dir, docker, monkeypatch):
    state, _ = docker
    name = "e" * 32
    state.write_text(name + "\n")
    _ledger_of(_me(pid=_dead_pid(), start="7"), _ledger_dir, [name])
    monkeypatch.setattr(containers.os, "geteuid", lambda: os.geteuid() + 1)
    assert containers.reap() == []
    assert state.read_text().split() == [name]


def test_without_proc_nothing_is_recorded_and_no_guardian_starts(_ledger_dir, monkeypatch, caplog):
    # Off Linux every owner would read as dead, and a guardian would remove
    # the box ~100 ms after setup. Cannot tell means: do nothing.
    monkeypatch.setattr(containers, "_start_time", lambda pid: None)
    started = []
    monkeypatch.setattr(containers, "start_guardian", lambda path: started.append(path))
    assert containers.Owner.current() is None
    assert containers.record("f" * 32) is None
    assert containers.record("0" * 32) is None
    assert started == [] and not _ledger_dir.exists()
    assert sum("not guarded" in r.message for r in caplog.records) == 1


def test_an_owner_that_cannot_be_checked_reads_as_alive(monkeypatch):
    me = containers.Owner.current()
    dead = _me(pid=_dead_pid())
    monkeypatch.setattr(containers, "_start_time", lambda pid: None)
    assert dead.alive() is True
    monkeypatch.setattr(containers, "_pid_namespace", lambda: None)
    assert me.alive() is True


def test_the_owners_docker_selection_replaces_ours(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://elsewhere:2375")
    monkeypatch.setenv("DOCKER_CONTEXT", "other")
    env = containers._docker_env(_me(docker_env={"DOCKER_HOST": "unix:///owner.sock"}))
    assert env["DOCKER_HOST"] == "unix:///owner.sock"
    assert "DOCKER_CONTEXT" not in env
    assert containers._docker_env(_me(docker_env={})).get("DOCKER_HOST") is None
