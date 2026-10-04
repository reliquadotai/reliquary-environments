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
    first = containers.record("vf-aaa", guard=False)
    second = containers.record("vf-bbb", guard=False)
    assert first == second and first.parent == _ledger_dir
    ledger = containers.Ledger.read(first)
    assert ledger.names == ("vf-aaa", "vf-bbb")
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
    state.write_text("vf-dead1\nvf-dead2\nvf-live\nsomeone-elses\n")
    me = containers.Owner.current()
    dead = _ledger_of(containers.Owner(**{**me.__dict__, "pid": _dead_pid(), "start": "7"}), _ledger_dir, ["vf-dead1", "vf-gone", "vf-dead2"])
    live = _ledger_of(me, _ledger_dir, ["vf-live"])
    elsewhere = _ledger_of(containers.Owner(**{**me.__dict__, "host": "other-host", "pid": 1, "start": "1"}), _ledger_dir, ["someone-elses"])
    assert sorted(containers.reap()) == ["vf-dead1", "vf-dead2"]
    assert state.read_text().split() == ["vf-live", "someone-elses"]
    assert not dead.exists()
    assert live.exists() and elsewhere.exists()


def test_a_ledger_is_kept_when_docker_cannot_be_asked(_ledger_dir, tmp_path, monkeypatch):
    bin_dir = tmp_path / "broken"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text("#!/bin/sh\necho 'Cannot connect to the Docker daemon' >&2\nexit 1\n")
    (bin_dir / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    me = containers.Owner.current()
    dead = _ledger_of(containers.Owner(**{**me.__dict__, "pid": _dead_pid(), "start": "7"}), _ledger_dir, ["vf-x"])
    assert containers.reap() == []
    assert dead.exists()


def test_the_guardian_removes_a_killed_owners_containers(_ledger_dir, docker):
    # The owner records two containers, starting its guardian, then dies by
    # SIGKILL -- which runs no finally, no atexit, nothing of Python's.
    state, log = docker
    state.write_text("vf-one\nvf-two\nunrelated\n")
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import sys, time
                from pathlib import Path
                from reliquary_terminal import containers
                containers.record("vf-one", Path({str(_ledger_dir)!r}))
                containers.record("vf-two", Path({str(_ledger_dir)!r}))
                print("ready", flush=True)
                time.sleep(600)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert owner.stdout.readline().strip() == "ready"
    assert state.read_text().split() == ["vf-one", "vf-two", "unrelated"]
    owner.send_signal(signal.SIGKILL)
    owner.wait()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and state.read_text().split() != ["unrelated"]:
        time.sleep(0.2)
    assert state.read_text().split() == ["unrelated"]
    assert "rm --force vf-one vf-two" in log.read_text()
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
