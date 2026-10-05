"""The in-box helper, run against a temporary root instead of `/`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reliquary_terminal import tmax_box

HELPER = Path(tmax_box.__file__)


def _write(root: Path, path: str, content: str, mode: int = 0o644) -> Path:
    target = root / path.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    target.chmod(mode)
    return target


def test_agent_setup_deletes_only_the_hidden_inputs(tmp_path):
    _write(tmp_path, "/home/user/.truth.json", "42")
    _write(tmp_path, "/home/user/data.csv", "a,b")
    tmax_box.hide(["/home/user/.truth.json", "/home/user/missing"], root=str(tmp_path))
    assert not (tmp_path / "home/user/.truth.json").exists()
    assert (tmp_path / "home/user/data.csv").read_text() == "a,b"


def test_relay_restores_edited_deleted_and_replaced_inputs(tmp_path):
    _write(tmp_path, "/home/user/.truth.json", "42", 0o600)
    _write(tmp_path, "/app/oracle", "#!/bin/sh\necho ok\n", 0o755)
    _write(tmp_path, "/app/expected/out.txt", "right")
    protected = ["/home/user/.truth.json", "/app/oracle", "/app/expected/out.txt"]
    tmax_box.stash(protected, root=str(tmp_path))
    # What an agent's restored artifacts could look like.
    (tmp_path / "home/user/.truth.json").write_text("forged")
    (tmp_path / "app/oracle").unlink()
    (tmp_path / "app/expected/out.txt").unlink()
    (tmp_path / "app/expected/out.txt").mkdir()
    tmax_box.relay(root=str(tmp_path))
    assert (tmp_path / "home/user/.truth.json").read_text() == "42"
    assert (tmp_path / "app/oracle").read_text() == "#!/bin/sh\necho ok\n"
    assert os.stat(tmp_path / "app/oracle").st_mode & 0o777 == 0o755
    assert (tmp_path / "app/expected/out.txt").read_text() == "right"


def test_relay_does_not_write_through_a_symlinked_input(tmp_path):
    # The attack: point the protected `.expected` at the agent's own output,
    # so that restoring it writes the answer into the output.
    _write(tmp_path, "/home/user/.expected", "answer")
    tmax_box.stash(["/home/user/.expected"], root=str(tmp_path))
    output = _write(tmp_path, "/home/user/output.txt", "wrong")
    (tmp_path / "home/user/.expected").unlink()
    (tmp_path / "home/user/.expected").symlink_to(output)
    tmax_box.relay(root=str(tmp_path))
    assert output.read_text() == "wrong"
    assert not (tmp_path / "home/user/.expected").is_symlink()
    assert (tmp_path / "home/user/.expected").read_text() == "answer"


def test_relay_does_not_write_through_a_symlinked_parent(tmp_path):
    _write(tmp_path, "/home/user/ref/truth.txt", "answer")
    tmax_box.stash(["/home/user/ref/truth.txt"], root=str(tmp_path))
    elsewhere = tmp_path / "home/user/mine"
    elsewhere.mkdir()
    import shutil

    shutil.rmtree(tmp_path / "home/user/ref")
    (tmp_path / "home/user/ref").symlink_to(elsewhere)
    tmax_box.relay(root=str(tmp_path))
    assert not (tmp_path / "home/user/ref").is_symlink()
    assert (tmp_path / "home/user/ref/truth.txt").read_text() == "answer"
    assert list(elsewhere.iterdir()) == []


def test_relay_without_a_stash_fails_closed(tmp_path):
    with pytest.raises(SystemExit):
        tmax_box.relay(root=str(tmp_path))


def test_stash_refuses_an_input_setup_did_not_create(tmp_path):
    with pytest.raises(SystemExit, match="not a regular file"):
        tmax_box.stash(["/home/user/.truth.json"], root=str(tmp_path))


def test_the_stash_lives_outside_both_artifact_roots():
    from reliquary_terminal.tmax import ARTIFACT_ROOTS

    stash = "/" + tmax_box.STASH
    assert not any(stash == root or stash.startswith(root + "/") for root in ARTIFACT_ROOTS)


def test_the_helper_runs_as_a_script_under_python_3_10_syntax(tmp_path):
    # The box runs it with Ubuntu 22.04's python3 (3.10): it must compile
    # without 3.11+ syntax. `compile` with feature_version checks the grammar.
    source = HELPER.read_text()
    compile(source, str(HELPER), "exec", flags=0, dont_inherit=True, _feature_version=10)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"protected": [], "hidden": []}))
    run = subprocess.run([sys.executable, str(HELPER), "bogus"], capture_output=True, text=True, check=False)
    assert run.returncode != 0 and "usage" in run.stderr
