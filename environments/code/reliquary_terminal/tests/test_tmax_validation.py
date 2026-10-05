"""The decisions of the box phase, without a box."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile

import pytest

from reliquary_terminal import tmax_validation as V


def _tar(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def _untar(archive: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def test_snapshot_parsing_handles_odd_names():
    raw = b"/home/user/a b\0" + b"12\0" + b"1700000000.5\0" + b"/app/x\ty\0" + b"0\0" + b"1.0\0"
    assert V.parse_snapshot(raw) == {"/home/user/a b": (12, "1700000000.5"), "/app/x\ty": (0, "1.0")}


def test_protected_inputs_are_setup_made_and_untouched():
    base = {"/usr/lib/python3.10/os.py": (1, "1"), "/etc/hosts": (5, "1")}
    after_setup = {**base, "/etc/hosts": (9, "2"), "/home/user/.truth.json": (2, "3"), "/home/user/in.csv": (4, "3"), "/app/oracle": (7, "3")}
    after_solution = {**after_setup, "/home/user/in.csv": (5, "4"), "/home/user/out.txt": (1, "4")}
    pyc = "/usr/local/lib/python3.10/dist-packages/__pycache__/x.cpython-310-pytest-9.1.1.pyc"
    after_setup[pyc] = after_solution[pyc] = (3, "3")
    candidates = {pyc, "/usr/lib/python3.10/os.py", "/etc/hosts", "/home/user/.truth.json", "/home/user/in.csv", "/home/user/out.txt", "/app/oracle", "/tests/test_final_state.py"}
    assert V.protected_inputs(candidates, base, after_setup, after_solution) == ["/app/oracle", "/etc/hosts", "/home/user/.truth.json"]
    assert V.changed_paths(after_setup, after_solution) == ["/home/user/in.csv", "/home/user/out.txt"]


def test_hidden_inputs_are_the_ones_the_instruction_does_not_name():
    protected = ["/home/user/.truth.json", "/app/data/input.csv", "/app/oracle_parser"]
    instruction = "Read input.csv under /app/data and match our reference oracle binary."
    assert V.hidden_inputs(protected, instruction) == ["/home/user/.truth.json", "/app/oracle_parser"]
    assert V.looks_like_answer("/home/user/.truth.json") and V.looks_like_answer("/app/oracle_parser")
    assert not V.looks_like_answer("/app/data/input.csv")


def test_literal_paths():
    src = 'P = "/home/user/out.txt"\nQ = os.path.join("/app", "x")\nR = f"/home/user/{name}"\nS = "relative/path"\n'
    assert V.literal_paths(src) == {"/home/user/out.txt", "/app"}


def test_mutants():
    archive = _tar({"home/user/out.json": b'{"total": 42, "rows": [1, 2]}\n{"ok": true}\n', "home/user/bin": b"\x7fELF\0\1", "home/user/same.txt": b"keep 1\n"})
    changed = ["/home/user/out.json", "/home/user/bin"]
    zero, n = V.mutate_archive(archive, changed, "zero")
    assert n == 2 and _untar(zero) == {"home/user/out.json": b"", "home/user/bin": b"", "home/user/same.txt": b"keep 1\n"}
    perturbed, n = V.mutate_archive(archive, changed, "perturb")
    files = _untar(perturbed)
    assert n == 1  # the binary is left alone
    assert files["home/user/out.json"] == b'{"total": 43, "rows": [2, 3]}\n'
    assert files["home/user/bin"] == b"\x7fELF\0\1" and files["home/user/same.txt"] == b"keep 1\n"
    assert V.mutate_archive(None, changed, "zero") == (None, 0)
    both, total = V.mutate_artifacts({"/home/user": archive, "/app": None}, changed, "zero")
    assert total == 2 and both["/app"] is None


@pytest.mark.parametrize(
    "checks,expected",
    [
        (V.Checks(setup_ok=False), ["setup_failed"]),
        (V.Checks(setup_ok=True, initial_ok=False), ["initial_state_fails"]),
        (V.Checks(setup_ok=True, deterministic=False), ["setup_nondeterministic"]),
        (V.Checks(noop_rewards=[1.0], solution_rewards=[1, 1, 1]), ["noop_passes"]),
        (V.Checks(noop_rewards=[0.0], artifact_over_cap=True), ["artifact_cap"]),
        (V.Checks(noop_rewards=[0.0], solution_rewards=[0, 0, 0]), ["solution_fails"]),
        (V.Checks(noop_rewards=[0.0], solution_rewards=[1, 0, 1]), ["solution_unstable"]),
        (V.Checks(noop_rewards=[0.0], solution_rewards=[1, 1, 1], mutation_rewards={"zero": 0.0, "perturb": 1.0}), ["mutation_passes"]),
        (V.Checks(noop_rewards=[0.0], solution_rewards=[1, 1, 1], mutation_rewards={"zero": 0.0, "perturb": 0.0}), []),
    ],
)
def test_verdicts(checks, expected):
    assert V.verdict(checks)[0] == expected


def test_hidden_inputs_fall_back_to_visible_unless_they_are_answers():
    kept = V.Checks(noop_rewards=[0.0], solution_rewards=[0, 0, 0], hidden_retry_rewards=[1, 1, 1], hidden=["/app/data/input.csv"], mutation_rewards={"zero": 0.0})
    assert V.verdict(kept) == ([], [])
    answer = V.Checks(noop_rewards=[0.0], solution_rewards=[0, 0, 0], hidden_retry_rewards=[1, 1, 1], hidden=["/home/user/.truth.json"])
    assert V.verdict(answer)[0] == ["answer_needed_by_reference"]
    hidden_ok = V.Checks(noop_rewards=[0.0], solution_rewards=[1, 1, 1], hidden=["/home/user/.truth.json"], mutation_rewards={"zero": 0.0})
    assert V.verdict(hidden_ok) == ([], ["/home/user/.truth.json"])


def test_record_is_what_the_manifest_merges(tmp_path):
    record = V.result_record("t", "img@sha256:" + "0" * 64, 1, ["/a"], V.Checks(noop_rewards=[0.0], solution_rewards=[1, 1, 1], mutation_rewards={"zero": 0.0}), {"setup_seconds": 1.5})
    assert json.loads(json.dumps(record))["reasons"] == []
    assert record["run"] == 1 and record["protected"] == ["/a"]


def test_the_audit_plugin_records_reads_and_subprocess_paths(tmp_path, monkeypatch):
    # Run pytest with the plugin on a test that reads a file and runs a
    # program, in a subprocess so the audit hook stays out of this process.
    monkeypatch.setattr(V, "AUDIT_LOG", str(tmp_path / "audit.json"))
    plugin = V.AUDIT_PLUGIN.replace("/tmp/reliquary-tmax-audit.json", str(tmp_path / "audit.json"))
    (tmp_path / "reliquary_tmax_audit.py").write_text(plugin)
    data = tmp_path / "data.txt"
    data.write_text("x")
    (tmp_path / "test_x.py").write_text(
        "import subprocess\n"
        f"def test_x():\n    open({str(data)!r}).read()\n"
        "    subprocess.run(['/bin/true', '/home/user/thing'])\n"
        "    subprocess.run('cat /etc/hostname > /dev/null', shell=True)\n"
    )
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "reliquary_tmax_audit", "-p", "no:cacheprovider", str(tmp_path / "test_x.py")],
        cwd=tmp_path, env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"}, capture_output=True, text=True, check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    seen = set(json.loads((tmp_path / "audit.json").read_text()))
    assert {str(data), "/bin/true", "/home/user/thing", "/etc/hostname"} <= seen


def test_the_box_script_refuses_to_run_without_the_explicit_opt_in(tmp_path):
    from pathlib import Path

    script = Path(__file__).parent.parent / "scripts" / "tmax_validate.py"
    env = {k: v for k, v in __import__("os").environ.items() if k != "RELIQUARY_TMAX_I_HAVE_A_BOX"}
    run = subprocess.run(
        [sys.executable, str(script), "run", "--image", "x@sha256:" + "0" * 64, "--out", str(tmp_path)],
        capture_output=True, text=True, env=env, check=False,
    )
    assert run.returncode != 0 and "RELIQUARY_TMAX_I_HAVE_A_BOX" in run.stderr
    assert list(tmp_path.iterdir()) == []
