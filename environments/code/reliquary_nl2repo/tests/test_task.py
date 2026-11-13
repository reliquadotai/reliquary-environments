"""The task's hooks: the box scripts against temporary directories, the reward against a
scripted box."""

from __future__ import annotations

import asyncio
import io
import json
import os
import pickle
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary_nl2repo import hygiene, source, submission, task as task_module
from reliquary_nl2repo.task import ARCHIVE_PATH, FINALIZE_SCRIPT, INSTRUCTION, plan_command
from reliquary_nl2repo.task import test_script as build_script
from reliquary_nl2repo.taskset import NL2RepoConfig, NL2RepoTaskset


def a_task(task_id="tinydb"):
    taskset = NL2RepoTaskset(NL2RepoConfig(split="eval_all", id="reliquary-nl2repo",
                                           tasks=[task_id]))
    return next(iter(taskset.load()))


# --- commands -------------------------------------------------------------------

@pytest.mark.parametrize("command, text, is_pytest", [
    ("set AWS_DEFAULT_REGION=us-east-1", "export AWS_DEFAULT_REGION=us-east-1", False),
    ("pip install -e .", "pip install --no-build-isolation --no-deps -e .", False),
    ("pip install  -e .", "pip install --no-build-isolation --no-deps  -e .", False),
    ("python -m pip install x", "python -m pip install --no-build-isolation --no-deps x", False),
    ("pytest --continue-on-collection-errors tests",
     "pytest --continue-on-collection-errors tests --junitxml=/tmp/nl2repo-junit-3.xml", True),
    ("xvfb-run -a pytest tests", "xvfb-run -a pytest tests --junitxml=/tmp/nl2repo-junit-3.xml",
     True),
    ("pip install pytest", "pip install --no-build-isolation --no-deps pytest", False),
    ("touch README.md && pip install -e .",
     "touch README.md && pip install --no-build-isolation --no-deps -e .", False),
])
def test_plan_command(command, text, is_pytest):
    assert plan_command(command, 3) == (text, is_pytest)


def test_every_task_has_a_pytest_command():
    for task_id in source.task_ids():
        commands = source.test_commands(task_id)
        assert any(plan_command(c, i)[1] for i, c in enumerate(commands)), task_id


def test_the_script_runs_in_order_with_exports_and_a_budget(tmp_path):
    commands = ["echo one > a.txt", "set ZZ_VAR=it works", "echo \"$ZZ_VAR\" > b.txt",
                "sleep 30", "echo never > c.txt"]
    script = build_script(commands, budget_s=8, workdir=str(tmp_path))
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "a.txt").read_text() == "one\n"
    assert (tmp_path / "b.txt").read_text() == "it works\n"
    assert not (tmp_path / "c.txt").exists()
    assert Path("/tmp/nl2repo-cmd-3.exit").read_text().strip() == "124"
    assert Path("/tmp/nl2repo-cmd-4.exit").read_text().strip() == "skipped"


def test_summary_passed_counts_like_upstream_but_not_xpassed():
    log = "== 10 passed, 2 failed in 1.0s ==\n== 3 passed, 1 xpassed in 0.1s ==\n"
    assert task_module.summary_passed(log) == 13


# --- finalize ---------------------------------------------------------------------

def finalize(tmp_path, max_bytes=10**7):
    out = tmp_path / "logs" / "artifacts" / "workspace.tar.gz"
    done = subprocess.run(["sh", "-c", FINALIZE_SCRIPT, "t", str(out), str(max_bytes),
                           str(tmp_path / "ws")], capture_output=True, text=True)
    return done.returncode, out


def members(path):
    with tarfile.open(path) as tar:
        return sorted(m.name.removeprefix("./") for m in tar.getmembers())


@pytest.fixture
def ws(tmp_path):
    work = tmp_path / "ws"
    for rel in ("pkg/__init__.py", "pkg/build/keep.py", "README.md", ".git/config",
                "pkg/__pycache__/x.pyc", ".venv/lib/a.py", "build/lib/a.py", "dist/a.whl",
                "pkg.egg-info/PKG-INFO", "node_modules/x/index.js"):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text("x")
    os.symlink("/etc/hostname", work / "link")
    return tmp_path


def test_finalize_archives_regular_files_without_toolchain_dirs(ws):
    code, out = finalize(ws)
    assert code == 0
    assert members(out) == ["README.md", "pkg/__init__.py", "pkg/build/keep.py"]


def test_finalize_clears_the_artifact_directory_first(ws):
    art = ws / "logs" / "artifacts"
    art.mkdir(parents=True)
    (art / "workspace.tar.gz").write_text("planted")
    (art / "other").write_text("planted")
    code, out = finalize(ws)
    assert code == 0 and sorted(p.name for p in art.iterdir()) == ["workspace.tar.gz"]


def test_finalize_replaces_a_linked_artifact_directory(ws):
    (ws / "logs").mkdir()
    (ws / "elsewhere").mkdir()
    os.symlink(ws / "elsewhere", ws / "logs" / "artifacts")
    code, out = finalize(ws)
    assert code == 0 and not (ws / "logs" / "artifacts").is_symlink()
    assert list((ws / "elsewhere").iterdir()) == []


def test_finalize_leaves_no_archive_over_the_cap_or_without_a_workspace(ws, tmp_path):
    code, out = finalize(ws, max_bytes=10)
    assert code == 6 and not out.exists()
    other = tmp_path / "other"
    other.mkdir()
    assert finalize(other)[0] == 3


def test_an_empty_workspace_is_an_empty_archive(tmp_path):
    (tmp_path / "ws").mkdir()
    code, out = finalize(tmp_path)
    assert code == 0 and members(out) == []


# --- hygiene ----------------------------------------------------------------------

def fake_site(tmp_path):
    site = tmp_path / "lib" / "site-packages"
    site.mkdir(parents=True)
    for name, direct in (("zzlocal", {"url": "file:///project", "dir_info": {}}),
                         ("zzeditable", {"url": "file:///p2", "dir_info": {"editable": True}}),
                         ("zzwheel", {"url": "file:///w/zz.whl", "archive_info": {}})):
        info = site / f"{name}-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\n\n")
        (info / "direct_url.json").write_text(json.dumps(direct))
        (site / f"{name}.py").write_text("x = 1\n")
        (info / "RECORD").write_text(f"{name}.py,,\n{info.name}/METADATA,,\n")
    (site / "zzdevelop.egg-link").write_text("/gone/zzdevelop\n.")
    (site / "easy-install.pth").write_text("/gone/zzdevelop\n/kept/elsewhere\n")
    return site


def test_uninstall_removes_local_directory_installs_and_develop_links(tmp_path):
    site = fake_site(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(site)}
    # -S and no getsitepackages(): only the fake site is seen, never this interpreter's.
    isolated = "import site; site.getsitepackages = lambda: []; exec(SRC)".replace(
        "SRC", repr(hygiene.UNINSTALL_PY))
    done = subprocess.run([sys.executable, "-S", "-c", isolated], env=env,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    left = sorted(p.name for p in site.iterdir())
    assert left == ["easy-install.pth", "zzwheel-1.0.dist-info", "zzwheel.py"]
    assert (site / "easy-install.pth").read_text() == "/kept/elsewhere\n"
    report = json.loads(done.stdout)
    assert set(report["removed"]) == {"zzlocal", "zzeditable", "zzdevelop.egg-link"}


def test_uninstall_script_passes_the_source_as_argv0(tmp_path):
    site = fake_site(tmp_path)
    shim = tmp_path / "bin"
    shim.mkdir()
    # -I drops PYTHONPATH: the shim re-adds the fake site through a .pth-free wrapper.
    # The shim drops `-I -c` and runs the source with only the fake site visible.
    (shim / "python3").write_text(f"#!/bin/sh\nshift 2\nexec {sys.executable} -S -c \"import sys, site; "
                                  f"site.getsitepackages = lambda: []; "
                                  f"sys.path.insert(0, '{site}'); exec(sys.argv[1])\" \"$@\"\n")
    (shim / "python3").chmod(0o755)
    done = subprocess.run(["sh", "-c", hygiene.UNINSTALL_SCRIPT, hygiene.UNINSTALL_PY],
                          env={**os.environ, "PATH": f"{shim}:{os.environ['PATH']}"},
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (site / "zzlocal.py").exists()


def test_remove_script(tmp_path):
    (tmp_path / "copy" / "pkg").mkdir(parents=True)
    done = subprocess.run(["sh", "-c", hygiene.REMOVE_SCRIPT, "r", str(tmp_path / "copy"),
                           str(tmp_path / "absent")], capture_output=True)
    assert done.returncode == 0 and not (tmp_path / "copy").exists()
    assert subprocess.run(["sh", "-c", hygiene.REMOVE_SCRIPT, "r", "relative"]).returncode == 2


def test_reference_paths_are_absolute_and_name_known_tasks():
    for task_id, paths in hygiene.REFERENCE_PATHS.items():
        assert task_id in source.task_ids()
        assert all(p.startswith("/") and p not in ("/", "/workspace", "/usr") for p in paths)


# --- data -------------------------------------------------------------------------

def test_task_data():
    task = a_task("asteval")
    data = task.data
    assert task.key == "asteval" and data.prompt == INSTRUCTION
    assert data.reference_paths == ["/project"]
    assert data.image == "ghcr.io/multimodal-art-projection/nl2repobench/asteval:1.0"
    assert data.network_allow == [] and data.workdir == "/workspace"
    assert [a.source for a in data.artifacts] == [ARCHIVE_PATH]
    assert data.timeout.scoring == 810.0 and data.timeout.setup <= 600
    assert data.test_case_count == 227 and data.spec_sha256 == source.spec_sha256("asteval")
    pickle.dumps(task)


def test_the_hash_is_stable_and_binds_the_spec_and_commands():
    task = a_task()
    assert task.hash == a_task().hash
    changed = task.data.model_copy(update={"spec_sha256": "0" * 64})
    assert type(task)(changed).hash != task.hash
    changed = task.data.model_copy(update={"test_commands": ["pytest"]})
    assert type(task)(changed).hash != task.hash


# --- reward on a scripted box -----------------------------------------------------

def gz(files):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def junit(passed, failed=0):
    cases = "".join('<testcase name="p%d"/>' % i for i in range(passed))
    cases += "".join('<testcase name="f%d"><failure/></testcase>' % i for i in range(failed))
    return f"<testsuites><testsuite>{cases}</testsuite></testsuites>".encode()


class ScriptedBox:
    config = SimpleNamespace(type="reliquary-sandbox")

    def __init__(self, files):
        self.files = dict(files)
        self.runs = []

    async def run(self, argv, env):
        self.runs.append(argv)
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def read(self, path, max_bytes=None):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write(self, path, data):
        self.files[path] = data


def grade(task, files):
    box = ScriptedBox(files)
    trace = SimpleNamespace(info={}, metrics={})
    trace.record_metrics = trace.metrics.update
    reward = asyncio.run(task.pass_rate(box, trace))
    return reward, trace, box


def test_reward_is_the_capped_pass_rate_from_the_reports():
    task = a_task("tinydb")  # pip install -e . then pytest: the report is command 1's
    count = task.data.test_case_count
    reward, trace, box = grade(task, {ARCHIVE_PATH: gz({"tinydb/__init__.py": b""}),
                                      "/tmp/nl2repo-junit-1.xml": junit(count // 2, 3)})
    assert reward == (count // 2) / count
    assert trace.metrics["passed"] == count // 2 and trace.metrics["reported"] == count // 2 + 3
    clean = box.files[task_module.CLEAN_ARCHIVE]
    with tarfile.open(fileobj=io.BytesIO(clean)) as tar:
        assert tar.getnames() == ["tinydb/__init__.py"]
    reward, _, _ = grade(task, {ARCHIVE_PATH: gz({}),
                                "/tmp/nl2repo-junit-1.xml": junit(count + 50)})
    assert reward == 1.0


def test_a_missing_report_or_archive():
    task = a_task("tinydb")
    reward, trace, box = grade(task, {})
    assert reward == 0.0 and trace.metrics["empty_workspace"] == 1.0
    assert trace.metrics["report_errors"] == 1.0
    assert any("bash" == argv[0] for argv in box.runs)  # the tests still ran


def test_a_forged_report_with_entities_counts_nothing():
    task = a_task("tinydb")
    forged = b'<!DOCTYPE t [<!ENTITY x "y">]>' + junit(10)
    reward, trace, _ = grade(task, {ARCHIVE_PATH: gz({}), "/tmp/nl2repo-junit-1.xml": forged})
    assert reward == 0.0 and trace.metrics["report_errors"] == 1.0


def test_an_unreadable_archive_is_zero_without_running_anything():
    task = a_task("tinydb")
    reward, trace, box = grade(task, {ARCHIVE_PATH: b"garbage"})
    assert reward == 0.0 and trace.metrics["archive_refused"] == 1.0 and box.runs == []


def test_own_modules_reach_the_task():
    assert a_task("pytest-cov").data.own_modules == ["pytest_cov"]
    assert a_task("tinydb").data.own_modules == []
