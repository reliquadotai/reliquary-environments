"""The `tmax` split wired into the taskset and the grading box, on a
synthetic manifest and source. No container: a fake box records what setup
uploads and runs."""

from __future__ import annotations

import io
import json
import tarfile

import pytest
import verifiers.v1 as vf
from test_tmax import make_task
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.tasksets.harbor.taskset import verifier_box_data

from reliquary_terminal import TerminalEnv, taskset, tmax, tmax_select
from reliquary_terminal.grading import TerminalTask

IMAGE = "registry.example/reliquary-tmax-base@sha256:" + "a" * 64
IDS = [f"task_{i:06d}_{i:08x}" for i in range(1, 6)]


@pytest.fixture
def tmax_env(tmp_path, monkeypatch):
    """Five synthetic tasks, four kept, and the cache under tmp_path."""
    src = tmp_path / "src"
    for tid in IDS:
        make_task(src, tid)
    tasks = {tid: {"status": "kept", "run": 0, "protected": ["/home/user/.truth.json"], "hidden": ["/home/user/.truth.json"]} for tid in IDS[:4]}
    tasks[IDS[4]] = {"status": "excluded", "reasons": ["mutation_passes"]}
    manifest = {"base_image": IMAGE, "counts": {"status": {"kept": 4, "excluded": 1}}, "tasks": tasks}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(tmax_select, "MANIFEST", path)
    monkeypatch.setattr(tmax, "CACHE", tmp_path / "cache")
    return src, path


def _config(**kwargs):
    return vf.taskset_config_type("reliquary-terminal")(id="reliquary-terminal", **kwargs)


def _load(src, **kwargs):
    return list(vf.load_taskset(_config(split="tmax", tmax_source=src, **kwargs)))


def test_tmax_yields_the_kept_tasks_in_hash_order(tmax_env, tmp_path):
    src, _ = tmax_env
    tasks = _load(src)
    assert [t.data.name for t in tasks] == sorted(IDS[:4], key=tmax_select.order_key)
    assert [t.data.name for t in _load(src, num_tasks=2)] == [t.data.name for t in tasks[:2]]
    t = tasks[0]
    assert type(t) is TerminalTask
    assert t.data.image == IMAGE
    assert t.data.workdir == "/home/user"
    assert t.data.network_allow == []
    assert t.data.upload_environment is False
    assert [(a.source, a.required) for a in t.data.artifacts] == [("/app", False), ("/home/user", False)]
    assert t.data.verifier.workdir == "/" and t.data.verifier.network_allow == []
    assert t.data.env["LC_ALL"] == "C.UTF-8"
    assert t.data.verifier.env == t.data.env
    assert t.data.prompt == "Put 42 in /home/user/out.txt."
    assert "truth" not in t.data.prompt
    assert t.data.resources.cpu == taskset.TMAX_CPUS
    assert t.data.task_dir.startswith(str(tmp_path / "cache"))
    # The grading box is the same image, in /, with the same environment.
    box = verifier_box_data(t.data)
    assert box.image == IMAGE and box.workdir == "/" and box.env == t.data.env


def test_tmax_refuses_a_manifest_with_nothing_validated(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"base_image": None, "counts": {"status": {"pending": 3}}, "tasks": {"t": {"status": "pending"}}}))
    monkeypatch.setattr(tmax_select, "MANIFEST", path)
    with pytest.raises(ValueError, match="no validated tasks yet"):
        list(vf.load_taskset(_config(split="tmax")))


def test_the_shipped_manifest_has_no_kept_task_before_the_box_phase():
    manifest = tmax_select.load_manifest()
    assert manifest["source"]["revision"] == tmax.SOURCE_REVISION
    assert manifest["counts"]["status"].get("kept", 0) == 0
    assert manifest["base_image"] is None
    statuses = {e["status"] for e in manifest["tasks"].values()}
    assert statuses <= {"pending", "excluded"}
    for entry in manifest["tasks"].values():
        assert entry["status"] != "excluded" or entry["reasons"]
        assert all(r in tmax_select.STATIC_REASONS for r in entry.get("reasons", []))
    assert len(manifest["tasks"]) == manifest["counts"]["tasks"] == 14601


def test_tmax_rejects_a_zip_that_is_not_the_pinned_one(tmp_path, monkeypatch, tmax_env):
    bogus = tmp_path / "tasks.zip"
    with io.BytesIO() as buffer:
        import zipfile

        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("x/task.json", "{}")
        bogus.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="not the pinned"):
        list(vf.load_taskset(_config(split="tmax", tmax_source=bogus)))


class SetupBox:
    def __init__(self, exit_code=0):
        self.files: dict[str, bytes] = {}
        self.runs: list[list[str]] = []
        self.exit_code = exit_code

    async def write(self, path, data):
        self.files[path] = data

    async def run(self, argv, env):
        self.runs.append(list(argv))
        return ProgramResult(self.exit_code, "", "boom" if self.exit_code else "")


async def test_setup_uploads_the_bundle_and_runs_it_in_the_agent_role(tmax_env, monkeypatch):
    src, _ = tmax_env
    task = _load(src, num_tasks=1)[0]
    box = SetupBox()
    await task.setup(box)
    archive = box.files[f"{tmax.SETUP_DIR}.tgz"]
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        names = set(tar.getnames())
    assert {"setup.sh", "post.sh", "inputs.json", "tmax_box.py", "files/fixtures/image.png"} <= names
    script = box.runs[-1][-1]
    assert f"bash {tmax.SETUP_DIR}/setup.sh agent" in script
    assert script.rstrip().endswith("exit $rc") and f"rm -rf {tmax.SETUP_DIR}" in script
    # Neither tests nor the reference solution reach the agent's box.
    assert not any(n.startswith(("tests", "solution")) for n in names)


async def test_a_failed_setup_raises(tmax_env, monkeypatch):
    src, _ = tmax_env
    task = _load(src, num_tasks=1)[0]
    with pytest.raises(RuntimeError, match=r"tmax setup \(agent\) failed"):
        await task.setup(SetupBox(exit_code=1))


async def test_the_grading_box_runs_setup_in_the_grade_role(tmax_env, monkeypatch):
    src, _ = tmax_env
    task = _load(src, num_tasks=1)[0]
    env = TerminalEnv(
        vf.env_config_type("reliquary-terminal")(taskset=_config(split="eval"), verifier_runtime=vf.DockerConfig())
    )
    graders = []

    async def fake_grade(config, grader, solution):
        graders.append(grader)
        box = SetupBox()
        await grader.setup(box)
        assert f"bash {tmax.SETUP_DIR}/setup.sh grade" in box.runs[-1][-1]
        return 1.0

    monkeypatch.setattr(env, "_grade", fake_grade)
    solver = vf.Trace(agent=vf.AgentInfo(config=vf.AgentConfig()), task=vf.TraceTask(type="TerminalTask", data=task.data))
    solver.ok = True
    await env.finalize(task, vf.Episode(task=solver.task, traces=[solver]))
    assert graders and graders[0].setup_role == "grade"
    assert task.setup_role == "agent"
    assert solver.rewards["solved"].value == 1.0


def test_unknown_setup_role_is_refused():
    import asyncio

    with pytest.raises(ValueError):
        asyncio.run(tmax.run_setup(SetupBox(), None, "root"))
