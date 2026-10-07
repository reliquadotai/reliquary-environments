"""Shared helpers for the container-backed terminal tests."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.tasksets.harbor.taskset import HarborTask, make_tar


def _install_episode_task_shim() -> None:
    """reliquary-sandbox is private and absent in CI: when its `episode_task` cannot be
    imported, put the test-only copy (`episode_task_shim.py`, stdlib only) in its place so
    the hook tests run anyway. `test_episode_task_shim.py` checks the copy against the
    real module wherever that is installed."""
    try:
        import reliquary_sandbox.episode_task  # noqa: F401
    except ImportError:
        name = "reliquary_sandbox.episode_task"
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("episode_task_shim.py"))
        module = importlib.util.module_from_spec(spec)
        parent = sys.modules.get("reliquary_sandbox")
        if parent is None:
            parent = types.ModuleType("reliquary_sandbox")
            parent.__path__ = []
            sys.modules["reliquary_sandbox"] = parent
        sys.modules[name] = module
        spec.loader.exec_module(module)
        parent.episode_task = module


_install_episode_task_shim()


BOX_CPU = 2.0
BOX_MEMORY = 4.0  # GB


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "version"], capture_output=True, timeout=10, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    # Opt-in: a machine that merely has Docker (a laptop, a shared host) must
    # not start pulling images and running task containers by accident.
    if os.environ.get("RELIQUARY_DOCKER_TESTS") != "1":
        skip = pytest.mark.skip(reason="container tests run only with RELIQUARY_DOCKER_TESTS=1")
    elif _docker_available():
        return
    else:
        skip = pytest.mark.skip(reason="no reachable Docker daemon")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)


def task(split: str, name: str) -> HarborTask:
    config = vf.taskset_config_type("reliquary-terminal")
    for candidate in vf.load_taskset(config(id="reliquary-terminal", split=split)):
        if candidate.data.name == name:
            return candidate
    raise AssertionError(f"{name} is not in the {split} split")


@asynccontextmanager
async def provisioned(task: HarborTask) -> AsyncIterator[vf.Runtime]:
    """The box the rollout pipeline builds for `task`'s agent: same image,
    workdir and network policy, already past trusted setup."""
    resources = task.data.resources
    config = vf.DockerConfig(
        image=task.data.image,
        workdir=task.data.workdir,
        allow=task.data.network_allow,
        # The task's own request, never more than BOX_CPU / BOX_MEMORY: the
        # container hosts these tests share are capped per container.
        cpu=min(resources.cpu or BOX_CPU, BOX_CPU),
        memory=min(resources.memory or BOX_MEMORY, BOX_MEMORY),
    )
    async with provision_runtime(config, env=task.runtime_env()) as box:
        await box.prepare_setup()
        await task.setup(box)
        await box.prepare_execution([])
        yield box


async def run_reference_solution(task: HarborTask, box: vf.Runtime) -> None:
    solution = Path(task.data.task_dir) / "solution"
    await box.write("/tmp/solution.tgz", make_tar(solution))
    staged = await box.run(
        ["sh", "-c", "mkdir -p /solution && tar -xzf /tmp/solution.tgz -C /solution"], {}
    )
    assert staged.exit_code == 0, staged.stderr
    # From the task's workdir, as Harbor runs it.
    await box.run(["bash", "/solution/solve.sh"], {})


def trace(task: HarborTask) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
    )


@pytest.fixture(autouse=True)
def _ledger_dir(tmp_path, monkeypatch):
    """Container ledgers of a test run go to a scratch directory, never the
    user's own (`reliquary_terminal.containers`)."""
    from reliquary_terminal import containers

    root = tmp_path / "ledgers"
    monkeypatch.setattr(containers, "LEDGER_DIR", root)
    monkeypatch.setattr(containers, "_mine", {})
    monkeypatch.setenv("RELIQUARY_TERMINAL_LEDGER_DIR", str(root))
    return root
