"""Shared helpers for the container-backed terminal tests."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.tasksets.harbor.taskset import HarborTask, make_tar


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
    if _docker_available():
        return
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

