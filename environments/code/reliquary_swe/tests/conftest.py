"""Shared fixtures for the container-backed SWE taskset tests.

`runtime` provisions the same box the rollout pipeline would build for the
first task's agent phase: same image, same workdir, same network policy.

The network part is not automatic. `DockerRuntime.start()` always leaves a
restricted container in a "trusted setup" state -- egress wide open -- and
only `prepare_execution` installs the iptables redirect that actually severs
it (see `verifiers.v1.runtimes.docker.DockerRuntime.prepare_execution`; the
real pipeline calls it in `rollout.py`, right before the agent's turns start,
mirrored here since these tests call `Task.setup`/`finalize` directly instead
of going through that pipeline). Skipping this call would make
`test_the_container_cannot_reach_the_network` pass for the wrong reason: not
because the policy is enforced, but because nothing ever enforced it.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes import provision_runtime

from reliquary_swe.env import SweEnv, SweEnvConfig
from reliquary_swe.taskset import SweTask


def _docker_available() -> bool:
    """Whether a real, reachable Docker daemon exists here.

    `shutil.which` alone would pass on a machine that has the `docker` CLI
    installed but no daemon behind it (or one this process cannot reach), so
    `docker version` -- a real round trip to the daemon, never an image pull
    or a container start -- is what actually decides it.
    """
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(["docker", "version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Skip `@docker` tests when no Docker daemon is reachable.

    Spec section 9 and both READMEs promise `uv run pytest` stays runnable on
    a machine that cannot host images; without this hook, `@docker` tests
    instead ERROR on such a machine (no `runtime`/`grading_runtime` fixture
    can be built), which is not the same promise.
    """
    if _docker_available():
        return
    skip_docker = pytest.mark.skip(reason="no reachable Docker daemon")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip_docker)


# Small and already pulled on the container host (see remote-test): 15 tests,
# ~2.3s to grade (measured on the box). Most grading goldens score this one
# instance so the routine suite stays fast.
GOLDEN = "astropy__astropy-12907"

# django's own test runner (`./tests/runtests.py`), not pytest -- and, for
# this specific instance, test_patch touches only .txt fixtures, so its own
# get_test_directives is empty and a genuine run (post-restoration) means
# django's *entire* suite (~4 min; see test_adapter.py's own slow test on
# the same instance). Used for the one repo-family-specific negative control
# CRITICAL 2's fix needs: nothing about test_patch names this file, so
# nothing about test_patch's own restoration would ever protect it.
DJANGO_GOLDEN = "django__django-10097"

# sphinx via `tox --current-env`, reading in-repo `tox.ini` -- the other
# repo-family CRITICAL 2 needs a negative control from. Its own test_patch
# also happens to *add* two files rather than modify any existing one, which
# is exactly the shape CRITICAL 1's batched-checkout bug silently ate whole.
SPHINX_GOLDEN = "sphinx-doc__sphinx-8595"


def _first_task() -> SweTask:
    config = vf.taskset_config_type("reliquary-swe")
    return next(iter(vf.load_taskset(config(id="reliquary-swe")).head(1)))


def _task_for(instance_id: str) -> SweTask:
    config = vf.taskset_config_type("reliquary-swe")
    for task in vf.load_taskset(config(id="reliquary-swe")):
        if task.data.instance_id == instance_id:
            return task
    raise AssertionError(f"{instance_id} is not in the taskset")


@asynccontextmanager
async def provisioned_runtime(task: SweTask) -> AsyncIterator[vf.Runtime]:
    """Provision and tear down a real box for `task`, network-restricted and
    execution-ready. Factored out of the `runtime` fixture below so a test
    that needs a *different* task's image (e.g. a specific corpus instance,
    not the taskset's first) can still get the same real setup rather than
    a hand-rolled, possibly-diverging one.
    """
    docker_config = vf.DockerConfig(
        image=task.data.image,
        workdir=task.data.workdir,
        allow=task.data.network_allow,
    )
    async with provision_runtime(docker_config) as box:
        # Marks the box as having been through one trusted-setup pass; a
        # no-op here (this box is fresh, never reused across attempts) but
        # the correct call per `prepare_execution`'s own contract.
        await box.prepare_setup()
        # No framework routes: this taskset's tools run as shell commands
        # inside the box, not as HTTP calls a harness needs to reach out for.
        await box.prepare_execution([])
        yield box


@pytest.fixture
async def runtime() -> AsyncIterator[vf.Runtime]:
    async with provisioned_runtime(_first_task()) as box:
        yield box


@pytest.fixture
async def grading_runtime() -> AsyncIterator[vf.Runtime]:
    """A pristine box provisioned from the golden instance's own image, torn
    down after every test -- so one golden's tampering (e.g. a patch that
    adds a conftest.py) can never leak into the next golden's box.
    """
    async with provisioned_runtime(_task_for(GOLDEN)) as box:
        yield box


@pytest.fixture
async def django_runtime() -> AsyncIterator[vf.Runtime]:
    async with provisioned_runtime(_task_for(DJANGO_GOLDEN)) as box:
        yield box


@pytest.fixture
async def sphinx_runtime() -> AsyncIterator[vf.Runtime]:
    async with provisioned_runtime(_task_for(SPHINX_GOLDEN)) as box:
        yield box


def _trace(task: SweTask) -> vf.Trace:
    # Matches tests/test_episode.py's own `_trace`: `setup`/`finalize` only
    # read `trace.info`, so the rest of this shape just needs to validate.
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        ),
    )


async def run_gold_episode(task: SweTask) -> vf.Episode:
    """The whole loop, without a model: apply `task.data.gold_patch` inside a
    fresh agent box, capture it exactly as `SweTask.finalize` does, then let
    `SweEnv.finalize` grade the capture in a second, separately provisioned
    box. Exercises capture and grading together -- neither one mocked.
    """
    trace = _trace(task)
    async with provisioned_runtime(task) as box:
        await task.setup(trace, box)
        await box.write("/tmp/gold.diff", task.data.gold_patch.encode())
        applied = await box.run(["git", "apply", "-v", "/tmp/gold.diff"], {})
        assert applied.exit_code == 0, f"gold patch did not apply: {applied.stderr}"
        await task.finalize(trace, box)
    # The real pipeline sets this via the harness's own turn loop; faked here
    # since finalize()'s grading gate is `solution.ok`, not "a patch exists".
    trace.ok = True

    episode = vf.Episode(
        task=vf.TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        ),
        traces=[trace],
    )
    config_cls = vf.taskset_config_type("reliquary-swe")
    env = SweEnv(
        SweEnvConfig(
            taskset=config_cls(id="reliquary-swe"),
            # This box is Docker-only; the config's own default (Prime) has no
            # host here. The task's real image/workdir/network policy still
            # come from resolve_runtime_config, exactly as a real run would.
            grading_runtime=vf.DockerConfig(),
        )
    )
    await env.finalize(task, episode)
    return episode


def unreachable_runtime_config() -> vf.DockerConfig:
    """A runtime nothing can provision. Not a real image reference, so `docker
    run` refuses it on syntax alone -- no registry round trip, no DNS wait,
    just a fast, deterministic provisioning failure standing in for "the
    grading box could not be reached".
    """
    return vf.DockerConfig(image="not a docker image reference")


def task_with_patch(instance_id: str = GOLDEN) -> SweTask:
    """A `SweTask` with no declared image, so `resolve_runtime_config` leaves
    `unreachable_runtime_config`'s own (broken) image in place instead of
    overriding it with this instance's real one (it only injects `task.data.image`
    when that field is set -- see `verifiers.v1.utils.compile.resolve_runtime_config`).
    Everything else is a real corpus row; `_grade` never reads it, since
    provisioning fails first, but that keeps this fixture honest if it ever does.
    """
    data = _task_for(instance_id).data.model_copy(update={"image": None})
    return SweTask(data)


def episode_with_patch(instance_id: str = GOLDEN) -> vf.Episode:
    """An episode whose one trace is already `ok` and already carries a
    captured patch in `info["patch"]` -- the shape `SweEnv.finalize` expects
    on entry. Pairs with `task_with_patch`; the patch text itself is never
    read when provisioning fails before grading does.
    """
    task = task_with_patch(instance_id)
    trace = _trace(task)
    trace.ok = True
    trace.info["patch"] = "diff --git a/nonexistent.py b/nonexistent.py\n"
    return vf.Episode(
        task=vf.TraceTask(
            type=type(task).__name__, data=task.data, key=task.key, hash=task.hash
        ),
        traces=[trace],
    )
