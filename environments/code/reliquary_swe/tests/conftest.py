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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes import provision_runtime

from reliquary_swe.taskset import SweTask

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
