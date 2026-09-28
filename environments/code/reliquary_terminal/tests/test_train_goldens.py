"""The `train` split graded in a fresh box that receives only the agent's
`/app`, through `HarborEnv.finalize` -- the path a training rollout takes."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import verifiers.v1 as vf
from conftest import provisioned, task, trace
from verifiers.v1.tasksets.harbor.env import HarborEnv, HarborEnvConfig

docker = pytest.mark.docker

GOLDEN = "candidate-0036-software-data-engineering"
REFERENCE = Path(__file__).parent / "reference_solutions" / f"{GOLDEN}.sh"


async def _episode(extra: str = "", solve: bool = True) -> vf.Episode:
    """Run the agent phase without a model -- the reference fix (or nothing),
    then `extra` -- capture its artifacts, and grade them elsewhere."""
    t = task("train", GOLDEN)
    solver = trace(t)
    async with provisioned(t) as box:
        if solve:
            await box.write("/tmp/solve.sh", REFERENCE.read_bytes())
            solved = await box.run(["sh", "/tmp/solve.sh"], {})
            assert solved.exit_code == 0, solved.stderr or solved.stdout
        if extra:
            ran = await box.run(["sh", "-c", extra], {})
            assert ran.exit_code == 0, ran.stderr
        await t.graded_elsewhere().finalize(solver, box)
    solver.ok = True
    episode = vf.Episode(
        task=vf.TraceTask(type=type(t).__name__, data=t.data, key=t.key, hash=t.hash),
        traces=[solver],
    )
    env = HarborEnv(
        HarborEnvConfig(
            taskset=vf.taskset_config_type("reliquary-terminal")(
                id="reliquary-terminal", split="train"
            ),
            verifier_runtime=vf.DockerConfig(),
        )
    )
    await env.finalize(t, episode)
    return episode


@docker
async def test_the_reference_fix_scores_one_in_a_fresh_box():
    episode = await _episode()
    assert episode.traces[0].reward == 1.0


@docker
async def test_an_untouched_app_scores_zero():
    episode = await _episode(solve=False)
    assert episode.traces[0].reward == 0.0


@docker
async def test_a_planted_pytest_hook_zeroes_even_a_correct_fix():
    # The same correct fix, plus a conftest.py the pristine tree never had.
    # Checked to have travelled with the artifacts, so the zero is the tests'
    # own guard rejecting it, not the plant failing to land.
    episode = await _episode(extra="echo '# planted' > /app/conftest.py")
    archive = episode.traces[0].state.artifacts["/app"]
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        names = tar.getnames()
    assert "app/conftest.py" in names
    assert episode.traces[0].reward == 0.0
