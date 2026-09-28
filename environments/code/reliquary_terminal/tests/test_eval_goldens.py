"""The `eval` split (Terminal-Bench 2.1) graded as it ships, in the agent's box."""

from __future__ import annotations

import pytest
from conftest import provisioned, run_reference_solution, task, trace

docker = pytest.mark.docker

# One of the three tasks whose image works outside `/app`: its reference
# solution scored 0 ("not a git repository") until the workdir was read from
# the image, so this golden is what pins `image_workdir`.
WORKDIR_GOLDEN = "terminal-bench/fix-git"


@docker
async def test_the_reference_solution_scores_one_in_the_images_own_workdir():
    t = task("eval", WORKDIR_GOLDEN)
    assert t.data.workdir == "/app/personal-site"
    async with provisioned(t) as box:
        await run_reference_solution(t, box)
        score = await t.solved(box, trace(t))
    assert score == 1.0


@docker
async def test_an_untouched_box_scores_zero():
    t = task("eval", WORKDIR_GOLDEN)
    async with provisioned(t) as box:
        score = await t.solved(box, trace(t))
    assert score == 0.0
