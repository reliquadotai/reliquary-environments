"""Grade each rollout in a container the agent never entered.

The same shape as `verifiers.v1.tasksets.harbor.env.HarborEnv`: the solve runs
in the agent's box, then `finalize` provisions a fresh box from the task's own
image, restores the collected patch, grades there, and records the reward onto
the solver's trace.

Network is severed the same way `tests/conftest.py` already does it for every
grading box: `prepare_setup()` then `prepare_execution([])`, never skipped.
`DockerRuntime.start()` alone leaves a restricted container in an open
"trusted setup" state (egress `allow=["*"]`); only `prepare_execution` installs
the redirect that actually cuts it. A grading box nobody watches must not keep
live network past that point.

Infrastructure failures retry and then raise. They never score. A grading box
that could not be reached must not read as reward 0 -- that is an error about
us, not about the agent's patch.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.runtimes import RuntimeConfig, provision_runtime
from verifiers.v1.utils.compile import resolve_runtime_config
from verifiers.v1.utils.retries import backoff

from reliquary_swe import grading
from reliquary_swe.taskset import SweTask

logger = logging.getLogger(__name__)


class SweEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig()
    """The policy under training; pin `--env.agent.harness.*` to choose which
    harness drives it. Varying it across runs is the point."""
    grading_runtime: RuntimeConfig | None = None
    """Where grading runs. None derives it from the agent's runtime policy."""
    grading_retries: int = Field(2, ge=0)
    """Extra attempts at provisioning-and-grading before the episode fails.

    The intent is infrastructure retry, but the `except Exception` in
    `_grade` cannot actually tell that apart from `grading.grade`'s own
    deterministic raise (a freshly provisioned box that will not even check
    out `base_commit` -- see that function's docstring): a genuinely broken
    instance's image pays for `grading_retries` extra, guaranteed-identical
    provision-and-grade attempts before it fails. Narrowing the `except` to
    exclude that one raise would fix this at the cost of new retry-boundary
    behavior on a path this package leans on for real infra flakiness; left
    as a known, accepted cost instead of claimed away."""


class SweEnv(vf.Env[SweEnvConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if not isinstance(task, SweTask):
            raise TypeError(f"the swe env runs swe tasks; got {type(task).__name__}")
        await agents.agent.run(task.graded_elsewhere())

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        """Grade the solver's captured patch in a box it never touched, and
        record the result onto its trace. A trace that never completed
        (`not solution.ok`) is left unscored, same as an agent that never
        produced a patch at all -- there is nothing here to grade. The task records no
        reward of its own here (`graded_elsewhere`)."""
        if not isinstance(task, SweTask):
            return
        solution = episode.traces[0]
        if not solution.ok:
            return
        patch = solution.info.get("patch", "")
        report = await self._grade(task, patch)
        solution.record_reward("patch_passes_tests", report.reward)
        solution.info["swe_report"] = {
            "applied": report.applied,
            # Whether every path the grader's OWN restoration strategy
            # touched came back confirmed at base_commit -- not whether that
            # strategy attempted every path that mattered (see
            # grading.Report.restored). False here means this 0 may be a
            # grader-side failure, not the agent's.
            "restored": report.restored,
            "fail_to_pass_passed": report.fail_to_pass_passed,
            "fail_to_pass_total": len(task.data.fail_to_pass),
            "pass_to_pass_passed": report.pass_to_pass_passed,
            "pass_to_pass_total": len(task.data.pass_to_pass),
            # Monitoring, not reward -- IMPORTANT 3. `results_parsed == 0`
            # is per-instance and ambiguous on its own (a syntactically
            # broken patch legitimately produces none); aggregated across a
            # batch, `pass_to_pass_passed == 0` for every instance at once is
            # the canary an adapter break actually looks like -- see
            # grading.grade's own docstring.
            "results_parsed": report.results_parsed,
            "test_command_exit_code": report.test_command_exit_code,
            # Polyglot and R2E only (empty otherwise): the one clue to
            # whether a 0 was a real failing test or a runner that never
            # started -- see grading.Report.test_output_tail.
            "test_output_tail": report.test_output_tail,
        }

    async def _grade(self, task: SweTask, patch: str) -> grading.Report:
        base = (
            self.config.grading_runtime
            if self.config.grading_runtime is not None
            else self.config.agent.runtime
        )
        config = resolve_runtime_config(base, task)
        last: Exception | None = None
        for attempt in range(self.config.grading_retries + 1):
            if attempt:
                delay = backoff(attempt - 1)
                logger.warning(
                    "swe grading attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    self.config.grading_retries + 1,
                    last,
                    delay,
                )
                await asyncio.sleep(delay)
            try:
                async with AsyncExitStack() as boxes:
                    async with asyncio.timeout(task.data.timeout.scoring):
                        box = await boxes.enter_async_context(
                            provision_runtime(config, env=task.runtime_env())
                        )
                        await box.prepare_setup()
                        await box.prepare_execution([])
                        return await grading.grade(box, task.data, patch)
            except Exception as e:  # noqa: BLE001 - each attempt's failure is retried
                last = e
        assert last is not None
        raise last
