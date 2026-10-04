"""verifiers' `bash` harness with a per-command timeout.

The upstream program (`verifiers.v1.harnesses.bash.program`, at the pinned
revision) runs each `bash` tool call through `subprocess.run(..., timeout=3600)`:
one hour per command, with no way to change it, and on expiry only `bash`
itself is killed -- a pipeline's other processes run on. In the 2026-10-02
qualification run (Qwen3.8-27B, `bash` harness) a single repository-wide
`grep` held its rollout for 1,489 s.

This harness is that program with one function replaced: `run_bash` runs the
command in its own session, and past `command_timeout` seconds kills the whole
process group and tells the agent so. Everything else -- tools, prompts,
interception, MCP -- is the upstream harness, unmodified.
"""

from __future__ import annotations

from pydantic import Field

from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.bash import BashHarness, BashHarnessConfig
from verifiers.v1.harnesses.bash.harness import PROGRAM_SOURCE as UPSTREAM_PROGRAM

# Measured on the 2026-10-02 qualification traces (gap between a tool-calling
# model turn and the next one, i.e. the command's own wall time, on
# sandbox-dev-01): 2,120 terminal commands, Qwen3-4B and Qwen3.8-27B, max 10 s,
# p99 2.9 s. 180 s is 18x the longest observed and still caps a runaway at
# 3 minutes, against the 1,489 s the uncapped one took.
DEFAULT_COMMAND_TIMEOUT_SECONDS = 180.0

_MAIN = '\n\nif __name__ == "__main__":\n    asyncio.run(main())\n'

# Appended after the upstream definitions, so `main()` -- which looks `run_bash`
# up at call time -- runs this one. `{timeout!r}` is the only substitution.
_RUN_BASH = '''

import os as _os
import signal as _signal

COMMAND_TIMEOUT = {timeout!r}


def run_bash(command: str) -> str:
    """`bash -c command` in its own session, killed with every process it
    started once COMMAND_TIMEOUT seconds have passed."""
    try:
        proc = subprocess.Popen(
            ["bash", "-c", command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"error: {{e}}"
    try:
        stdout, stderr = proc.communicate(timeout=COMMAND_TIMEOUT)
        return stdout + stderr
    except subprocess.TimeoutExpired:
        pass
    with suppress(ProcessLookupError):
        _os.killpg(proc.pid, _signal.SIGKILL)
    try:
        # Output read so far is kept across the timeout (subprocess docs).
        stdout, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        # A descendant that left the process group still holds the pipes:
        # give up on them rather than wait on it.
        stdout, stderr = "", ""
        for pipe in (proc.stdout, proc.stderr):
            with suppress(Exception):
                pipe.close()
        with suppress(Exception):
            proc.wait(timeout=1)
    return (
        stdout
        + stderr
        + f"\\n[command timed out after {{COMMAND_TIMEOUT:g}} s and was killed, "
        "with every process it started. Run long jobs in the background with "
        "their output redirected to a file, or narrow the command.]"
    )
'''


def program_source(command_timeout: float) -> str:
    """The upstream bash program with `run_bash` replaced.

    Raises if the pinned program no longer ends the way this patch expects,
    rather than shipping a program whose timeout silently does not apply.
    """
    if not UPSTREAM_PROGRAM.endswith(_MAIN) or "\ndef run_bash(command: str) -> str:\n" not in UPSTREAM_PROGRAM:
        raise RuntimeError(
            "verifiers' bash program changed shape; re-check reliquary_terminal.harness "
            "against it before raising the pin"
        )
    body = UPSTREAM_PROGRAM[: -len(_MAIN)]
    return body + _RUN_BASH.format(timeout=float(command_timeout)) + _MAIN


class TerminalHarnessConfig(BashHarnessConfig):
    command_timeout: float = Field(DEFAULT_COMMAND_TIMEOUT_SECONDS, gt=0)
    """Seconds one `bash` tool call may run before it is killed, with every
    process it started; the agent gets its output so far and a note saying so.
    Set with `--env.agent.harness.command-timeout`."""


class _ProgramSwap:
    """The runtime, except that preparing the upstream bash program prepares
    ours instead. `BashHarness.launch` names its program by module constant,
    so this is how it runs a different one without being copied here."""

    def __init__(self, runtime, source: str) -> None:
        self._runtime = runtime
        self._source = source

    async def prepare_uv_script(self, script, env=None, *, activate=True):
        if script == UPSTREAM_PROGRAM:
            script = self._source
        return await self._runtime.prepare_uv_script(script, env, activate=activate)

    def __getattr__(self, name):
        return getattr(self._runtime, name)


class TerminalHarness(BashHarness, Harness[TerminalHarnessConfig]):
    """`bash` with a per-command timeout (see the module docstring)."""

    @property
    def program(self) -> str:
        return program_source(self.config.command_timeout)

    async def setup(self, runtime) -> None:
        await runtime.prepare_uv_script(self.program, self.config.resolved_env)

    async def launch(self, ctx, trace, runtime, *args, **kwargs):
        return await super().launch(ctx, trace, _ProgramSwap(runtime, self.program), *args, **kwargs)


__all__ = [
    "DEFAULT_COMMAND_TIMEOUT_SECONDS",
    "TerminalHarness",
    "TerminalHarnessConfig",
    "program_source",
]
