"""verifiers' `bash` harness with a per-command timeout.

The upstream program (`verifiers.v1.harnesses.bash.program`, at the pinned
revision) runs each `bash` tool call through `subprocess.run(..., timeout=3600)`:
one hour per command, with no way to change it, and on expiry only `bash`
itself is killed -- a pipeline's other processes run on. In the 2026-10-02
qualification run (Qwen3.8-27B, `bash` harness) a single repository-wide
`grep` held its rollout for 1,489 s.

This harness is that program with one function replaced: `run_bash` runs the
command in its own session, past `command_timeout` seconds kills its process
group and tells the agent so, and keeps only the last MiB of each output
stream. Everything else -- tools, prompts,
interception, MCP -- is the upstream harness, unmodified.
"""

from __future__ import annotations

import hashlib

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

# The pinned program (verifiers b2e4e815) this patch was written and tested
# against; any other text is refused rather than patched blind.
UPSTREAM_PROGRAM_SHA256 = "5ddf3f27512ed9519b29c286756bb74fa3dcbd1d4a558fae36b4111e11d94fdb"
# Output kept per command (the last bytes of each stream).
MAX_OUTPUT_BYTES = 1024 * 1024

_MAIN = '\n\nif __name__ == "__main__":\n    asyncio.run(main())\n'

# Appended after the upstream definitions, so `main()` -- which looks `run_bash`
# up at call time -- runs this one. `{timeout!r}` is the only substitution.
_RUN_BASH = '''

import os as _os
import signal as _signal
import threading as _threading
import time as _time

COMMAND_TIMEOUT = {timeout!r}
MAX_OUTPUT_BYTES = {max_output!r}


class _Tail:
    """The last MAX_OUTPUT_BYTES of a stream, read on a thread, so a command
    printing without end (`yes`) cannot grow this process without bound."""

    def __init__(self, stream):
        self.data = bytearray()
        self.dropped = 0
        self._stream = stream
        self.thread = _threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        with suppress(Exception):
            while chunk := self._stream.read1(65536):
                self.data += chunk
                excess = len(self.data) - MAX_OUTPUT_BYTES
                if excess > 0:
                    del self.data[:excess]
                    self.dropped += excess

    def text(self):
        out = bytes(self.data).decode(errors="replace")
        if self.dropped:
            out = f"[... {{self.dropped}} earlier bytes dropped ...]\\n" + out
        return out


def run_bash(command: str) -> str:
    """`bash -c command` in its own session; past COMMAND_TIMEOUT seconds its
    process group is killed and the output so far returned with a note."""
    try:
        proc = subprocess.Popen(
            ["bash", "-c", command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"error: {{e}}"
    stdout, stderr = _Tail(proc.stdout), _Tail(proc.stderr)
    # The deadline covers the output too: a background job that inherited
    # the pipes keeps the command "running" for the agent until it exits.
    deadline = _time.monotonic() + COMMAND_TIMEOUT
    timed_out = False
    try:
        proc.wait(timeout=COMMAND_TIMEOUT)
        for tail in (stdout, stderr):
            tail.thread.join(timeout=max(0.0, deadline - _time.monotonic()))
        timed_out = stdout.thread.is_alive() or stderr.thread.is_alive()
    except subprocess.TimeoutExpired:
        timed_out = True
    if timed_out:
        with suppress(ProcessLookupError):
            _os.killpg(proc.pid, _signal.SIGKILL)
        with suppress(Exception):
            proc.wait(timeout=5)
        # A descendant that left the process group may still hold the pipes:
        # wait a little for the rest, then return what there is.
        grace = _time.monotonic() + 5
        for tail in (stdout, stderr):
            tail.thread.join(timeout=max(0.0, grace - _time.monotonic()))
    out = stdout.text() + stderr.text()
    if timed_out:
        out += (
            f"\\n[command timed out after {{COMMAND_TIMEOUT:g}} s; its process group "
            "was killed. Run long jobs in the background with their output "
            "redirected to a file, or narrow the command.]"
        )
    return out
'''


def program_source(command_timeout: float) -> str:
    """The upstream bash program with `run_bash` replaced.

    Raises if the program is not the pinned one, rather than shipping a
    program whose timeout silently does not apply.
    """
    digest = hashlib.sha256(UPSTREAM_PROGRAM.encode()).hexdigest()
    if digest != UPSTREAM_PROGRAM_SHA256 or not UPSTREAM_PROGRAM.endswith(_MAIN):
        raise RuntimeError(
            "verifiers' bash program changed shape; re-check reliquary_terminal.harness "
            "against it before raising the pin"
        )
    body = UPSTREAM_PROGRAM[: -len(_MAIN)]
    return body + _RUN_BASH.format(timeout=float(command_timeout), max_output=MAX_OUTPUT_BYTES) + _MAIN


class TerminalHarnessConfig(BashHarnessConfig):
    command_timeout: float = Field(DEFAULT_COMMAND_TIMEOUT_SECONDS, gt=0)
    """Seconds one `bash` tool call may run (output included) before its
    process group is killed; the agent gets its output so far (the last
    MAX_OUTPUT_BYTES of each stream) and a note saying so.
    Set with `--env.agent.harness.command-timeout`."""


class _ProgramSwap:
    """The runtime, except that preparing the upstream bash program prepares
    ours instead. `BashHarness.launch` names its program by module constant,
    so this is how it runs a different one without being copied here."""

    def __init__(self, runtime, source: str) -> None:
        self._runtime = runtime
        self._source = source

    async def prepare_uv_script(self, script, env=None, *, activate=True):
        if script != UPSTREAM_PROGRAM:
            raise RuntimeError(
                "reliquary-terminal: the bash harness prepared a script other than the "
                "pinned program; re-check reliquary_terminal.harness against verifiers"
            )
        return await self._runtime.prepare_uv_script(self._source, env, activate=activate)

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
