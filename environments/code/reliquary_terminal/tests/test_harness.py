"""The bash harness's per-command timeout, without a container: the patched
program is executed on this host."""

from __future__ import annotations

import os
import subprocess
import time

import pytest
import verifiers.v1 as vf
from verifiers.v1.harnesses.bash.harness import PROGRAM_SOURCE as UPSTREAM_PROGRAM
from verifiers.v1.utils.loaders import default_harness_id, load_harness

from reliquary_terminal import harness


def _program(timeout: float) -> dict:
    namespace: dict = {"__name__": "reliquary_terminal_program_under_test"}
    exec(compile(harness.program_source(timeout), "program.py", "exec"), namespace)
    return namespace


def test_a_command_past_the_timeout_is_killed_and_the_agent_told():
    run_bash = _program(1)["run_bash"]
    started = time.monotonic()
    out = run_bash("echo before; sleep 30; echo after")
    assert time.monotonic() - started < 10
    assert out.startswith("before\n")
    assert "\nafter\n" not in out
    assert "command timed out after 1 s" in out


def test_the_whole_pipeline_dies_not_just_bash(tmp_path):
    # Upstream's subprocess.run(timeout=...) kills `bash` alone; the rest of a
    # pipeline ran on. Every process the command started must be gone.
    marker = tmp_path / "pids"
    run_bash = _program(1)["run_bash"]
    run_bash(f"sleep 7601 | sleep 7602 & echo $! >> {marker}; sh -c 'echo $$ >> {marker}; sleep 7603'")
    pids = [int(p) for p in marker.read_text().split()]
    assert pids
    time.sleep(0.5)
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    leftover = subprocess.run(["pgrep", "-f", "^sleep 760[123]$"], capture_output=True, text=True)
    assert leftover.stdout.strip() == ""


def test_a_command_inside_the_timeout_is_untouched():
    run_bash = _program(5)["run_bash"]
    assert run_bash("echo out; echo err >&2") == "out\nerr\n"


def test_a_backgrounded_job_with_its_output_redirected_returns_at_once():
    run_bash = _program(5)["run_bash"]
    started = time.monotonic()
    assert run_bash("(sleep 3 > /dev/null 2>&1 &); echo ok") == "ok\n"
    assert time.monotonic() - started < 2


def test_a_descendant_that_left_the_group_cannot_hang_the_call():
    # setsid puts it beyond the group kill, still holding the output pipe.
    run_bash = _program(1)["run_bash"]
    started = time.monotonic()
    out = run_bash("setsid sleep 7620; true")
    assert time.monotonic() - started < 10
    assert "timed out" in out
    subprocess.run(["pkill", "-f", "^sleep 7620$"], check=False)


def test_the_program_is_upstreams_with_only_run_bash_replaced():
    ours = harness.program_source(180)
    assert ours.startswith(UPSTREAM_PROGRAM.rsplit('\n\nif __name__ == "__main__":', 1)[0])
    assert "COMMAND_TIMEOUT = 180.0" in ours
    assert ours.endswith('if __name__ == "__main__":\n    asyncio.run(main())\n')
    # The replacement is defined after main(), which resolves it at call time.
    assert ours.index("COMMAND_TIMEOUT = 180.0") > ours.index("async def main()")


def test_this_package_is_its_own_default_harness_with_a_180_s_timeout():
    assert default_harness_id("reliquary-terminal") == "reliquary-terminal"
    config = vf.harness_config_type("reliquary-terminal")(id="reliquary-terminal")
    assert config.command_timeout == harness.DEFAULT_COMMAND_TIMEOUT_SECONDS == 180.0
    assert isinstance(load_harness(config), harness.TerminalHarness)


class _RecordingRuntime:
    def __init__(self) -> None:
        self.scripts: list[str] = []
        self.argv: list[str] | None = None

    async def prepare_uv_script(self, script, env=None, *, activate=True):
        self.scripts.append(script)
        return ["python", "program.py"]

    async def run_program(self, argv, env):
        self.argv = argv
        return vf.runtimes.ProgramResult(exit_code=0, stdout="", stderr="")


async def test_setup_and_launch_both_run_the_patched_program():
    config = vf.harness_config_type("reliquary-terminal")(id="reliquary-terminal", command_timeout=42)
    h = load_harness(config)
    runtime = _RecordingRuntime()
    await h.setup(runtime)
    data = vf.TaskData(idx=0, prompt="hello")
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="TerminalTask", data=data),
    )
    ctx = vf.clients.ModelContext(client=vf.EvalClientConfig(), model="m", sampling=vf.SamplingConfig())
    await h.launch(ctx, trace, runtime, "http://x/v1", "secret", {}, data)
    assert len(runtime.scripts) == 2
    assert all("COMMAND_TIMEOUT = 42.0" in s for s in runtime.scripts)
    assert runtime.argv is not None and "--prompt=hello" in runtime.argv
