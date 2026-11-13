"""The reward against a scripted box (the shell scripts run on gVisor in the gold check)."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from reliquary_terminal_bench import task as task_module
from reliquary_terminal_bench.task import PATHS, STATE, TerminalBenchTask, task_data


class ScriptedBox:
    config = SimpleNamespace(type="reliquary-sandbox")

    def __init__(self, files=None, exits=None):
        self.files = dict(files or {})
        self.exits = dict(exits or {})
        self.ran = []

    async def run(self, argv, env):
        script = argv[2] if len(argv) > 2 else " ".join(argv)
        self.ran.append(script)
        if argv[:2] == ["sh", "-c"] and script.startswith('test -f'):
            return SimpleNamespace(exit_code=0 if argv[4] in self.files else 1, stdout="", stderr="")
        for marker, code in self.exits.items():
            if marker in script:
                return SimpleNamespace(exit_code=code, stdout="", stderr="boom")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def read(self, path, max_bytes=None):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write(self, path, data):
        self.files[path] = data


def trace():
    t = SimpleNamespace(info={}, metrics={})
    t.record_metrics = t.metrics.update
    return t


def solved(box):
    task = TerminalBenchTask(task_data("regex-log", 0, "eval_all"))
    t = trace()
    return asyncio.run(task.solved(box, t)), t, box


def test_no_state_is_zero_and_runs_nothing():
    reward, t, box = solved(ScriptedBox())
    assert reward == 0.0 and t.metrics == {"state_missing": 1.0}
    assert not any("test.sh" in s for s in box.ran)


@pytest.mark.parametrize("text,expected", [("1", 1.0), ("0", 0.0), ("", 0.0), ("2", 0.0),
                                           ("nan?", 0.0)])
def test_reward_txt(text, expected):
    files = {STATE: b"x", PATHS: b"y", "/logs/verifier/reward.txt": text.encode()}
    reward, t, box = solved(ScriptedBox(files))
    assert reward == expected
    assert any("bash /tests/test.sh" in s for s in box.ran)
    assert t.metrics["test_sh_exit_code"] == 0.0


def test_reward_json_wins_and_may_be_keyed():
    files = {STATE: b"x", PATHS: b"y", "/logs/verifier/reward.txt": b"0",
             "/logs/verifier/reward.json": json.dumps({"reward": 1}).encode()}
    assert solved(ScriptedBox(files))[0] == 1.0


def test_unappliable_state_is_zero():
    files = {STATE: b"x", PATHS: b"y", "/logs/verifier/reward.txt": b"1"}
    reward, t, _ = solved(ScriptedBox(files, exits={"comm -z": 13}))
    assert reward == 0.0 and t.metrics == {"state_unappliable": 1.0}


def test_the_tests_run_offline_from_the_workdir():
    files = {STATE: b"x", PATHS: b"y", "/logs/verifier/reward.txt": b"1"}
    _, _, box = solved(ScriptedBox(files))
    assert "cd /app && bash /tests/test.sh" in box.ran[-3] or any(
        "cd /app && bash /tests/test.sh" in s for s in box.ran)
    assert task_module.GRADING_ENV["UV_OFFLINE"] == "1"


def test_setup_failure_is_ours():
    task = TerminalBenchTask(task_data("regex-log", 0, "eval_all"))
    with pytest.raises(RuntimeError, match="setup failed"):
        asyncio.run(task.setup(ScriptedBox(exits={"rm -rf -- /opt/tb-verifier": 3})))


def test_a_capture_error_is_recorded_not_raised():
    task = TerminalBenchTask(task_data("regex-log", 0, "eval_all"))
    t = trace()
    asyncio.run(task.finalize(t, ScriptedBox(exits={"tar --null": 7})))
    assert t.info["state_error"] == "state_too_large"
    with pytest.raises(RuntimeError, match="could not clear"):
        asyncio.run(task.finalize(trace(), ScriptedBox(exits={"tar --null": 90})))


def test_the_capture_prunes_harbors_dirs_and_reads_ctimes():
    script = task_module.CAPTURE_SCRIPT
    for pruned in ("./proc", "./sys", "./dev", "./logs", "./tests", "./opt/tb-verifier"):
        assert f"-path {pruned}" in script
    assert "-cnewer /var/lib/reliquary-tb/work-start" in script
    assert "--no-recursion" in script
