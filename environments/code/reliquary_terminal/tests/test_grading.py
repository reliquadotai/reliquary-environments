"""Per-test results kept next to the reward, without a container: a fake box
plays `test.sh`."""

from __future__ import annotations

import json

import pytest
import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask

from reliquary_terminal import TerminalEnv, grading
from reliquary_terminal.grading import TerminalTask

CTRF = {
    "results": {
        "tool": {"name": "pytest"},
        "summary": {"tests": 3, "passed": 1, "failed": 1, "skipped": 1},
        "tests": [
            {"name": "test_a", "status": "passed", "duration": 1},
            {"name": "test_b", "status": "failed", "duration": 2, "message": "assert 1 == 2" + "x" * 5000},
            {"name": "test_c", "status": "skipped", "duration": 0},
        ],
    }
}


class FakeBox:
    """Files and `test.sh` behaviour in place of a container."""

    def __init__(self, reward: str | None, ctrf: bytes | None, output="ran tests\n", exit_code=1):
        self.files: dict[str, bytes] = {"/logs/verifier/ctrf.json": b'{"stale": true}'}
        self.reward, self.ctrf, self.output, self.exit_code = reward, ctrf, output, exit_code
        self.runs: list[list[str]] = []

    async def run(self, argv, env):
        self.runs.append(list(argv))
        if argv[:2] == ["rm", "-f"]:
            for path in argv[2:]:
                self.files.pop(path, None)
            return ProgramResult(0, "", "")
        if argv == ["bash", "/tests/test.sh"]:
            if self.reward is not None:
                self.files["/logs/verifier/reward.txt"] = self.reward.encode()
            if self.ctrf is not None:
                self.files["/logs/verifier/ctrf.json"] = self.ctrf
            return ProgramResult(self.exit_code, self.output, "guard: REJECT planted conftest.py\n")
        raise AssertionError(f"unexpected command {argv}")

    async def read(self, path, max_bytes=None):
        if path not in self.files:
            raise SandboxError(f"read {path!r}: No such file")
        return self.files[path]


def _task() -> TerminalTask:
    return TerminalTask(HarborData(idx=0, name="t", prompt="p"))


def _trace(task) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
    )


async def test_failed_tests_are_named_and_the_reward_is_unchanged():
    task = _task()
    trace = _trace(task)
    box = FakeBox(reward="0\n", ctrf=json.dumps(CTRF).encode())
    assert await task._graded(box, trace) == 0.0
    # The same reward verifiers' own HarborTask reads from the same box.
    plain = HarborTask(task.data)
    assert await plain._graded(FakeBox(reward="0\n", ctrf=None), _trace(plain)) == 0.0
    grading_info = trace.info["grading"]
    assert grading_info["exit_code"] == 1
    assert "ran tests" in grading_info["output_tail"]
    assert "REJECT planted conftest.py" in grading_info["output_tail"]
    report = grading_info["ctrf"]
    assert report["counts"] == {"passed": 1, "failed": 1, "skipped": 1}
    assert [t["name"] for t in report["tests"]] == ["test_a", "test_b", "test_c"]
    failed = report["tests"][1]
    assert failed["message"].startswith("assert 1 == 2")
    assert len(failed["message"]) == grading.MAX_MESSAGE_CHARS
    assert "message" not in report["tests"][0]
    assert trace.metrics == {"tests_total": 3.0, "tests_passed": 1.0, "tests_failed": 1.0}


async def test_a_passing_run_keeps_reward_one():
    task = _task()
    trace = _trace(task)
    ctrf = {"results": {"tests": [{"name": "t1", "status": "passed"}]}}
    box = FakeBox(reward="1\n", ctrf=json.dumps(ctrf).encode(), exit_code=0)
    assert await task._graded(box, trace) == 1.0
    assert trace.info["grading"]["ctrf"]["counts"] == {"passed": 1}


async def test_a_stale_report_is_never_read_as_this_runs():
    # test.sh stopped before pytest (the anti-hack guard), so it wrote no
    # report; the one already in the box must not stand in for it.
    task = _task()
    trace = _trace(task)
    box = FakeBox(reward="0\n", ctrf=None)
    assert await task._graded(box, trace) == 0.0
    assert box.runs[0] == ["rm", "-f", grading.CTRF]
    assert trace.info["grading"]["ctrf"] is None
    assert "No such file" in trace.info["grading"]["ctrf_error"]
    assert trace.metrics == {}


async def test_a_malformed_report_is_an_error_not_a_crash():
    task = _task()
    trace = _trace(task)
    box = FakeBox(reward="1\n", ctrf=b"{not json", exit_code=0)
    assert await task._graded(box, trace) == 1.0
    assert trace.info["grading"]["ctrf"] is None
    assert trace.info["grading"]["ctrf_error"].startswith("JSONDecodeError")


def test_summarize_rejects_what_is_not_ctrf():
    with pytest.raises((KeyError, ValueError)):
        grading.summarize_ctrf(b'{"results": {}}')
    with pytest.raises(ValueError):
        grading.summarize_ctrf(b'{"results": {"tests": 3}}')


def test_both_splits_yield_terminal_tasks():
    config = vf.taskset_config_type("reliquary-terminal")
    for split in ("eval", "train"):
        tasks = list(vf.load_taskset(config(id="reliquary-terminal", split=split)))
        assert tasks and all(type(t) is TerminalTask for t in tasks)


async def test_the_separate_grading_box_is_graded_as_a_terminal_task(monkeypatch):
    env = TerminalEnv(
        vf.env_config_type("reliquary-terminal")(
            taskset=vf.taskset_config_type("reliquary-terminal")(id="reliquary-terminal", split="train"),
            verifier_runtime=vf.DockerConfig(),
        )
    )
    task = next(iter(vf.load_taskset(env.config.taskset)))
    seen = []

    async def fake_grade(config, grader, solution):
        seen.append(grader)
        return 1.0

    monkeypatch.setattr(env, "_grade", fake_grade)
    solver = _trace(task)
    solver.ok = True
    episode = vf.Episode(task=solver.task, traces=[solver])
    await env.finalize(task, episode)
    assert len(seen) == 1 and type(seen[0]) is TerminalTask
    assert seen[0].data.workdir == "/"
    assert solver.rewards["solved"].value == 1.0


async def test_a_deeply_nested_report_cannot_change_the_outcome():
    # json.loads raises RecursionError (not ValueError) on this.
    task = _task()
    trace = _trace(task)
    box = FakeBox(reward="1\n", ctrf=b"[" * 100000, exit_code=0)
    assert await task._graded(box, trace) == 1.0
    assert trace.info["grading"]["ctrf"] is None
    assert trace.info["grading"]["ctrf_error"].startswith("RecursionError")


async def test_the_stored_tests_are_capped_but_counted_in_full():
    task = _task()
    trace = _trace(task)
    many = grading.MAX_TESTS + 500
    tests = [{"name": "t" * 10_000, "status": "failed", "message": "m"} for _ in range(many)]
    box = FakeBox(reward="0\n", ctrf=json.dumps({"results": {"tests": tests}}).encode())
    assert await task._graded(box, trace) == 0.0
    report = trace.info["grading"]["ctrf"]
    assert len(report["tests"]) == grading.MAX_TESTS
    assert report["omitted"] == 500
    assert report["counts"] == {"failed": many}
    assert all(len(t["name"]) == grading.MAX_NAME_CHARS for t in report["tests"])
    assert trace.metrics["tests_total"] == float(many)


async def test_a_failing_ledger_never_fails_the_rollout(monkeypatch, caplog):
    from verifiers.v1.runtimes import DockerRuntime

    from reliquary_terminal import containers

    def broken(name, *args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(containers, "record", broken)
    box = DockerRuntime(vf.DockerConfig())
    await _task().setup(box)  # nothing else to set up for this task
    assert any("could not record container" in r.message for r in caplog.records)
