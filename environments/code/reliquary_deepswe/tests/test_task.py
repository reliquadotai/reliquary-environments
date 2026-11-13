"""The task's hooks: the capture against real git, the reward against a scripted box."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary_deepswe import overlay, source, task as task_module
from reliquary_deepswe.task import RUN_SCRIPT, STAGE_SCRIPT
from reliquary_deepswe.taskset import DeepSWEConfig, DeepSWETaskset


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class LocalBox:
    """A verifiers-shaped runtime over this machine's shell (tests only)."""

    config = SimpleNamespace(type="reliquary-sandbox")

    async def run(self, argv, env):
        done = subprocess.run(argv, env={**os.environ, **env}, capture_output=True, text=True)
        return SimpleNamespace(exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    work = tmp_path / "app"
    work.mkdir()
    git(work, "init", "-q")
    (work / "a.py").write_text("x = 1\n")
    git(work, "add", "-A")
    git(work, "commit", "-qm", "base")
    base = git(work, "rev-parse", "HEAD")
    patch = tmp_path / "logs" / "artifacts" / "model.patch"
    monkeypatch.setattr(task_module, "WORKDIR", str(work))
    monkeypatch.setattr(task_module, "PATCH_PATH", str(patch))
    task = DeepSWETaskset(DeepSWEConfig(split="eval_all", id="reliquary-deepswe")).task_at(0)
    task.data = task.data.model_copy(update={"base_commit": base})
    return SimpleNamespace(work=work, base=base, patch=patch, task=task, tmp=tmp_path)


def finalize(repo):
    trace = SimpleNamespace(info={})
    asyncio.run(repo.task.finalize(trace, LocalBox()))
    return trace


def test_setup_accepts_the_base_and_refuses_another_head(repo):
    asyncio.run(repo.task.setup(LocalBox()))
    abbreviated = repo.task.data.model_copy(update={"base_commit": repo.base[:7]})
    asyncio.run(type(repo.task)(abbreviated).setup(LocalBox()))
    (repo.work / "a.py").write_text("x = 2\n")
    git(repo.work, "commit", "-qam", "moved")
    with pytest.raises(RuntimeError, match="not the base"):
        asyncio.run(repo.task.setup(LocalBox()))


def test_committed_work_is_captured(repo):
    (repo.work / "a.py").write_text("x = 2\n")
    (repo.work / "b.py").write_text("y = 1\n")
    git(repo.work, "checkout", "-qb", "feature")
    git(repo.work, "add", "-A")
    git(repo.work, "commit", "-qm", "work")
    (repo.work / "c.py").write_text("uncommitted\n")
    trace = finalize(repo)
    patch = repo.patch.read_text()
    assert "patch_error" not in trace.info
    assert "+x = 2" in patch and "b/b.py" in patch and "c.py" not in patch
    subprocess.run(["git", "checkout", "-q", repo.base], cwd=repo.work, check=True)
    subprocess.run(["git", "clean", "-fdq"], cwd=repo.work, check=True)
    subprocess.run(["git", "apply", "--check", str(repo.patch)], cwd=repo.work, check=True)


def test_uncommitted_work_is_an_empty_patch(repo):
    (repo.work / "a.py").write_text("x = 2\n")
    finalize(repo)
    assert repo.patch.read_bytes() == b""


def test_a_planted_patch_never_travels(repo):
    repo.patch.parent.mkdir(parents=True)
    repo.patch.write_text("planted")
    finalize(repo)
    assert repo.patch.read_bytes() == b""


def test_a_planted_link_is_replaced_by_a_real_directory(repo):
    target = repo.tmp / "elsewhere"
    target.mkdir()
    (repo.tmp / "logs").mkdir()
    (repo.tmp / "logs" / "artifacts").symlink_to(target)
    finalize(repo)
    assert not repo.patch.parent.is_symlink() and repo.patch.is_file()
    assert not any(target.iterdir())


def test_the_agents_git_settings_never_run(repo):
    marker = repo.tmp / "ran"
    hook = repo.tmp / "hook.sh"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    hook.chmod(0o755)
    (repo.work / "a.py").write_text("x = 2\n")
    git(repo.work, "commit", "-qam", "work")
    for key, value in (("core.fsmonitor", hook), ("diff.external", hook),
                       ("core.hooksPath", repo.tmp), ("filter.zz.clean", hook),
                       ("filter.zz.required", "true"), ("diff.zz.textconv", hook),
                       ("core.pager", hook)):
        git(repo.work, "config", key, str(value))
    (repo.work / ".gitattributes").write_text("* filter=zz diff=zz\n")
    finalize(repo)
    assert not marker.exists()
    assert "+x = 2" in repo.patch.read_text()


def test_no_repository_is_no_patch(repo):
    (repo.work / ".git").rename(repo.tmp / "moved")
    trace = finalize(repo)
    assert trace.info["patch_error"] == "no_head" and not repo.patch.exists()


def test_a_patch_over_the_cap_is_dropped(repo, monkeypatch):
    monkeypatch.setattr(task_module, "MAX_PATCH_BYTES", 100)
    (repo.work / "a.py").write_text("x = 2\n" * 100)
    git(repo.work, "commit", "-qam", "big")
    trace = finalize(repo)
    assert trace.info["patch_error"] == "patch_too_large" and not repo.patch.exists()


# --- the reward, against a scripted grading box -------------------------------

def ctrf_task() -> str:
    for task_id in source.task_ids():
        config = overlay.effective_config(task_id)
        if (config["grade"]["format"] == "ctrf" and config["grade"]["node_id"] == "name"
                and source.task_toml(task_id)["metadata"]["language"] != "go"):
            return task_id
    raise AssertionError("no CTRF task")


class ScriptedBox:
    """Files in a dict; `test.sh` runs write what `outcomes` says, one per run."""

    config = SimpleNamespace(type="reliquary-sandbox")

    def __init__(self, patch: bytes | None, outcomes: list):
        self.files = {} if patch is None else {task_module.PATCH_PATH: patch}
        self.outcomes = list(outcomes)
        self.runs: list[str] = []

    async def write(self, path, data):
        self.files[path] = data

    async def read(self, path, max_bytes=None):
        if path not in self.files:
            raise FileNotFoundError(path)
        data = self.files[path]
        if max_bytes is not None and len(data) > max_bytes:
            raise OSError("too large")
        return data

    async def run(self, argv, env):
        command = argv[-1] if argv[:2] == ["sh", "-c"] else " ".join(argv)
        self.runs.append(command)
        if command == STAGE_SCRIPT:
            self.files = {k: v for k, v in self.files.items() if not k.startswith("/logs/verifier")}
        elif command == RUN_SCRIPT:
            assert env == task_module.GRADING_ENV
            self.files.update(self.outcomes.pop(0))
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


def outcome(config, passing: bool, message: str = "AssertionError"):
    tests = [{"name": i, "status": "passed"} for i in config["p2p_node_ids"]]
    tests += [{"name": i, "status": "passed" if passing else "failed", "message": message}
              for i in config["f2p_node_ids"]]
    report = json.dumps({"results": {"tests": tests}}).encode()
    return {path: report for path in config["grade"]["reports"]}


def solve(task_id, box):
    tasks = DeepSWETaskset(DeepSWEConfig(split="eval_all", id="reliquary-deepswe",
                                         tasks=[task_id]))
    task = next(iter(tasks.load()))
    trace = SimpleNamespace(info={}, metrics={})
    trace.record_metrics = trace.metrics.update
    reward = asyncio.run(task.solved(box, trace))
    return reward, trace


GOOD_PATCH = b"diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-a\n+b\n"


def test_passing_reports_grade_1():
    task_id = ctrf_task()
    config = overlay.effective_config(task_id)
    box = ScriptedBox(GOOD_PATCH, [outcome(config, True)])
    reward, trace = solve(task_id, box)
    assert reward == 1.0 and trace.metrics["f2p"] == 1.0
    assert box.runs.count(RUN_SCRIPT) == 1
    assert box.files["/tmp/deepswe-tests.tgz"] == overlay.tests_archive(task_id)


def test_a_reward_file_in_the_box_is_never_read():
    task_id = ctrf_task()
    config = overlay.effective_config(task_id)
    forged = {**outcome(config, False), "/logs/verifier/reward.json": b'{"reward": 1}',
              "/logs/verifier/reward.txt": b"1"}
    reward, _ = solve(task_id, ScriptedBox(GOOD_PATCH, [forged]))
    assert reward == 0.0


@pytest.mark.parametrize("patch, metric", [
    (None, "empty_patch"), (b"  \n", "empty_patch"),
    (b"diff --git a/node_modules/x b/node_modules/x\n", "patch_refused"),
])
def test_no_tests_run_for_an_empty_or_refused_patch(patch, metric):
    box = ScriptedBox(patch, [])
    reward, trace = solve(ctrf_task(), box)
    assert reward == 0.0 and trace.metrics[metric] == 1.0 and RUN_SCRIPT not in box.runs


def test_an_oversized_patch_is_0():
    box = ScriptedBox(b"x" * (task_module.MAX_PATCH_BYTES + 1), [])
    reward, trace = solve(ctrf_task(), box)
    assert reward == 0.0 and trace.metrics["patch_unreadable"] == 1.0


def timeout_outcome(config, retried_ok: bool):
    """Every f2p passes; every p2p fails on a timeout (or passes on a retry)."""
    tests = [{"name": i, "status": "passed"} for i in config["f2p_node_ids"]]
    tests += [{"name": i, "status": "passed" if retried_ok else "failed",
               "message": "TimeoutError: exceeded"} for i in config["p2p_node_ids"]]
    report = json.dumps({"results": {"tests": tests}}).encode()
    return {path: report for path in config["grade"]["reports"]}


def p2p_task() -> str:
    for task_id in source.task_ids():
        config = overlay.effective_config(task_id)
        if (config["p2p_node_ids"] and config["grade"]["format"] == "ctrf"
                and config["grade"]["node_id"] == "name"
                and source.task_toml(task_id)["metadata"]["language"] != "go"):
            return task_id
    raise AssertionError


def test_a_timeout_only_failure_recovers_when_both_retries_pass():
    task_id = p2p_task()
    config = overlay.effective_config(task_id)
    box = ScriptedBox(GOOD_PATCH, [timeout_outcome(config, False), timeout_outcome(config, True),
                                   timeout_outcome(config, True)])
    reward, trace = solve(task_id, box)
    assert reward == 1.0 and trace.metrics["retry_recovered"] == 1.0
    assert trace.metrics["timeout_retries"] == 2.0


def test_a_retry_that_fails_keeps_the_first_run():
    task_id = p2p_task()
    config = overlay.effective_config(task_id)
    box = ScriptedBox(GOOD_PATCH, [timeout_outcome(config, False), timeout_outcome(config, False)])
    reward, trace = solve(task_id, box)
    assert reward == 0.0 and trace.metrics["timeout_retries"] == 1.0
    assert box.runs.count(RUN_SCRIPT) == 2


def test_no_retry_when_the_budget_cannot_hold_two_more_runs(monkeypatch):
    task_id = p2p_task()
    config = overlay.effective_config(task_id)
    monkeypatch.setattr(task_module, "RETRY_BUDGET_S", 0.0)
    box = ScriptedBox(GOOD_PATCH, [timeout_outcome(config, False)])
    reward, trace = solve(task_id, box)
    assert reward == 0.0 and trace.metrics["retry_skipped_for_budget"] == 1.0


def test_graded_elsewhere_records_nothing():
    task = DeepSWETaskset(DeepSWEConfig(split="eval_all", id="reliquary-deepswe")).task_at(0)
    assert asyncio.run(task.graded_elsewhere().solved(ScriptedBox(None, []), None)) == {}


def test_the_verifier_runs_from_the_staged_tests():
    """test.sh calls `python3 /tests/grader.py prepare|grade`: grader.py is the shared
    verifier, importing the task's own grader as `upstream_grader`."""
    staged = overlay.staged_tests(source.task_ids()[0])
    assert b"python3 /tests/grader.py prepare" in staged["test.sh"]
    assert b"import upstream_grader" in staged["grader.py"]
    assert Path(overlay.VERIFIER).name == "verifier.py"
