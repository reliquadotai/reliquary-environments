"""The task's hooks: the capture against real git, the reward against a scripted box."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary_swebench_pro import scoring, source, task as task_module
from reliquary_swebench_pro.task import APPLY_SCRIPT, RUN_SCRIPT, STAGE_SCRIPT
from reliquary_swebench_pro.taskset import SweProConfig, SweProTaskset


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class LocalBox:
    """A verifiers-shaped runtime over this machine's shell (tests only)."""

    config = SimpleNamespace(type="reliquary-sandbox")

    async def run(self, argv, env):
        done = subprocess.run(argv, env={**os.environ, **env}, capture_output=True, text=True)
        return SimpleNamespace(exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr)

    async def write(self, path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)


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
    task = SweProTaskset(SweProConfig(split="eval_all", id="reliquary-swebench-pro")).task_at(0)
    task.data = task.data.model_copy(update={"base_commit": base})
    return SimpleNamespace(work=work, base=base, patch=patch, task=task, tmp=tmp_path)


def finalize(repo):
    trace = SimpleNamespace(info={})
    asyncio.run(repo.task.finalize(trace, LocalBox()))
    return trace


def test_setup_accepts_the_base_and_refuses_another_head(repo):
    asyncio.run(repo.task.setup(LocalBox()))
    (repo.work / "a.py").write_text("x = 2\n")
    git(repo.work, "commit", "-qam", "moved")
    with pytest.raises(RuntimeError, match="not the base"):
        asyncio.run(repo.task.setup(LocalBox()))


def apply_in_pristine_clone(repo):
    clone = repo.tmp / "pristine"
    subprocess.run(["git", "clone", "-q", str(repo.work), str(clone)], check=True)
    subprocess.run(["git", "checkout", "-q", repo.base], cwd=clone, check=True)
    subprocess.run(["git", "apply", str(repo.patch)], cwd=clone, check=True)
    return clone


def test_committed_and_uncommitted_work_are_both_captured(repo):
    asyncio.run(repo.task.setup(LocalBox()))
    (repo.work / "a.py").write_text("x = 2\n")
    (repo.work / "b.py").write_text("y = 1\n")
    git(repo.work, "checkout", "-qb", "feature")
    git(repo.work, "add", "-A")
    git(repo.work, "commit", "-qm", "work")
    (repo.work / "c.py").write_text("uncommitted\n")
    (repo.work / "a.py").write_text("x = 3\n")
    trace = finalize(repo)
    patch = repo.patch.read_text()
    assert "patch_error" not in trace.info
    assert "+x = 3" in patch and "b/b.py" in patch and "b/c.py" in patch
    clone = apply_in_pristine_clone(repo)
    assert (clone / "a.py").read_text() == "x = 3\n" and (clone / "c.py").exists()


def test_uncommitted_work_scores_as_committed_work(repo):
    asyncio.run(repo.task.setup(LocalBox()))
    (repo.work / "a.py").write_text("x = 2\n")
    finalize(repo)
    uncommitted = repo.patch.read_bytes()
    git(repo.work, "commit", "-qam", "work")
    finalize(repo)
    assert uncommitted == repo.patch.read_bytes() and b"+x = 2" in uncommitted


def test_a_deletion_is_captured(repo):
    (repo.work / "a.py").unlink()
    finalize(repo)
    assert "deleted file mode" in repo.patch.read_text()
    assert not (apply_in_pristine_clone(repo) / "a.py").exists()


def test_an_untouched_box_is_an_empty_patch(repo):
    (repo.work / "shipped untracked [1].txt").write_text("from the image\n")
    asyncio.run(repo.task.setup(LocalBox()))
    trace = finalize(repo)
    assert "patch_error" not in trace.info and repo.patch.read_bytes() == b""


def test_the_images_untracked_files_and_toolchain_directories_are_left_out(repo):
    (repo.work / "shipped.txt").write_text("from the image\n")
    asyncio.run(repo.task.setup(LocalBox()))
    (repo.work / "shipped.txt").write_text("changed by the agent\n")
    for path in (".gocache/00/x", "node_modules/m/index.js", "sub/__pycache__/m.pyc",
                 ".venv/lib/x.pth"):
        (repo.work / path).parent.mkdir(parents=True, exist_ok=True)
        (repo.work / path).write_text("x\n")
    (repo.work / "src").mkdir()
    (repo.work / "src" / "target").write_text("a source file named target\n")
    finalize(repo)
    patch = repo.patch.read_text()
    assert "src/target" in patch
    for left_out in ("shipped.txt", ".gocache", "node_modules", "__pycache__", ".venv"):
        assert left_out not in patch


def test_exclude_patterns_match_paths_literally():
    from reliquary_swebench_pro.task import exclude_patterns

    assert exclude_patterns(["a*b", "[x]", "d/e ", "#f"]) == "/a\\*b\n/\\[x]\n/d/e\\ \n/#f\n"
    with pytest.raises(ValueError):
        exclude_patterns(["bad\nname"])


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
    assert trace.info["patch_error"] == "no_repository" and not repo.patch.exists()


def test_a_patch_over_the_cap_is_dropped(repo, monkeypatch):
    monkeypatch.setattr(task_module, "MAX_PATCH_BYTES", 100)
    (repo.work / "a.py").write_text("x = 2\n" * 100)
    git(repo.work, "commit", "-qam", "big")
    trace = finalize(repo)
    assert trace.info["patch_error"] == "patch_too_large" and not repo.patch.exists()


# --- the reward, against a scripted grading box -------------------------------

QUTE = "instance_qutebrowser__qutebrowser-77c3557995704a683cdb67e2a3055f7547fa22c3-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d"


class ScriptedBox:
    """Files in a dict; each `test.sh` run writes the logs `outcomes` gives, in order."""

    config = SimpleNamespace(type="reliquary-sandbox")

    def __init__(self, patch: bytes | None, outcomes: list, apply_exit: int = 0):
        self.files = {} if patch is None else {task_module.PATCH_PATH: patch}
        self.outcomes = list(outcomes)
        self.apply_exit = apply_exit
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
        if command == APPLY_SCRIPT:
            return SimpleNamespace(exit_code=self.apply_exit, stdout="", stderr="rejected")
        if command == STAGE_SCRIPT:
            self.files = {k: v for k, v in self.files.items() if not k.startswith("/logs/verifier")}
        elif command == RUN_SCRIPT:
            assert env == task_module.GRADING_ENV
            self.files.update(self.outcomes.pop(0))
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


def logs(statuses: dict[str, str]) -> dict[str, bytes]:
    stdout = "".join(f"{name} {status} [ 50%]\n" for name, status in statuses.items())
    return {scoring.STDOUT_LOG: stdout.encode(), scoring.STDERR_LOG: b""}


def required(task_id=QUTE):
    config = source.config(task_id)
    return scoring.test_names(config["fail_to_pass"]) | scoring.test_names(config["pass_to_pass"])


def solve(box, task_id=QUTE):
    tasks = SweProTaskset(SweProConfig(split="eval_all", id="reliquary-swebench-pro",
                                       tasks=[task_id]))
    task = next(iter(tasks.load()))
    trace = SimpleNamespace(info={}, metrics={})
    trace.record_metrics = trace.metrics.update
    return asyncio.run(task.solved(box, trace)), trace


GOOD_PATCH = b"diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-a\n+b\n"


def test_passing_logs_grade_1_after_apply_and_stage():
    box = ScriptedBox(GOOD_PATCH, [logs({n: "PASSED" for n in required()})])
    reward, trace = solve(box)
    assert reward == 1.0 and trace.metrics["f2p_passed"] == trace.metrics["f2p_total"]
    assert box.runs.index(APPLY_SCRIPT) < box.runs.index(STAGE_SCRIPT) < box.runs.index(RUN_SCRIPT)
    assert box.files[task_module.TESTS_ARCHIVE] == task_module.tests_archive(QUTE)


def test_a_reward_file_in_the_box_is_never_read():
    one = sorted(required())[0]
    forged = {**logs({n: "FAILED" if n == one else "PASSED" for n in required()}),
              "/logs/verifier/reward.txt": b"1", "/logs/verifier/output.json":
              json.dumps({"tests": [{"name": n, "status": "PASSED"} for n in required()]}).encode()}
    reward, _ = solve(ScriptedBox(GOOD_PATCH, [forged]))
    assert reward == 0.0


def test_logs_missing_after_the_run_are_0():
    reward, trace = solve(ScriptedBox(GOOD_PATCH, [{}]))
    assert reward == 0.0 and trace.metrics["evaluation_valid"] == 0.0


def test_a_patch_that_does_not_apply_is_0_and_runs_no_test():
    box = ScriptedBox(GOOD_PATCH, [], apply_exit=1)
    reward, trace = solve(box)
    assert reward == 0.0 and trace.metrics["patch_failed"] == 1.0 and RUN_SCRIPT not in box.runs


@pytest.mark.parametrize("patch, metric", [
    (None, "empty_patch"), (b"  \n", "empty_patch"),
    (b"diff --git a/node_modules/x b/node_modules/x\n", "patch_refused"),
])
def test_no_tests_run_for_an_empty_or_refused_patch(patch, metric):
    box = ScriptedBox(patch, [])
    reward, trace = solve(box)
    assert reward == 0.0 and trace.metrics[metric] == 1.0
    assert APPLY_SCRIPT not in box.runs and RUN_SCRIPT not in box.runs


def test_an_oversized_patch_is_0():
    box = ScriptedBox(b"x" * (task_module.MAX_PATCH_BYTES + 1), [])
    reward, trace = solve(box)
    assert reward == 0.0 and trace.metrics["patch_unreadable"] == 1.0


def test_graded_elsewhere_records_nothing():
    task = SweProTaskset(SweProConfig(split="eval_all", id="reliquary-swebench-pro")).task_at(0)
    assert asyncio.run(task.graded_elsewhere().solved(ScriptedBox(None, []), None)) == {}


def test_the_staged_tests_are_upstreams_tests_dir():
    import gzip, io, tarfile
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(task_module.tests_archive(QUTE)))) as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        modes = {m.name: m.mode for m in tar.getmembers()}
    assert members == source.tests_files(QUTE)
    assert modes["test.sh"] == 0o755 and modes["config.json"] == 0o644
    assert task_module.tests_archive.__wrapped__(QUTE) == task_module.tests_archive(QUTE)


def test_the_replay_applies_as_scale_does():
    assert "git apply --verbose" in APPLY_SCRIPT and "git apply --3way" in APPLY_SCRIPT
    assert "patch --batch --fuzz=3 -p1" in APPLY_SCRIPT
    assert RUN_SCRIPT.endswith("bash /tests/test.sh > /tmp/swepro-test-stdout.txt 2>&1")
