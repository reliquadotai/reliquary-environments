"""finalize against real git, in a scratch repository (no Docker): whatever git
configuration or repository content the agent leaves behind, nothing of it runs during
the capture, and a patch it planted never travels. Each case arms a hostile setting
whose command would plant a patch and leave a mark; the capture must run none of them
and write the agent's honest diff (or, when git cannot capture, nothing at all)."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from reliquary_swe import corpus, taskset
from reliquary_swe.taskset import BASE_REF, UNTRACKED_FILE, task_for

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


class LocalBox:
    """A box on this machine: commands run in the workdir with a minimal environment
    (PATH and a HOME of the test's own), files are read and written in place."""

    def __init__(self, workdir: Path, home: Path):
        self.workdir, self.home = workdir, home
        self.config = SimpleNamespace(type="reliquary-sandbox", workdir=str(workdir))

    async def run(self, argv, env):
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=self.workdir, env={"PATH": os.environ["PATH"], "HOME": str(self.home),
                                          **env},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        return SimpleNamespace(exit_code=proc.returncode, stdout=out.decode(errors="replace"),
                               stderr=err.decode(errors="replace"))

    async def read(self, path, max_bytes=None):
        data = Path(path).read_bytes()
        if max_bytes is not None and len(data) > max_bytes:
            raise OSError(27, "File too large")
        return data

    async def write(self, path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(bytes(data))


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                   check=True, capture_output=True, env={"PATH": os.environ["PATH"],
                                                         "HOME": str(cwd),
                                                         "GIT_CONFIG_NOSYSTEM": "1"})


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A repository at its base (base ref and untracked list as setup leaves them), an
    image-shipped untracked file, then the agent's work: an edit and a commit."""
    w, home = tmp_path / "repo", tmp_path / "home"
    w.mkdir()
    home.mkdir()
    patch = tmp_path / "artifacts" / "patch.diff"
    monkeypatch.setattr(taskset, "PATCH_PATH", str(patch))
    git(w, "init", "-q")
    (w / "a.py").write_text("hi\n")
    git(w, "add", "a.py")
    git(w, "commit", "-qm", "base")
    git(w, "update-ref", BASE_REF, "HEAD")
    (w / "shipped.txt").write_text("from the image\n")
    (w / UNTRACKED_FILE).write_bytes(b"shipped.txt")
    (w / "a.py").write_text("hi\nedit\n")
    (w / "c.py").write_text("committed\n")
    git(w, "add", "c.py")
    git(w, "commit", "-qm", "agent")
    # The agent's payload: a script that plants a patch and leaves a mark.
    mark = tmp_path / "ran"
    plant = tmp_path / "plant.sh"
    plant.write_text(f"#!/bin/sh\nmkdir -p {patch.parent}\nprintf PLANTED > {patch}\n"
                     f"echo \"$0 $*\" >> {mark}\nexit 1\n")
    plant.chmod(0o755)
    row = dataclasses.replace(
        corpus.SweRow(instance_id="i", repo="o/r", problem_statement="p", fail_to_pass=(),
                      pass_to_pass=(), gold_patch="g", base_commit="HEAD",
                      image="r@sha256:" + "a" * 64),
        workdir=str(w))
    return SimpleNamespace(w=w, home=home, patch=patch, mark=mark, plant=plant,
                           task=task_for(row, 0, "train"), box=LocalBox(w, home),
                           tmp=tmp_path)


def trace(task):
    return vf.Trace(agent=vf.AgentInfo(config=vf.AgentConfig()),
                    task=vf.TraceTask(type="SweTask", data=task.data))


async def finalize(world):
    t = trace(world.task)
    await world.task.finalize(t, world.box)
    return t


def assert_honest(world, t):
    assert not world.mark.exists(), world.mark.read_text()
    raw = world.patch.read_bytes()
    assert b"PLANTED" not in raw
    assert b"+edit" in raw and b"+committed" in raw and b"shipped.txt" not in raw
    assert trace_patch(t) == raw


def trace_patch(t):
    return t.info["patch"].encode()


def config(world, text):
    with open(world.w / ".git" / "config", "a") as f:
        f.write(text)


def hook(directory: Path, plant: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("post-index-change", "pre-commit", "post-checkout", "reference-transaction"):
        (directory / name).write_text(f"#!/bin/sh\nexec {plant} hook-{name}\n")
        (directory / name).chmod(0o755)


async def test_an_honest_capture_is_the_diff_against_the_base(world):
    assert_honest(world, await finalize(world))
    clone = world.tmp / "clone"
    git(world.tmp, "clone", "-q", str(world.w), str(clone))
    git(clone, "checkout", "-q", "HEAD~1")
    git(clone, "apply", "--check", str(world.patch))  # applies at the base


@pytest.mark.parametrize("kind", ["clean", "process"])
async def test_a_filter_the_agent_configured_does_not_run(world, kind):
    config(world, f'[filter "zz"]\n\t{kind} = {world.plant}\n\trequired = true\n')
    (world.w / ".gitattributes").write_text("* filter=zz\n")
    world.patch.parent.mkdir()
    world.patch.write_bytes(b"PLANTED before")
    assert_honest(world, await finalize(world))


async def test_attributes_under_git_info_do_not_run_a_filter(world):
    config(world, f'[filter "zz"]\n\tclean = {world.plant}\n\trequired = true\n')
    (world.w / ".git" / "info").mkdir(exist_ok=True)
    (world.w / ".git" / "info" / "attributes").write_text("* filter=zz\n")
    assert_honest(world, await finalize(world))


async def test_fsmonitor_does_not_run(world):
    config(world, f"[core]\n\tfsmonitor = {world.plant}\n")
    assert_honest(world, await finalize(world))


async def test_hooks_do_not_run(world):
    hook(world.w / ".git" / "hooks", world.plant)
    elsewhere = world.tmp / "hooks"
    hook(elsewhere, world.plant)
    config(world, f"[core]\n\thooksPath = {elsewhere}\n")
    assert_honest(world, await finalize(world))


async def test_diff_drivers_textconv_and_external_diff_do_not_run(world):
    config(world, f'[diff "zz"]\n\tcommand = {world.plant}\n\ttextconv = {world.plant}\n'
                  f"[diff]\n\texternal = {world.plant}\n")
    (world.w / ".gitattributes").write_text("* diff=zz\n")
    assert_honest(world, await finalize(world))


async def test_an_included_config_does_not_reach_the_capture(world):
    included = world.tmp / "included"
    included.write_text(f'[core]\n\tfsmonitor = {world.plant}\n'
                        f'[filter "zz"]\n\tclean = {world.plant}\n\trequired = true\n')
    config(world, f"[include]\n\tpath = {included}\n")
    (world.w / ".gitattributes").write_text("* filter=zz\n")
    assert_honest(world, await finalize(world))


async def test_global_and_xdg_config_do_not_reach_the_capture(world):
    hostile = (f'[core]\n\tfsmonitor = {world.plant}\n\thooksPath = {world.tmp / "hooks"}\n'
               f'[filter "zz"]\n\tclean = {world.plant}\n\trequired = true\n')
    hook(world.tmp / "hooks", world.plant)
    (world.home / ".gitconfig").write_text(hostile)
    (world.home / ".config" / "git").mkdir(parents=True)
    (world.home / ".config" / "git" / "config").write_text(hostile)
    (world.w / ".gitattributes").write_text("* filter=zz\n")
    assert_honest(world, await finalize(world))


async def test_a_worktree_redirected_by_config_is_ignored(world):
    elsewhere = world.tmp / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "a.py").write_text("PLANTED\n")
    config(world, f"[core]\n\tworktree = {elsewhere}\n")
    assert_honest(world, await finalize(world))


async def test_a_replace_ref_in_the_agents_repository_does_not_reach_the_capture(world):
    # Replacing the base commit by an empty one would put every file in the diff. The
    # scratch repository has no refs of the agent's, replace refs included: this checks
    # that design, not GIT_NO_REPLACE_OBJECTS (which only guards the base lookup).
    empty = subprocess.run(["git", "commit-tree", "4b825dc642cb6eb9a060e54bf8d69288fbee4904",
                            "-m", "x"], cwd=world.w, capture_output=True, text=True,
                           env={"PATH": os.environ["PATH"], "GIT_AUTHOR_NAME": "t",
                                "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                                "GIT_COMMITTER_EMAIL": "t@t"}, check=True).stdout.strip()
    base = subprocess.run(["git", "rev-parse", BASE_REF], cwd=world.w, capture_output=True,
                          text=True, check=True).stdout.strip()
    git(world.w, "update-ref", f"refs/replace/{base}", empty)
    assert_honest(world, await finalize(world))
    assert b"+++ b/a.py" in world.patch.read_bytes()
    assert b"--- /dev/null\n+++ b/a.py" not in world.patch.read_bytes()


async def test_a_nested_repository_with_hooks_runs_nothing(world):
    nested = world.w / "sub"
    nested.mkdir()
    git(nested, "init", "-q")
    (nested / "f").write_text("x\n")
    git(nested, "add", "f")
    git(nested, "commit", "-qm", "n")
    with open(nested / ".git" / "config", "a") as f:
        f.write(f"[core]\n\tfsmonitor = {world.plant}\n")
    hook(nested / ".git" / "hooks", world.plant)
    await finalize(world)
    assert not world.mark.exists()
    assert b"PLANTED" not in world.patch.read_bytes()


async def test_a_stale_index_lock_neither_breaks_the_capture_nor_lets_a_plant_travel(world):
    (world.w / ".git" / "index.lock").write_text("")
    world.patch.parent.mkdir()
    world.patch.write_bytes(b"PLANTED before")
    assert_honest(world, await finalize(world))


async def test_a_git_file_pointing_elsewhere_captures_nothing_and_runs_nothing(world):
    moved = world.tmp / "moved.git"
    (world.w / ".git").rename(moved)
    (world.w / ".git").write_text(f"gitdir: {moved}\n")
    with open(moved / "config", "a") as f:
        f.write(f'[core]\n\tfsmonitor = {world.plant}\n'
                f'[filter "zz"]\n\tclean = {world.plant}\n\trequired = true\n')
    (world.w / ".gitattributes").write_text("* filter=zz\n")
    world.patch.parent.mkdir()
    world.patch.write_bytes(b"PLANTED before")
    t = await finalize(world)
    assert not world.mark.exists()
    assert not world.patch.exists() and "patch" not in t.info and t.info["patch_error"]


@pytest.mark.parametrize("break_it", ["objects", "moved"])
async def test_a_broken_repository_lets_no_planted_patch_travel(world, break_it):
    if break_it == "objects":
        shutil.rmtree(world.w / ".git" / "objects")
    else:
        (world.w / ".git").rename(world.tmp / "moved.git")
    world.patch.parent.mkdir()
    world.patch.write_bytes(b"PLANTED before")
    t = await finalize(world)
    assert not world.patch.exists() and "patch" not in t.info and t.info["patch_error"]


async def test_the_capture_leaves_no_scratch_repository_behind(world, monkeypatch):
    seen = []
    run = world.box.run

    async def spy(argv, env):
        seen.append(env.get("GIT_DIR", ""))
        return await run(argv, env)

    monkeypatch.setattr(world.box, "run", spy)
    await finalize(world)
    scratch = {d for d in seen if d.startswith("/tmp/reliquary-capture-")}
    assert scratch and not any(Path(d).exists() for d in scratch)


async def test_a_base_on_a_missing_object_of_a_promisor_remote_fetches_nothing(world):
    # Peeling a missing object in a partial clone lazily fetches it from the agent's
    # remote, through a transport command of the agent's (here core.sshCommand). The base
    # is looked up without reading any object; the scratch repository, which has no
    # remote, peels it: the capture fails closed and nothing planted travels.
    config(world, "[core]\n\trepositoryformatversion = 1\n"
                  f"\tsshCommand = {world.plant}\n"
                  "[extensions]\n\tpartialClone = origin\n"
                  '[remote "origin"]\n\turl = ssh://somewhere/repo\n\tpromisor = true\n')
    (world.w / ".git" / "refs" / "reliquary" / "base").write_text("1" * 40 + "\n")
    world.patch.parent.mkdir()
    world.patch.write_bytes(b"PLANTED before")
    t = await finalize(world)
    assert not world.mark.exists(), world.mark.read_text()
    assert not world.patch.exists() and "patch" not in t.info and t.info["patch_error"]


def isolated_git(cwd, home, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                   check=True, capture_output=True,
                   env={"PATH": os.environ["PATH"], "HOME": str(home),
                        "GIT_CONFIG_NOSYSTEM": "1"})


@pytest.fixture
def image(tmp_path, monkeypatch):
    """An image's repository before setup: its base commit and a file it ships untracked.
    setup's cleanup is left out (`true`): only the base ref, the untracked list and the
    check of an untouched box run."""
    w, home = tmp_path / "repo", tmp_path / "home"
    w.mkdir()
    home.mkdir()
    monkeypatch.setattr(taskset, "cleanup_script", lambda split: "true")
    isolated_git(w, home, "init", "-q")
    (w / "a.py").write_text("hi\n")
    isolated_git(w, home, "add", "a.py")
    isolated_git(w, home, "commit", "-qm", "base")
    (w / "shipped.txt").write_text("from the image\n")
    row = dataclasses.replace(
        corpus.SweRow(instance_id="i", repo="o/r", problem_statement="p", fail_to_pass=(),
                      pass_to_pass=(), gold_patch="g", base_commit="HEAD",
                      image="r@sha256:" + "a" * 64),
        workdir=str(w))
    task = task_for(row, 0, "train")
    return SimpleNamespace(w=w, home=home, task=task, box=LocalBox(w, home), tmp=tmp_path)


async def setup(image):
    await image.task.setup(trace(image.task), image.box)


async def test_setup_of_an_untouched_image_captures_an_empty_patch(image, monkeypatch):
    patch = image.tmp / "artifacts" / "patch.diff"
    monkeypatch.setattr(taskset, "PATCH_PATH", str(patch))
    await setup(image)
    assert (image.w / UNTRACKED_FILE).read_bytes() == b"shipped.txt"
    t = trace(image.task)
    await image.task.finalize(t, image.box)
    assert t.info["patch"] == "" and patch.read_bytes() == b""


@pytest.mark.parametrize("where", ["global", "local"])
async def test_an_excludes_file_outside_the_capture_neither_hides_nor_widens(image, where,
                                                                             monkeypatch):
    # An excludes file the capture never reads (a global one, or the repository's own
    # core.excludesFile) hid the image's build.log from the untracked list, and the
    # capture then credited it to the agent: an honest agent graded 0. setup lists under
    # the capture's own ignore rules.
    (image.w / "build.log").write_text("image build output\n")
    (image.tmp / "ignore").write_text("*.log\n")
    if where == "global":
        (image.home / ".gitconfig").write_text(f"[core]\n\texcludesFile = {image.tmp}/ignore\n")
    else:
        with open(image.w / ".git" / "config", "a") as f:
            f.write(f"[core]\n\texcludesFile = {image.tmp}/ignore\n")
    patch = image.tmp / "artifacts" / "patch.diff"
    monkeypatch.setattr(taskset, "PATCH_PATH", str(patch))
    await setup(image)
    assert set((image.w / UNTRACKED_FILE).read_bytes().split(b"\0")) == {
        b"build.log", b"shipped.txt"}
    t = trace(image.task)
    await image.task.finalize(t, image.box)
    assert t.info["patch"] == ""


def _filemode(image):
    with open(image.w / ".git" / "config", "a") as f:
        f.write("[core]\n\tfilemode = false\n")
    (image.w / "a.py").chmod(0o755)


def _autocrlf(image):
    with open(image.w / ".git" / "config", "a") as f:
        f.write("[core]\n\tautocrlf = true\n")
    (image.w / "a.py").unlink()
    isolated_git(image.w, image.home, "checkout", "--", "a.py")
    assert (image.w / "a.py").read_bytes() == b"hi\r\n"


def _lfs_outside_the_repository(image):
    # A filter defined in the global config (where `git lfs install` puts it): the
    # committed blob is the pointer, the work tree holds the content.
    (image.home / ".gitconfig").write_text(
        '[filter "lfs"]\n\tclean = sed s/CONTENT/POINTER/\n'
        "\tsmudge = sed s/POINTER/CONTENT/\n\trequired = true\n")
    (image.w / ".gitattributes").write_text("*.bin filter=lfs\n")
    (image.w / "big.bin").write_text("CONTENT\n")
    isolated_git(image.w, image.home, "add", ".gitattributes", "big.bin")
    isolated_git(image.w, image.home, "commit", "-qm", "lfs")


@pytest.mark.parametrize("arm", [_filemode, _autocrlf, _lfs_outside_the_repository],
                         ids=["filemode", "autocrlf", "lfs"])
async def test_an_image_the_capture_sees_differently_fails_setup(image, arm):
    # The image's own git reads its box as clean, the capture (no config of the image's)
    # does not: every honest agent would carry that difference and grade 0. setup finds
    # it on the untouched box and fails as ours.
    arm(image)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=image.w, capture_output=True, text=True, check=True,
                            env={"PATH": os.environ["PATH"], "HOME": str(image.home),
                                 "GIT_CONFIG_NOSYSTEM": "1"}).stdout
    assert status == ""
    with pytest.raises(RuntimeError, match="untouched box"):
        await setup(image)


async def test_an_exclude_file_that_is_a_fifo_does_not_hang_the_capture(world):
    exclude = world.w / ".git" / "info" / "exclude"
    exclude.parent.mkdir(exist_ok=True)
    exclude.unlink(missing_ok=True)
    os.mkfifo(exclude)
    try:
        t = await asyncio.wait_for(finalize(world), 30)
    finally:
        try:  # a git still blocked on the FIFO reads EOF and exits
            os.close(os.open(exclude, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass
    assert_honest(world, t)
