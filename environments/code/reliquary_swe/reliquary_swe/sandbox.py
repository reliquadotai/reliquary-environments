"""reliquary-swe's tasks as signed-episode sandbox tasks (reliquary-sandbox).

A gateway lists `reliquary-swe = "reliquary_swe.sandbox:sandbox_task"` in `episode_envs`.
Per task:

* prepare (the agent's box): the exact cleanup `SweTask.setup` runs
  (`taskset.cleanup_script`), then `refs/reliquary/base` -> HEAD and the image's
  untracked files in `.git/reliquary-untracked`. verifiers keeps both in host memory;
  a gateway keeps no env memory across a restart, so they live in the box, where the
  agent can rewrite them -- and its git config (`diff.external`) too. The diff is
  therefore agent-controlled: it may name any path, the image's untracked and ignored
  files included. Grading is what makes that safe, not the capture: it runs in a
  pristine box and refuses, for every split, a patch touching a path untracked or
  ignored at the base or under `.venv` (`grading._forbidden_patch_paths`);
* extract (the agent's box): the diff `vf.capture_patch` takes against that base,
  without the image's untracked files. Published to the miner as the episode's state
  (its `final_diff`);
* grade (a pristine box of the same image): `grading.grade`, unchanged, with its
  untracked-path refusal and every anti-tamper restoration.

Agent-attributable outcomes are values, not exceptions: a git refusal in the agent's
box, or a box whose git stopped answering because the agent wrecked it (`rm -rf` of its
workdir), is an empty diff (graded like verifiers grades it), and an untracked list the
agent deleted, replaced or inflated is an empty list. The step deadline and real box
faults are never swallowed: they reach the sandbox, which records them itself. A
pristine box that is not what the corpus says, found before the agent's patch touches it
(`grading.PristineBoxError`: checking out the base, stripping history, setting hidden
tests aside, listing untracked paths), is ours: `grade` raises the sandbox's
`EnvInfraError` and the episode is aborted, logged and counted per env. Any other
exception from `grade` is an env bug the gateway grades 0.0 `grading_failed` and counts
per env, so watch both counters.

reliquary-sandbox is imported only inside `sandbox_task` and `grade`: it is not a
dependency of this public package, and a gateway always has it.

At import this module refuses any verifiers but the pinned one (`VERIFIERS_COMMIT`, read
from the installed distribution's `direct_url.json`) and any change to the verifiers
modules the hooks rely on (`PINNED_VERIFIERS_MODULES`: `capture_patch` and
`snapshot_untracked`): a gateway with another verifiers fails at start.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.utils.git import snapshot_untracked

from reliquary_swe import corpus, grading, swesmith_adapter
from reliquary_swe.taskset import PROMPT, SweData, cleanup_script, task_for

VERIFIERS_COMMIT = "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"
PINNED_VERIFIERS_MODULES = {
    "verifiers.v1.utils.git": "3134629cebc6e0475d7e375a8479056548a11e6849dfb9f944eb608f38c4d4d0",
}
"""sha256 of each verifiers module whose behaviour the hooks rely on, at VERIFIERS_COMMIT."""


def installed_direct_url() -> str | None:
    try:
        return importlib.metadata.distribution("verifiers").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_verifiers_sources() -> dict[str, bytes]:
    return {name: Path(importlib.util.find_spec(name).origin).read_bytes()
            for name in PINNED_VERIFIERS_MODULES}


def require_pinned_verifiers(direct_url: str | None, sources: dict[str, bytes]) -> None:
    """Raise ImportError unless verifiers was installed from VERIFIERS_COMMIT and each
    pinned module is byte-identical to the one checked."""
    try:
        commit = json.loads(direct_url or "")["vcs_info"]["commit_id"]
    except (ValueError, KeyError, TypeError):
        commit = None
    if commit != VERIFIERS_COMMIT:
        raise ImportError(f"reliquary_swe.sandbox needs verifiers at {VERIFIERS_COMMIT} "
                          f"(installed from git); found {commit or 'no pinned commit'}")
    for name, digest in PINNED_VERIFIERS_MODULES.items():
        if hashlib.sha256(sources.get(name, b"")).hexdigest() != digest:
            raise ImportError(f"verifiers' {name} differs from the pinned one; re-check "
                              "reliquary_swe.sandbox against it before raising the pin")


# At import: a gateway loads this module when it starts.
require_pinned_verifiers(installed_direct_url(), installed_verifiers_sources())

ENV = "reliquary-swe"
TOOLS = ("bash", "edit")
BASE_REF = "refs/reliquary/base"
UNTRACKED_FILE = ".git/reliquary-untracked"
MAX_UNTRACKED_BYTES = 1024 * 1024
MAX_STATE_BYTES = 8 * 1024 * 1024
"""capture_patch caps a patch at 2 MB; its UTF-8 re-encoding can grow up to 3x."""
PREPARE_TIMEOUT_S = 600.0
GRADING_TIMEOUT_S = 810.0
"""The largest grading timeout plan 1's verification window allows (2g + 180 <= 1800)."""
GIB = 1024**3
LIMITS: dict[str, int] = {"memory_bytes": 4 * GIB, "disk_bytes": 10 * GIB, "pids": 1024,
                          "wall_s": 3600, "per_call_timeout_s": 600}
"""wall_s is the package's agent timeout (taskset._AGENT_TIMEOUT_SECONDS); memory is
provisional: no SWE box has declared any before (Docker's default is unlimited)."""

_RECORD_BASE = f' && git -C "$WORKDIR" update-ref {BASE_REF} HEAD'
"""Appended to the cleanup with `&&`, never `;`: R2E's cleanup ends on
`test ! -e /r2e_tests && test ! -e "$WORKDIR"/run_tests.sh`, a list `set -e` does not
exit on, so after a `;` a later command's success would hide a hidden-test leak."""


@dataclass(frozen=True)
class SplitRef:
    corpus: str
    num_images: int | None = None


def parse_split(split: str) -> SplitRef:
    if split == "eval":
        raise ValueError("reliquary-swe: SWE-bench Verified is an evaluation set; it is "
                         "never served on training sandboxes")
    if split in ("r2e", "polyglot"):
        return SplitRef(split)
    if split == "train":
        return SplitRef("train", corpus.DEFAULT_SWESMITH_IMAGES)
    if split.startswith("train:"):
        count = split.split(":", 1)[1]
        if count.isdigit() and str(int(count)) == count:
            corpus._check_swesmith_num_images(int(count))
            return SplitRef("train", int(count))
    raise ValueError(f"reliquary-swe serves train, train:<num_images>, r2e and polyglot on "
                     f"sandboxes, not {split!r}")


@functools.cache
def _polyglot_digests() -> dict[str, str]:
    path = files("reliquary_swe").joinpath("polyglot_digests.json")
    return json.loads(path.read_text()) if path.is_file() else {}


def row_for(split: str, index: int) -> tuple[SplitRef, corpus.SweRow]:
    ref = parse_split(split)
    if type(index) is not int or index < 0:
        raise IndexError(index)
    if ref.corpus == "train":
        # Built alone: the whole split's rows weigh ~4 GB (`corpus.swesmith_order`).
        row = corpus.swesmith_row_at(ref.num_images, index)
        swesmith_adapter.ensure_python_profile(row.repo)
    elif ref.corpus == "r2e":
        row = corpus.r2e_row_at(index)
    else:
        # One cached load of the whole split: `load_polyglot_rows` caches per count, so
        # asking for `index + 1` rows would keep a new prefix alive for every index served.
        rows = corpus.load_polyglot_rows(None)
        if index >= len(rows):
            raise IndexError(index)
        row = rows[index]
        image_of(row)  # a tag without a pinned digest is refused at resolve, by every entry
    return ref, row


def image_of(row: corpus.SweRow) -> str:
    if row.image is not None and "@sha256:" in row.image:
        return row.image
    pinned = _polyglot_digests().get(row.image or "")
    if pinned is None:
        raise ValueError(f"{row.instance_id}: image {row.image!r} has no pinned digest; "
                         "run scripts/pin_polyglot_digests.py")
    return pinned


@dataclass(frozen=True)
class Declaration:
    image: str
    workdir: str
    data: SweData
    tools: tuple[str, ...] = TOOLS
    grading_workdir: str | None = None
    publish_state: bool = True
    max_state_bytes: int = MAX_STATE_BYTES
    prepare_timeout_s: float = PREPARE_TIMEOUT_S
    grading_timeout_s: float = GRADING_TIMEOUT_S
    limits: dict[str, int] = field(default_factory=lambda: dict(LIMITS))


def declaration(split: str, index: int) -> Declaration:
    ref, row = row_for(split, index)
    return Declaration(image=image_of(row), workdir=row.workdir,
                       data=task_for(row, index, ref.corpus).data)


async def prepare(runtime: Any, *, data: SweData) -> str:
    script = cleanup_script(data.split) + _RECORD_BASE
    result = await runtime.run(["sh", "-c", script],
                               {"BASE_COMMIT": data.base_commit, "WORKDIR": data.workdir})
    if result.exit_code != 0:
        raise RuntimeError(
            f"environment preparation failed for {data.instance_id} (exit {result.exit_code}): "
            f"{(result.stderr or result.stdout).strip()[-500:]}")
    listed = "\0".join(await snapshot_untracked(runtime)).encode()
    if len(listed) > MAX_UNTRACKED_BYTES:
        # extract reads at most this much back; a longer list would silently stop
        # ignoring the image's files. An image this shape needs a closer look first.
        raise RuntimeError(
            f"environment preparation failed for {data.instance_id}: the image's untracked "
            f"list is {len(listed)} bytes, over {MAX_UNTRACKED_BYTES}")
    await runtime.write(f"{data.workdir}/{UNTRACKED_FILE}", listed)
    return (result.stdout or "") + (result.stderr or "")


async def extract(runtime: Any, *, data: SweData) -> bytes:
    base = ""
    head = await runtime.run(["git", "rev-parse", "--verify", "-q", BASE_REF], {})
    if head.exit_code == 0:
        base = head.stdout.strip()
    try:
        listed = await runtime.read(f"{data.workdir}/{UNTRACKED_FILE}",
                                    max_bytes=MAX_UNTRACKED_BYTES)
        ignore = [path for path in listed.decode("utf-8", "replace").split("\0") if path]
    except TimeoutError:
        raise  # the step deadline (an OSError subclass): the sandbox's extract_timeout
    except OSError:
        # Missing (FileNotFoundError), not a regular file (StateUnreadable) or inflated
        # (FileTooLarge): the agent's doing, so its diff simply carries the image's
        # untracked files and fails to apply in the pristine box, like verifiers'.
        ignore = []
    trace = SimpleNamespace(info={})
    try:
        await vf.capture_patch(trace, runtime, base_commit=base, ignore=ignore)
    except vf.SandboxError:
        # git failed and `true` failed too: the box answers but cannot run in its workdir
        # (the agent removed it). A box fault proper raised from our runtime instead, and
        # the sandbox recorded it before this point.
        return b""
    return str(trace.info.get("patch", "")).encode("utf-8")


def swe_facts(report: grading.Report, data: SweData) -> dict[str, Any]:
    """`SweEnv.finalize`'s `swe_report`, without the per-test map (it can be large)."""
    return {
        "applied": report.applied,
        "restored": report.restored,
        "fail_to_pass_passed": report.fail_to_pass_passed,
        "fail_to_pass_total": len(data.fail_to_pass),
        "pass_to_pass_passed": report.pass_to_pass_passed,
        "pass_to_pass_total": len(data.pass_to_pass),
        "results_parsed": report.results_parsed,
        "test_command_exit_code": report.test_command_exit_code,
        "test_output_tail": report.test_output_tail,
    }


async def grade(runtime: Any, state: bytes, *, data: SweData):
    from reliquary_sandbox.episode_task import EnvInfraError, GradeResult

    try:
        report = await grading.grade(runtime, data, state.decode("utf-8", "replace"))
    except grading.PristineBoxError as exc:
        # Raised only before the agent's patch touches the box: ours, so aborted.
        raise EnvInfraError(str(exc)) from exc
    return GradeResult(report.reward, swe_facts(report, data))


def sandbox_task(split: str, index: int):
    from reliquary_sandbox.episode_task import SandboxTask, TaskLimits

    found = declaration(split, index)
    return SandboxTask(
        image=found.image, workdir=found.workdir, tools=found.tools,
        prepare=functools.partial(prepare, data=found.data),
        extract=functools.partial(extract, data=found.data),
        grade=functools.partial(grade, data=found.data),
        max_state_bytes=found.max_state_bytes, prepare_timeout_s=found.prepare_timeout_s,
        grading_timeout_s=found.grading_timeout_s, grading_workdir=found.grading_workdir,
        publish_state=found.publish_state, limits=TaskLimits(**found.limits))


def sandbox_prompt(split: str, index: int) -> str:
    """The first user message, exactly; it never travels through the sandbox."""
    _, row = row_for(split, index)
    return PROMPT.format(workdir=row.workdir, problem_statement=row.problem_statement)


def sandbox_index(split: str, instance_id: str) -> int:
    ref = parse_split(split)
    if ref.corpus == "train":
        ids = [instance_id for _, instance_id in corpus.swesmith_order(ref.num_images)]
    elif ref.corpus == "r2e":
        ids = corpus.r2e_instance_ids()
    else:
        ids = [row.instance_id for row in corpus.load_polyglot_rows(None)]
    return ids.index(instance_id)


def sandbox_images(split: str, num_tasks: int | None = None) -> list[str]:
    ref = parse_split(split)
    if ref.corpus == "train":
        count = len(corpus.swesmith_order(ref.num_images))
        indexes = range(count if num_tasks is None else min(num_tasks, count))
        rows = (corpus.swesmith_row_at(ref.num_images, index) for index in indexes)
    elif ref.corpus == "r2e":
        rows = corpus.load_r2e_rows(num_tasks)
    else:
        rows = corpus.load_polyglot_rows(num_tasks)
    return list(dict.fromkeys(image_of(row) for row in rows))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m reliquary_swe.sandbox",
        description="Image manifests for sandbox hosts (also warms the row caches).")
    commands = parser.add_subparsers(dest="command", required=True)
    images = commands.add_parser("images", help="print the split's images as a JSON manifest")
    images.add_argument("--split", required=True)
    images.add_argument("--num-tasks", type=int, default=None)
    args = parser.parse_args(argv)
    manifest = {"env": ENV, "split": args.split, "num_tasks": args.num_tasks,
                "images": sandbox_images(args.split, args.num_tasks)}
    json.dump(manifest, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
