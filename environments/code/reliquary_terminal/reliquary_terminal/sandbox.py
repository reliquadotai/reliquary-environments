"""reliquary-terminal's training tasks as signed-episode sandbox tasks (reliquary-sandbox).

A gateway lists `reliquary-terminal = "reliquary_terminal.sandbox:sandbox_task"`. Per
task (the path `TerminalEnv.finalize` -> `HarborEnv._grade` takes today, step by step):

* prepare (the agent's box): `TerminalTask.setup`. Nothing for MiMo rows; tmax's setup
  bundle in role "agent" (Task 12);
* extract (the agent's box): the artifact roots that exist, through `runtime.archive`
  (never the image's tar), framed as `STATE_MAGIC` + `{"roots": [...]}` + newline +
  archive. A root the agent removed is listed as absent (verifiers' `required=False`):
  graded, never an error;
* grade (a pristine box, workdir `/`): `TerminalTask(verifier_box_data(data)).setup`
  in role "grade", `rm -rf` of absent roots, one `runtime.restore_archive` of the
  present roots, `_stage_tests(wipe=True)`, then `_graded` (anti-hack guard, test.sh,
  reward.txt, CTRF), all unchanged. A root the agent replaced by a symlink travels as
  that link (an absolute target outside the roots stays an inert link, so the tests
  fail on it); an archive the restore refuses (a relative link leaving the roots) is
  graded 0 here, with the reason under `state_error`.

Rows that need the network are never served (boxes have none). Environment values
must be literals: a `${VAR}` template would be resolved against the gateway's own
environment. The verifier's box must be the agent's image with the task's env and no
healthcheck: a sandbox grades in a pristine box of `image`, with `env`. Terminal-Bench
2.1 (`eval`) grades in the agent's box by design and is never served.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from verifiers.v1.tasksets.harbor.taskset import HarborData, verifier_box_data

from reliquary_terminal.grading import TerminalTask
from reliquary_terminal.harness import DEFAULT_COMMAND_TIMEOUT_SECONDS
from reliquary_terminal.taskset import TerminalConfig, load_train_rows, train_data

ENV = "reliquary-terminal"
TOOLS = ("bash",)
STATE_MAGIC = b"reliquary-terminal-state/1\n"
MAX_HEADER_BYTES = 4096
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_STATE_BYTES = MAX_ARCHIVE_BYTES + 64 * 1024
PREPARE_TIMEOUT_S = 600.0
GRADING_TIMEOUT_S = 600.0
COMMAND_TIMEOUT_S = int(DEFAULT_COMMAND_TIMEOUT_SECONDS)
FACT_TESTS = 20
MIB = 1024**2
_TEMPLATE = re.compile(r"\$\{")
_PRESENT = 'for r do if [ -e "$r" ] || [ -L "$r" ]; then printf "%s\\n" "$r"; fi; done'


@functools.cache
def _mimo_digests() -> dict[str, str]:
    path = files("reliquary_terminal").joinpath("mimo_digests.json")
    return json.loads(path.read_text()) if path.is_file() else {}


def literal_env(env: Mapping[str, str], what: str) -> dict[str, str]:
    for name, value in env.items():
        if _TEMPLATE.search(value):
            raise ValueError(f"{what} {name} is a template; sandbox tasks take literal values only")
    return dict(env)


def encode_state(present: list[str], archive: bytes) -> bytes:
    header = json.dumps({"roots": present}, separators=(",", ":")).encode()
    return STATE_MAGIC + header + b"\n" + archive


def decode_state(state: bytes, declared: tuple[str, ...]) -> tuple[list[str], bytes]:
    if not state.startswith(STATE_MAGIC):
        raise ValueError("not a reliquary-terminal state")
    header, newline, archive = state[len(STATE_MAGIC):].partition(b"\n")
    if not newline or len(header) > MAX_HEADER_BYTES:
        raise ValueError("the state header is malformed")
    parsed = json.loads(header)
    roots = parsed.get("roots") if isinstance(parsed, dict) else None
    if (not isinstance(roots, list) or roots != [r for r in declared if r in roots]
            or len(set(roots)) != len(roots)):
        raise ValueError("the state names roots the task does not declare, or out of order")
    if bool(roots) != bool(archive):
        raise ValueError("the state's roots and archive disagree")
    return roots, archive


@dataclass(frozen=True)
class Declaration:
    image: str
    workdir: str
    data: HarborData
    roots: tuple[str, ...]
    env: dict[str, str]
    limits: dict[str, Any]
    tools: tuple[str, ...] = TOOLS
    grading_workdir: str = "/"
    publish_state: bool = False
    max_state_bytes: int = MAX_STATE_BYTES
    prepare_timeout_s: float = PREPARE_TIMEOUT_S
    grading_timeout_s: float = GRADING_TIMEOUT_S


def _checked(data: HarborData) -> HarborData:
    verifier = data.verifier
    unsupported = [name for name, present in (
        ("healthcheck", data.healthcheck is not None), ("collect hooks", bool(data.collect)),
        ("environment upload", data.upload_environment),
        ("separate verifier missing", verifier is None),
        ("verifier image", verifier is not None and verifier.image not in (None, data.image)),
        ("verifier env", verifier is not None and verifier.env != data.env),
        ("verifier healthcheck", verifier is not None and verifier.healthcheck is not None),
    ) if present]
    if unsupported:
        raise ValueError(f"{data.name}: not servable on a sandbox ({', '.join(unsupported)})")
    literal_env(data.env, "env")
    literal_env(data.verifier_env, "verifier env")
    return data


def _train(index: int) -> Declaration:
    rows = load_train_rows()
    if type(index) is not int or not 0 <= index < len(rows):
        raise IndexError(index)
    row = rows[index]
    if row.get("allow_internet"):
        raise ValueError(f"{row['instance_id']} needs the network; sandbox boxes have none")
    data = _checked(train_data(row, index, TerminalConfig(id=ENV, split="train")))
    image = _mimo_digests().get(data.image or "")
    if image is None:
        raise ValueError(f"{data.name}: image {data.image!r} has no pinned digest; run "
                         "scripts/pin_mimo_digests.py")
    limits = {"cpus": float(row["cpus"]), "memory_bytes": int(row["memory_mb"]) * MIB,
              "disk_bytes": int(row["storage_mb"]) * MIB,
              "wall_s": int(math.ceil(row["agent_timeout_sec"])),
              "per_call_timeout_s": COMMAND_TIMEOUT_S}
    return Declaration(image=image, workdir=data.workdir, data=data,
                       roots=tuple(a.source for a in data.artifacts), env=dict(data.env),
                       limits=limits)


def declaration(split: str, index: int) -> Declaration:
    if split == "train":
        return _train(index)
    if split == "eval":
        raise ValueError("Terminal-Bench 2.1 grades in the agent's box by design; it is "
                         "never served on sandboxes")
    raise ValueError(f"reliquary-terminal serves train on sandboxes, not {split!r}")


async def prepare(runtime: Any, *, data: HarborData) -> str:
    await TerminalTask(data).setup(runtime)
    return ""


async def extract(runtime: Any, *, roots: tuple[str, ...]) -> bytes:
    listed = await runtime.run(["sh", "-c", _PRESENT, "roots", *roots], {})
    if listed.exit_code != 0:
        raise RuntimeError("the agent's box could not list its state roots")
    present_lines = set(listed.stdout.splitlines())
    present = [root for root in roots if root in present_lines]
    archive = await runtime.archive(present, MAX_ARCHIVE_BYTES) if present else b""
    return encode_state(present, archive)


class _GradingTrace:
    """What `TerminalTask._graded` writes to: `info["grading"]` and metrics."""

    def __init__(self) -> None:
        self.info: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}

    def record_metrics(self, values: Mapping[str, float]) -> None:
        self.metrics.update(values)


def _trimmed(grading: Any) -> Any:
    if not isinstance(grading, dict):
        return grading
    trimmed = dict(grading)
    ctrf = trimmed.get("ctrf")
    if isinstance(ctrf, dict):
        trimmed["ctrf"] = {**ctrf, "tests": list(ctrf.get("tests", []))[:FACT_TESTS]}
    return trimmed


async def grade(runtime: Any, state: bytes, *, data: HarborData, roots: tuple[str, ...]):
    from reliquary_sandbox.episode_task import GradeResult

    try:
        present, archive = decode_state(state, roots)
    except ValueError as exc:
        return GradeResult(0.0, {"state_error": str(exc)[:200]})
    grader = TerminalTask(verifier_box_data(data))
    grader.setup_role = "grade"
    await grader.setup(runtime)
    for root in roots:
        if root not in present:
            removed = await runtime.run(["rm", "-rf", "--", root], {})
            if removed.exit_code != 0:
                raise RuntimeError(f"could not remove {root} in the grading box")
    if present:
        try:
            await runtime.restore_archive(archive, present)
        except OSError as exc:
            # The sandbox refused the agent's archive (StateUnreadable, FileTooLarge, both
            # OSError): the agent's outcome. Infrastructure faults are recorded by the
            # runtime itself and still abort the episode.
            return GradeResult(0.0, {"state_error": str(exc)[:200]})
    await grader._stage_tests(runtime, wipe=True)
    trace = _GradingTrace()
    score = await grader._graded(runtime, trace)
    reward = score.get("reward") if isinstance(score, dict) else score
    facts = {"grading": _trimmed(trace.info.get("grading")), "metrics": trace.metrics}
    if (isinstance(reward, bool) or not isinstance(reward, (int, float))
            or not math.isfinite(reward) or not 0.0 <= reward <= 1.0):
        return GradeResult(0.0, {**facts, "reward_out_of_range": repr(reward)[:100]})
    return GradeResult(float(reward), facts)


def sandbox_task(split: str, index: int):
    from reliquary_sandbox.episode_task import SandboxTask, TaskLimits

    found = declaration(split, index)
    return SandboxTask(
        image=found.image, workdir=found.workdir, tools=found.tools, env=found.env,
        prepare=functools.partial(prepare, data=found.data),
        extract=functools.partial(extract, roots=found.roots),
        grade=functools.partial(grade, data=found.data, roots=found.roots),
        max_state_bytes=found.max_state_bytes, prepare_timeout_s=found.prepare_timeout_s,
        grading_timeout_s=found.grading_timeout_s, grading_workdir=found.grading_workdir,
        publish_state=found.publish_state, limits=TaskLimits(**found.limits))


def sandbox_prompt(split: str, index: int) -> str:
    """The first user message, exactly; it never travels through the sandbox."""
    return declaration(split, index).data.prompt


def sandbox_images(split: str, num_tasks: int | None = None) -> list[str]:
    if split != "train":
        raise ValueError(f"no image list for {split!r}")
    rows = load_train_rows()
    count = len(rows) if num_tasks is None else num_tasks
    images = [declaration(split, index).image for index in range(min(count, len(rows)))
              if not rows[index].get("allow_internet")]  # never served, never pulled
    return list(dict.fromkeys(images))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m reliquary_terminal.sandbox")
    commands = parser.add_subparsers(dest="command", required=True)
    images = commands.add_parser("images", help="print the split's images as a JSON manifest")
    images.add_argument("--split", required=True)
    images.add_argument("--num-tasks", type=int, default=None)
    args = parser.parse_args(argv)
    json.dump({"env": ENV, "split": args.split, "num_tasks": args.num_tasks,
               "images": sandbox_images(args.split, args.num_tasks)}, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
