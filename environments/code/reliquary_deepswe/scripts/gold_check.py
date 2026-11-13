"""Grade every task's gold patch offline through this package's own hooks: the gate.

Run on a host shaped like a signed-episode gateway (Docker with gVisor's `runsc`, the
images pre-pulled by digest, e.g. `scripts/sandbox_images.py pull` over this package's
lock), never on a production gateway. Per task, as the sandbox bridge runs an episode:

1. agent box: a container of the pinned image, no network, the task's CPUs and memory;
   `Task.setup`; the reference calls (`conformance.reference_calls`: upload the gold
   patch, apply it, commit it) through `bash`; `Task.finalize` (the capture);
2. grading box: a fresh container of the same image; `Task.grading_setup`; the captured
   `/logs/artifacts/model.patch` restored; `Task.solved`, timed.

The boxes present themselves as a sandbox's (`config.type`), so the hooks take exactly
the paths they take on a gateway; unlike a gateway, processes are not killed between
commands. Output: one JSON line per task (reward, metrics, phase seconds, error), then a
summary. `--write-gate` records the tasks graded 1 within `--budget` seconds in
`reliquary_deepswe/gate.py` (the served `eval` split); a new gate is a new package
version. `--conformance` runs the declared attack cases instead of the references.

    uv run python scripts/gold_check.py --runtime runsc --jobs 8 --out gold.jsonl
    uv run python scripts/gold_check.py --tasks etree-xml-diff-patch --runtime runsc
    uv run python scripts/gold_check.py --runtime runsc --write-gate --from gold.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import platform
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from reliquary_deepswe import conformance, gate, source
from reliquary_deepswe.task import PATCH_PATH, SANDBOX_RUNTIME, WORKDIR, DeepSWETask
from reliquary_deepswe.taskset import DeepSWEConfig, DeepSWETaskset

PACKAGE = Path(gate.__file__).resolve().parent
LOCK = json.loads((PACKAGE / "sandbox-images.lock.json").read_text())["images"]
STEP_S = 810.0
"""A gateway's grading step (the bridge's MAX_GRADING_TIMEOUT_S)."""


class Box:
    """A verifiers-shaped runtime over `docker exec` into one container."""

    def __init__(self, docker: str, container: str):
        self.docker = docker
        self.container = container
        self.config = SimpleNamespace(type=SANDBOX_RUNTIME, workdir=WORKDIR)

    async def _exec(self, argv: list[str], env: dict[str, str] | None = None,
                    stdin: bytes | None = None, workdir: str = WORKDIR):
        flags = ["-i"] if stdin is not None else []
        for key, value in (env or {}).items():
            flags += ["-e", f"{key}={value}"]
        process = await asyncio.create_subprocess_exec(
            self.docker, "exec", *flags, "-w", workdir, self.container, *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await process.communicate(stdin)
        return process.returncode, out, err

    async def run(self, argv, env):
        code, out, err = await self._exec(list(argv), env)
        return SimpleNamespace(exit_code=code, stdout=out.decode("utf-8", "replace"),
                               stderr=err.decode("utf-8", "replace"))

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        code, out, _ = await self._exec(
            ["sh", "-c", 'if [ -f "$1" ] && [ ! -L "$1" ]; then wc -c < "$1"; '
                         'elif [ -e "$1" ] || [ -L "$1" ]; then echo special; fi', "read", path],
            workdir="/")
        size = out.decode().strip()
        if code or not size:
            raise FileNotFoundError(path)
        if not size.isdigit() or (max_bytes is not None and int(size) > max_bytes):
            raise OSError(f"{path}: not a regular file, or over {max_bytes} bytes")
        code, out, err = await self._exec(["cat", "--", path], workdir="/")
        if code:
            raise OSError(f"{path}: {err.decode()[-200:]}")
        return out

    async def write(self, path: str, data: bytes) -> None:
        code, _, err = await self._exec(
            ["sh", "-c", 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"', "write", path],
            stdin=data, workdir="/")
        if code:
            raise OSError(f"write {path}: {err.decode()[-200:]}")


async def start(args, task: DeepSWETask) -> Box:
    image = LOCK[task.data.image]
    name = f"deepswe-gold-{task.key[:40]}-{uuid.uuid4().hex[:8]}"
    resources = task.data.resources
    argv = [args.docker, "run", "-d", "--name", name, "--network", "none",
            "--cpus", str(resources.cpu), "--memory", f"{int(resources.memory * 1024)}m",
            "--workdir", WORKDIR, "--entrypoint", "sleep"]
    if args.runtime:
        argv += ["--runtime", args.runtime]
    process = await asyncio.create_subprocess_exec(
        *argv, image, "infinity", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await process.communicate()
    if process.returncode:
        raise RuntimeError(f"docker run {image}: {err.decode()[-300:]}")
    return Box(args.docker, out.decode().strip())


async def stop(box: Box | None) -> None:
    if box is not None:
        process = await asyncio.create_subprocess_exec(
            box.docker, "rm", "-f", box.container,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await process.wait()


def new_trace() -> SimpleNamespace:
    trace = SimpleNamespace(info={}, metrics={})
    trace.record_metrics = trace.metrics.update
    return trace


async def play(args, task: DeepSWETask, calls: list) -> dict:
    """One episode: the calls in the agent's box, then grading in a fresh one."""
    row: dict = {"task_id": task.key, "language": task.data.language}
    agent = grading = None
    clock = time.monotonic()
    try:
        agent = await start(args, task)
        await asyncio.wait_for(task.setup(agent), task.data.timeout.setup)
        row["setup_s"] = round(time.monotonic() - clock, 1)
        clock = time.monotonic()
        for tool, arguments in calls:
            result = await agent.run(["bash", "-c", arguments["command"]], {})
            if result.exit_code:
                raise RuntimeError(f"{tool} call failed ({result.exit_code}): "
                                   f"{(result.stderr or result.stdout)[-300:]}")
        row["solve_s"] = round(time.monotonic() - clock, 1)
        clock = time.monotonic()
        trace = new_trace()
        await asyncio.wait_for(task.finalize(trace, agent), STEP_S)
        row["finalize_s"] = round(time.monotonic() - clock, 1)
        try:
            patch = await agent.read(PATCH_PATH)
        except FileNotFoundError:
            patch = None
        row["patch_bytes"] = None if patch is None else len(patch)
        if trace.info.get("patch_error"):
            row["patch_error"] = trace.info["patch_error"]
        await stop(agent)
        agent = None

        clock = time.monotonic()
        grading = await start(args, task)
        row["grading_box_s"] = round(time.monotonic() - clock, 1)
        step = time.monotonic()
        await asyncio.wait_for(task.grading_setup(grading), STEP_S)
        if patch is not None:
            await grading.write(PATCH_PATH, patch)
        trace = new_trace()
        reward = await asyncio.wait_for(task.solved(grading, trace),
                                        max(1.0, STEP_S - (time.monotonic() - step)))
        row["grading_s"] = round(time.monotonic() - step, 1)
        row["reward"] = reward
        row["metrics"] = trace.metrics
        if reward != 1:
            row["detail"] = trace.info.get("deepswe")
    except TimeoutError:
        row["error"] = "timeout: past the step budget"
        row.setdefault("reward", 0.0)
    except Exception as exc:  # noqa: BLE001 - recorded per task, the run goes on
        row["error"] = f"{type(exc).__name__}: {str(exc)[-500:]}"
        row.setdefault("reward", 0.0)
    finally:
        await stop(agent)
        await stop(grading)
    row["within_budget"] = row.get("grading_s", STEP_S) <= args.budget
    return row


def episodes(args) -> list[tuple[DeepSWETask, str, list, float]]:
    tasks = DeepSWETaskset(DeepSWEConfig(split="eval_all", id="reliquary-deepswe",
                                         tasks=args.tasks or None))
    by_index = {task.data.idx: task for task in tasks.load()}
    if args.conformance:
        return [(by_index[c["index"]], c["name"], c["calls"], c["expect"])
                for c in conformance.conformance_cases("eval_all") if c["index"] in by_index]
    return [(task, f"reference_{index}", conformance.reference_calls("eval_all", index), 1.0)
            for index, task in sorted(by_index.items())]


async def run_all(args) -> list[dict]:
    limit = asyncio.Semaphore(args.jobs)
    out = open(args.out, "a") if args.out else None

    async def one(task, name, calls, expect):
        async with limit:
            row = await play(args, task, calls)
            row.update(case=name, expect=expect, ok=row.get("reward") == expect)
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            if out is not None:
                out.write(line + "\n")
                out.flush()
            return row

    try:
        return await asyncio.gather(*(one(*e) for e in episodes(args)))
    finally:
        if out is not None:
            out.close()


def write_gate(rows: list[dict], args) -> None:
    passed = sorted(r["task_id"] for r in rows
                    if r.get("case", "").startswith("reference_") and r.get("reward") == 1
                    and r.get("grading_s", STEP_S) <= args.budget and not r.get("error"))
    digest = hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in rows).encode())
    provenance = {"host": platform.node(), "runtime": args.runtime or "docker default",
                  "date": datetime.date.today().isoformat(), "budget_s": str(args.budget),
                  "results_sha256": digest.hexdigest(),
                  "verified": f"{len(passed)}/{len(source.task_ids())}"}
    text = Path(gate.__file__).read_text()
    head = text.split("VERIFIED:", 1)[0]
    body = ("VERIFIED: frozenset[str] | None = frozenset({\n"
            + "".join(f"    {t!r},\n" for t in passed) + "})\n"
            '"""Task ids whose gold patch graded 1 offline within the budget; None: not gated yet."""\n\n'
            "GOLD_CHECK: dict[str, str] = {\n"
            + "".join(f"    {k!r}: {v!r},\n" for k, v in provenance.items()) + "}\n"
            '"""Provenance of `VERIFIED`: where and when the gold check ran (host, runtime, date)."""\n\n'
            f"EXCLUDED: dict[str, str] = {gate.EXCLUDED!r}\n"
            '"""Task id -> why it is never served."""\n')
    Path(gate.__file__).write_text(head + body)
    print(f"gate: {len(passed)}/{len(source.task_ids())} verified", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--runtime", default="runsc",
                        help="Docker runtime of the boxes ('' for Docker's default)")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--budget", type=float, default=700.0,
                        help="a task grading slower than this is not verified")
    parser.add_argument("--out", help="append JSON lines here")
    parser.add_argument("--conformance", action="store_true",
                        help="run the declared attack cases on the goldens instead")
    parser.add_argument("--write-gate", action="store_true")
    parser.add_argument("--from", dest="source", help="gate from an earlier --out file")
    args = parser.parse_args()
    if args.source:
        rows = [json.loads(line) for line in Path(args.source).read_text().splitlines() if line]
    else:
        rows = asyncio.run(run_all(args))
    graded = [r for r in rows if "grading_s" in r]
    summary = {
        "cases": len(rows), "as_expected": sum(bool(r.get("ok")) for r in rows),
        "errors": sum(bool(r.get("error")) for r in rows),
        "over_budget": sorted(r["task_id"] for r in graded if not r["within_budget"]),
        "slowest_grading_s": max((r["grading_s"] for r in graded), default=None),
        "failed": sorted(r["task_id"] for r in rows if not r.get("ok")),
    }
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    if args.write_gate:
        if args.conformance:
            raise SystemExit("--write-gate takes reference runs, not --conformance")
        write_gate(rows, args)
    return 0 if summary["as_expected"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
