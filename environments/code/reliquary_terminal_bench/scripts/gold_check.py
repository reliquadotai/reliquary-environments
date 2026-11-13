"""Grade every task's reference solution offline through this package's own hooks: the gate.

Run on a host shaped like a signed-episode gateway (Docker with gVisor's `runsc`, the
verifier images present locally, `scripts/build_verifier_images.py`), never on a
production gateway. Per task, as the sandbox bridge runs an episode:

1. agent box: a container of the verifier image, no network, the task's CPUs and memory;
   `Task.setup`; the reference (`conformance.solve_calls`: `solution/` uploaded, `solve.sh`
   run from the WORKDIR) through `bash`, bounded by the task's agent timeout; every
   process stopped (`kill -9 -1`, as the gateway does); `Task.finalize` (the capture);
2. grading box: a fresh container of the same image; `Task.grading_setup`; the captured
   `/logs/artifacts` files written back; `Task.solved`, timed.

`--egress-network NET --egress-env JSON` puts the agent's box (never the grading box) on
the gateway's egress network with the profile's proxy settings, as reliquary-sandbox does
for an env given `{"network": {"profile": ..., "agent": true, "grading": false}}` (host:
`scripts/egress-host-setup.sh`, plus `gate.EGRESS_EXTRA_HOSTS`): the `eval` split's gate.
`--noop` adds a no-op episode per task (expected 0). `--conformance` plays the goldens'
declared cases (`conformance.conformance_cases`) instead. `--online` gives the agent's box
network (diagnostic: a reference that passes online but not offline needs network to
solve). Output: one JSON line per episode, then a summary. `--write-gate` records the
tasks whose reference graded 1 offline within `--budget` and whose no-op graded 0 in
`reliquary_terminal_bench/gate.py`; a new gate is a new package version.

    uv run python scripts/gold_check.py --images local --jobs 6 --noop --out offline.jsonl
    uv run python scripts/gold_check.py --write-gate eval_offline --from offline.jsonl
    uv run python scripts/gold_check.py --jobs 6 --noop --egress-network rq-egress \
        --egress-env "$(cat egress-env.json)" --out egress.jsonl
    uv run python scripts/gold_check.py --write-gate eval --from egress.jsonl
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

from reliquary_terminal_bench import conformance, gate, images, source
from reliquary_terminal_bench import taskset as taskset_module
from reliquary_terminal_bench.task import ARTIFACTS, PATHS, SANDBOX_RUNTIME, STATE, TerminalBenchTask
from reliquary_terminal_bench.taskset import TerminalBenchConfig, TerminalBenchTaskset

NOOP_CALLS = [["bash", {"command": "true"}]]
STEP_S = 810.0
"""A gateway's grading step (the bridge's MAX_GRADING_TIMEOUT_S)."""


class Box:
    """A verifiers-shaped runtime over `docker exec` into one container (as root)."""

    def __init__(self, docker: str, container: str, workdir: str):
        self.docker = docker
        self.container = container
        self.workdir = workdir
        self.config = SimpleNamespace(type=SANDBOX_RUNTIME, workdir=workdir)

    async def _exec(self, argv, env=None, stdin: bytes | None = None, workdir: str = "/",
                    timeout: float | None = None):
        flags = ["-i"] if stdin is not None else []
        for key, value in (env or {}).items():
            flags += ["-e", f"{key}={value}"]
        process = await asyncio.create_subprocess_exec(
            self.docker, "exec", *flags, "-w", workdir, self.container, *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(process.communicate(stdin), timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
        return process.returncode, out, err

    async def run(self, argv, env, timeout: float | None = None):
        code, out, err = await self._exec(list(argv), env, workdir=self.workdir, timeout=timeout)
        return SimpleNamespace(exit_code=code, stdout=out.decode("utf-8", "replace"),
                               stderr=err.decode("utf-8", "replace"))

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        code, out, _ = await self._exec(
            ["sh", "-c", 'if [ -f "$1" ] && [ ! -L "$1" ]; then wc -c < "$1"; '
                         'elif [ -e "$1" ] || [ -L "$1" ]; then echo special; fi', "read", path])
        size = out.decode().strip()
        if code or not size:
            raise FileNotFoundError(path)
        if not size.isdigit() or (max_bytes is not None and int(size) > max_bytes):
            raise OSError(f"{path}: not a regular file, or over {max_bytes} bytes")
        code, out, err = await self._exec(["cat", "--", path])
        if code:
            raise OSError(f"{path}: {err.decode()[-200:]}")
        return out

    async def write(self, path: str, data: bytes) -> None:
        code, _, err = await self._exec(
            ["sh", "-c", 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"', "write", path],
            stdin=data)
        if code:
            raise OSError(f"write {path}: {err.decode()[-200:]}")


def local_image(args, task: TerminalBenchTask) -> str:
    if args.images == "local":
        return f"{args.registry}/{task.data.image}"
    return images.lock()[task.data.image]


async def start(args, task: TerminalBenchTask, *, online: bool = False,
                egress: bool = False) -> Box:
    name = f"tbg-{task.key[:40]}-{uuid.uuid4().hex[:8]}"
    resources = task.data.resources
    argv = [args.docker, "run", "-d", "--name", name,
            "--cpus", str(resources.cpu), "--memory", f"{int(resources.memory * 1024)}m",
            "--workdir", task.data.workdir, "--entrypoint", "sleep"]
    if egress:
        argv += ["--network", args.egress_network]
        for key, value in json.loads(args.egress_env).items():
            argv += ["-e", f"{key}={value}"]
    elif not online:
        argv += ["--network", "none"]
    if args.runtime:
        argv += ["--runtime", args.runtime]
    process = await asyncio.create_subprocess_exec(
        *argv, local_image(args, task), "infinity",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await process.communicate()
    if process.returncode:
        raise RuntimeError(f"docker run: {err.decode()[-300:]}")
    return Box(args.docker, out.decode().strip(), task.data.workdir)


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


async def play(args, task: TerminalBenchTask, calls: list, *, online: bool) -> dict:
    """One episode: the calls in the agent's box, then grading in a fresh one."""
    egress = bool(args.egress_network)
    row: dict = {"task_id": task.key, "online": online, "egress": egress}
    agent = grading = None
    clock = time.monotonic()
    try:
        agent = await start(args, task, online=online, egress=egress)
        await asyncio.wait_for(task.setup(agent), task.data.timeout.setup)
        row["setup_s"] = round(time.monotonic() - clock, 1)
        clock = time.monotonic()
        budget = task.data.timeout.agent
        for _, arguments in calls:
            left = budget - (time.monotonic() - clock)
            if left <= 0:
                raise TimeoutError
            result = await agent.run(["bash", "-c", arguments["command"]], {}, timeout=left)
            row["solve_exit"] = result.exit_code
            row["solve_tail"] = (result.stdout + result.stderr)[-600:]
        row["solve_s"] = round(time.monotonic() - clock, 1)
        await agent.run(["sh", "-c", "kill -9 -1 2>/dev/null; sleep 1; true"], {})
        clock = time.monotonic()
        trace = new_trace()
        await asyncio.wait_for(task.finalize(trace, agent), STEP_S)
        row["finalize_s"] = round(time.monotonic() - clock, 1)
        if trace.info.get("state_error"):
            row["state_error"] = trace.info["state_error"]
        artifacts = {}
        for path in (STATE, PATHS):
            try:
                artifacts[path] = await agent.read(path)
            except FileNotFoundError:
                pass
        row["state_bytes"] = len(artifacts.get(STATE, b""))
        row["paths_bytes"] = len(artifacts.get(PATHS, b""))
        await stop(agent)
        agent = None

        clock = time.monotonic()
        grading = await start(args, task)
        row["grading_box_s"] = round(time.monotonic() - clock, 1)
        step = time.monotonic()
        await asyncio.wait_for(task.grading_setup(grading), STEP_S)
        for path, data in artifacts.items():
            await grading.write(path, data)
        trace = new_trace()
        reward = await asyncio.wait_for(task.solved(grading, trace),
                                        max(1.0, STEP_S - (time.monotonic() - step)))
        row["grading_s"] = round(time.monotonic() - step, 1)
        row["reward"] = reward
        row["metrics"] = trace.metrics
        detail = trace.info.get("terminal_bench") or {}
        row["tests"] = detail.get("tests")
        if reward != 1:
            row["detail"] = detail
    except TimeoutError:
        row["error"] = "timeout: past the agent timeout or the step budget"
        row.setdefault("reward", 0.0)
    except Exception as exc:  # noqa: BLE001 - recorded per task, the run goes on
        row["error"] = f"{type(exc).__name__}: {str(exc)[-500:]}"
        row.setdefault("reward", 0.0)
    finally:
        await stop(agent)
        await stop(grading)
    row["within_budget"] = row.get("grading_s", STEP_S) <= args.budget
    return row


def episodes(args) -> list[tuple[TerminalBenchTask, str, list, float, bool]]:
    tasks = TerminalBenchTaskset(TerminalBenchConfig(split="eval_all",
                                                     id="reliquary-terminal-bench",
                                                     tasks=args.tasks or None))
    if args.conformance:
        by_key = {task.key: task for task in tasks.load()}
        ids = source.task_ids()
        return [(by_key[ids[c["index"]]], c["name"], c["calls"], c["expect"], False)
                for c in conformance.conformance_cases("eval_all") if ids[c["index"]] in by_key]
    played = []
    for task in tasks.load():
        if task.key in gate.EXCLUDED and not args.include_excluded:
            continue
        if not args.skip_reference:
            played.append((task, "reference", conformance.solve_calls(task.key), 1.0,
                           args.online))
        if args.noop:
            played.append((task, "noop", NOOP_CALLS, 0.0, False))
    return played


async def run_all(args) -> list[dict]:
    limit = asyncio.Semaphore(args.jobs)
    out = open(args.out, "a") if args.out else None

    async def one(task, name, calls, expect, online):
        async with limit:
            row = await play(args, task, calls, online=online)
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
    split = args.write_gate
    offline = [r for r in rows if not r.get("online")
               and bool(r.get("egress")) == (split == "eval")]
    noop_bad = {r["task_id"] for r in offline if r.get("case") == "noop"
                and (r.get("reward") != 0 or r.get("error"))}
    noop_ok = {r["task_id"] for r in offline if r.get("case") == "noop"} - noop_bad
    passed = sorted(r["task_id"] for r in offline
                    if r.get("case") == "reference" and r.get("reward") == 1
                    and r.get("grading_s", STEP_S) <= args.budget and not r.get("error")
                    and r["task_id"] in noop_ok and r["task_id"] not in gate.EXCLUDED)
    digest = hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in rows).encode())
    provenance = {"host": args.host or platform.node(), "runtime": args.runtime or "docker default",
                  "date": datetime.date.today().isoformat(), "budget_s": str(args.budget),
                  "results_sha256": digest.hexdigest(),
                  "verified": f"{len(passed)}/{len(source.task_ids())}"}
    name = taskset_module.GATES[split]
    check = "GOLD_CHECK" if split == "eval" else "GOLD_CHECK_OFFLINE"
    if split == "eval":
        provenance["egress"] = f"agent phase, profile {gate.EGRESS_PROFILE} + EGRESS_EXTRA_HOSTS"
    text = Path(gate.__file__).read_text()
    start = text.index(f"{name}: frozenset")
    end = text.index('"""', text.index('"""', start) + 3) + 3
    doc_start = text.index('"""', start)
    doc = text[doc_start:end]
    text = (text[:start] + f"{name}: frozenset[str] | None = frozenset({{\n"
            + "".join(f"    {t!r},\n" for t in passed) + "})\n" + doc + text[end:])
    start = text.index(f"{check}: dict[str, str] = ")
    doc_start = text.index('"""', start)
    end = text.index('"""', doc_start + 3) + 3
    doc = text[doc_start:end]
    text = (text[:start] + f"{check}: dict[str, str] = {{\n"
            + "".join(f"    {k!r}: {v!r},\n" for k, v in provenance.items()) + "}\n"
            + doc + text[end:])
    Path(gate.__file__).write_text(text)
    print(f"gate {split}: {len(passed)}/{len(source.task_ids())} verified", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--runtime", default="runsc",
                        help="Docker runtime of the boxes ('' for Docker's default)")
    parser.add_argument("--images", choices=("local", "lock"), default="local",
                        help="local: the build host's tags under --registry; lock: the lock's digests")
    parser.add_argument("--registry", default="127.0.0.1:5001")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--budget", type=float, default=700.0,
                        help="a task grading slower than this is not verified")
    parser.add_argument("--out", help="append JSON lines here")
    parser.add_argument("--noop", action="store_true", help="also play a no-op episode per task")
    parser.add_argument("--online", action="store_true",
                        help="give the agent's box network (diagnostic, never gates)")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--conformance", action="store_true",
                        help="play the declared conformance cases on the goldens instead")
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument("--egress-network", default="",
                        help="the agent's box joins this Docker network (the gateway's egress)")
    parser.add_argument("--egress-env", default="{}",
                        help="JSON env of the egress profile (proxy variables and date pins)")
    parser.add_argument("--host", default="", help="provenance: where the gold check ran")
    parser.add_argument("--write-gate", choices=("eval", "eval_offline"),
                        help="record the verified tasks of this split in gate.py")
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
        "failed": sorted(f"{r['task_id']}:{r.get('case')}" for r in rows if not r.get("ok")),
    }
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    if args.write_gate:
        if args.conformance:
            raise SystemExit("--write-gate takes reference runs, not --conformance")
        write_gate(rows, args)
    return 0 if summary["as_expected"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
