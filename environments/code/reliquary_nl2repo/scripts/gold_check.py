"""Grade every task's reference and an empty workspace offline through this package's own
hooks: the gate.

Run on a host shaped like a signed-episode gateway (Docker with gVisor's `runsc`, the
images pulled by digest), never on a production gateway, after
`scripts/reference_source.py --out REFS` recovered the references from the images. Per
task, as the sandbox bridge runs an episode:

1. agent box: a container of the pinned image, no network, 4 CPUs, 8 GB, the image's
   user; `Task.setup`; a leak probe (can the box import the reference's top-level
   modules, with `/workspace` empty?); the reference unpacked into `/workspace` (the
   `reference` case) or nothing (the `empty` case); `Task.finalize`;
2. grading box: a fresh container of the same image; `Task.grading_setup`; the archive
   restored; `Task.pass_rate`, timed.

The boxes present themselves as a sandbox's (`config.type`), so the hooks take exactly
the paths they take on a gateway. Output: one JSON line per episode, then a summary.
`--write-gate` records the tasks whose reference graded 1 and whose empty workspace
graded 0, both within `--budget` seconds, with no leak found, in
`reliquary_nl2repo/gate.py`. `--conformance` runs the declared attack cases instead.

    uv run python scripts/gold_check.py --refs /root/refs --jobs 4 --out gold.jsonl
    uv run python scripts/gold_check.py --write-gate --from gold.jsonl
    uv run python scripts/gold_check.py --conformance --out conformance.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import io
import json
import platform
import sys
import tarfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from reliquary_nl2repo import conformance, gate, source, submission
from reliquary_nl2repo.task import ARCHIVE_PATH, SANDBOX_RUNTIME, WORKDIR, NL2RepoTask
from reliquary_nl2repo.taskset import NL2RepoConfig, NL2RepoTaskset

PACKAGE = Path(gate.__file__).resolve().parent
LOCK = json.loads((PACKAGE / "sandbox-images.lock.json").read_text())["images"]
STEP_S = 810.0
"""A gateway's grading step (the bridge's MAX_GRADING_TIMEOUT_S)."""
REFERENCE_TMP = "/tmp/reliquary-reference.tar.gz"
MAX_REFERENCE_FILE = 4 * 1024 * 1024
"""Upstream build contexts hold junk no agent would write (emoji: a 1 GB tar of itself;
justext: a Windows virtualenv): files over this are left out of the reference played."""
PROBE = (
    "import importlib.util, sys\n"
    "for name in sys.argv[1:]:\n"
    "    try:\n"
    "        spec = importlib.util.find_spec(name)\n"
    "    except Exception:\n"
    "        spec = None\n"
    "    if spec is not None and spec.origin not in (None, 'built-in', 'frozen'):\n"
    "        print(name, spec.origin)\n"
)


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
        code, out, err = await self._exec(list(argv), env, workdir="/")
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


async def start(args, task: NL2RepoTask) -> Box:
    image = LOCK[task.data.image]
    name = f"nl2repo-gold-{task.key[:40].lower()}-{uuid.uuid4().hex[:8]}"
    resources = task.data.resources
    argv = [args.docker, "run", "-d", "--name", name, "--network", "none",
            "--cpus", str(resources.cpu), "--memory", f"{int(resources.memory * 1024)}m",
            "--entrypoint", "sleep"]
    if args.runtime:
        argv += ["--runtime", args.runtime]
    process = await asyncio.create_subprocess_exec(
        *argv, image, "infinity", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await process.communicate()
    if process.returncode:
        raise RuntimeError(f"docker run {image}: {err.decode()[-300:]}")
    box = Box(args.docker, out.decode().strip())
    made = await box.run(["sh", "-c", f"mkdir -p {WORKDIR}"], {})
    if made.exit_code:
        raise RuntimeError(f"no {WORKDIR} in the box: {made.stderr[-200:]}")
    return box


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


def trim_reference(reference: bytes) -> tuple[bytes, int]:
    """The reference without toolchain directories and files over MAX_REFERENCE_FILE (what
    no submission would carry), recompressed; and how many files were left out."""
    out = io.BytesIO()
    left_out = 0
    with tarfile.open(fileobj=io.BytesIO(reference), mode="r:gz") as source, \
            tarfile.open(fileobj=out, mode="w:gz") as kept:
        for member in source:
            if not member.isreg():
                continue
            if member.size > MAX_REFERENCE_FILE \
                    or submission.drop_reason(member.name, ()) == "toolchain_dir":
                left_out += 1
                continue
            kept.addfile(member, source.extractfile(member))
    return out.getvalue(), left_out


def module_names(reference: bytes) -> list[str]:
    """The reference's top-level import names (packages and modules at its root or
    under `src/`), for the leak probe."""
    names: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(reference), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = member.name.split("/")
            if parts[0] == "src" and len(parts) > 1:
                parts = parts[1:]
            if len(parts) == 2 and parts[1] == "__init__.py":
                names.add(parts[0])
            elif len(parts) == 1 and parts[0].endswith(".py"):
                stem = parts[0][:-3]
                if stem not in ("setup", "conftest", "noxfile", "tasks", "fabfile") \
                        and not stem.startswith("test"):
                    names.add(stem)
    return sorted(n for n in names if n.isidentifier() and n not in ("tests", "test", "docs"))


async def probe(box: Box, names: list[str]) -> list[str]:
    if not names:
        return []
    result = await box.run(["sh", "-c", 'cd / && py=$(command -v python3 || command -v python) '
                                        '&& exec "$py" -c "$0" "$@"', PROBE, *names], {})
    return [line for line in result.stdout.splitlines() if line.strip()]


async def play(args, task: NL2RepoTask, case: str, calls: list | None,
               reference: bytes | None) -> dict:
    """One episode: the agent's box, then grading in a fresh one."""
    row: dict = {"task_id": task.key, "case": case}
    agent = grading = None
    clock = time.monotonic()
    try:
        agent = await start(args, task)
        await asyncio.wait_for(task.setup(agent), task.data.timeout.setup)
        row["setup_s"] = round(time.monotonic() - clock, 1)
        if reference is not None:
            reference, row["reference_left_out"] = trim_reference(reference)
            row["leaks"] = await probe(agent, module_names(reference))
            await agent.write(REFERENCE_TMP, reference)
            unpacked = await agent.run(["sh", "-c", f"tar -xzf {REFERENCE_TMP} -C {WORKDIR} "
                                                    f"--no-same-owner && rm -f {REFERENCE_TMP}"], {})
            if unpacked.exit_code:
                raise RuntimeError(f"unpacking the reference: {unpacked.stderr[-300:]}")
        for tool, arguments in calls or []:
            result = await agent.run(["bash", "-c", arguments["command"]], {})
            if result.exit_code:
                raise RuntimeError(f"{tool} call failed ({result.exit_code}): "
                                   f"{(result.stderr or result.stdout)[-300:]}")
        clock = time.monotonic()
        trace = new_trace()
        await asyncio.wait_for(task.finalize(trace, agent), STEP_S)
        row["finalize_s"] = round(time.monotonic() - clock, 1)
        try:
            archive = await agent.read(ARCHIVE_PATH)
        except FileNotFoundError:
            archive = None
        row["archive_bytes"] = None if archive is None else len(archive)
        if trace.info.get("capture_error"):
            row["capture_error"] = trace.info["capture_error"]
        await stop(agent)
        agent = None

        grading = await start(args, task)
        step = time.monotonic()
        await asyncio.wait_for(task.grading_setup(grading), STEP_S)
        if archive is not None:
            await grading.write(ARCHIVE_PATH, archive)
        trace = new_trace()
        reward = await asyncio.wait_for(task.pass_rate(grading, trace),
                                        max(1.0, STEP_S - (time.monotonic() - step)))
        row["grading_s"] = round(time.monotonic() - step, 1)
        row["reward"] = reward
        row["metrics"] = trace.metrics
        info = trace.info.get("nl2repo") or {}
        row["exits"] = info.get("exits")
        row["dropped"] = info.get("dropped")
        if reward != (1.0 if case == "reference" else 0.0):
            row["detail"] = {"report_errors": info.get("report_errors"),
                             "output_tail": (info.get("output_tail") or "")[-2500:]}
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


def episodes(args) -> list[tuple]:
    tasks = NL2RepoTaskset(NL2RepoConfig(split="eval_all", id="reliquary-nl2repo",
                                         tasks=args.tasks or None))
    by_index = {task.data.idx: task for task in tasks.load()}
    if args.conformance:
        return [(by_index[c["index"]], c["name"], c["calls"], None, c["expect"])
                for c in conformance.conformance_cases("eval_all") if c["index"] in by_index]
    out = []
    for _, task in sorted(by_index.items()):
        path = Path(args.refs) / f"{task.key}.tar.gz" if args.refs else None
        if "reference" in args.cases:
            if path is not None and path.is_file():
                out.append((task, "reference", None, path.read_bytes(), 1.0))
            else:
                print(json.dumps({"task_id": task.key, "case": "reference",
                                  "error": "no reference recovered"}), flush=True)
        if "empty" in args.cases:
            out.append((task, "empty", None, None, 0.0))
    return out


async def run_all(args) -> list[dict]:
    limit = asyncio.Semaphore(args.jobs)
    out = open(args.out, "a") if args.out else None

    async def one(task, case, calls, reference, expect):
        async with limit:
            row = await play(args, task, case, calls, reference)
            row.update(expect=expect, ok=row.get("reward") == expect)
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


def verified(rows: list[dict], budget: float, min_reference: float = 1.0,
             max_empty: float = 0.0) -> list[str]:
    """Tasks whose reference graded at least `min_reference` (1) and empty workspace at
    most `max_empty` (0), both within the budget, with no error and no reference module
    importable in the emptied agent box."""
    by_task: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], {})[row.get("case")] = row
    good = []
    for task_id, cases in by_task.items():
        ref, empty = cases.get("reference"), cases.get("empty")
        if not ref or not empty or ref.get("error") or empty.get("error"):
            continue
        if ref.get("reward", 0) < min_reference or empty.get("reward", 1) > max_empty \
                or ref.get("leaks"):
            continue
        if max(ref.get("grading_s", STEP_S), empty.get("grading_s", STEP_S)) > budget:
            continue
        good.append(task_id)
    return sorted(good)


def write_gate(rows: list[dict], args) -> None:
    passed = verified(rows, args.budget, args.min_reference, args.max_empty)
    digest = hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in rows).encode())
    provenance = {"host": platform.node(), "runtime": args.runtime or "docker default",
                  "date": datetime.date.today().isoformat(), "budget_s": str(args.budget),
                  "rule": f"reference >= {args.min_reference}, empty <= {args.max_empty}",
                  "results_sha256": digest.hexdigest(),
                  "verified": f"{len(passed)}/{len(source.task_ids())}"}
    text = Path(gate.__file__).read_text()
    head = text.split("VERIFIED:", 1)[0]
    body = ("VERIFIED: frozenset[str] | None = frozenset({\n"
            + "".join(f"    {t!r},\n" for t in passed) + "})\n"
            '"""Task ids whose reference scored 1 and empty workspace 0 offline within the budget."""\n\n'
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
    parser.add_argument("--cases", nargs="*", default=["reference", "empty"])
    parser.add_argument("--refs", help="directory of <task>.tar.gz references")
    parser.add_argument("--budget", type=float, default=700.0,
                        help="a task grading slower than this is not verified")
    parser.add_argument("--min-reference", type=float, default=1.0,
                        help="the reference must grade at least this")
    parser.add_argument("--max-empty", type=float, default=0.0,
                        help="the empty workspace must grade at most this")
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
        "episodes": len(rows), "as_expected": sum(bool(r.get("ok")) for r in rows),
        "errors": sum(bool(r.get("error")) for r in rows),
        "over_budget": sorted({r["task_id"] for r in graded if not r["within_budget"]}),
        "slowest_grading_s": max((r["grading_s"] for r in graded), default=None),
        "leaks": sorted({r["task_id"] for r in rows if r.get("leaks")}),
        "verified": len(verified(rows, args.budget, args.min_reference, args.max_empty)),
    }
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    if args.write_gate:
        if args.conformance:
            raise SystemExit("--write-gate takes reference runs, not --conformance")
        write_gate(rows, args)
    return 0 if summary["as_expected"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
