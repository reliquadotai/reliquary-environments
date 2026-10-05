"""The box phase of the `tmax` split (docs/tmax.md, section 9). It needs a
Docker host. It has NOT been run yet.

Two steps:

1. Build the shared base image from `reliquary_terminal/tmax_base/`, push it,
   and print its digest:

       uv run python scripts/tmax_validate.py build --repository REGISTRY/reliquary-tmax-base

2. Validate the pending tasks against that digest. This writes one JSON file
   per task into OUT and skips tasks that already have one, so it can be
   stopped and resumed:

       uv run python scripts/tmax_validate.py run --image REGISTRY/reliquary-tmax-base@sha256:... \\
           --out validation/ [--limit N] [--tasks ID ...] [--workers 6] [--source tasks.zip]

Then merge the results into the manifest:

       uv run python scripts/tmax_manifest.py --validation validation/

Per task, every container goes through verifiers' own runtime: the same
`TerminalTask.setup`, artifact collection and `TerminalEnv.finalize` that a
training rollout runs. All boxes have no network, 2 CPU and 4 GB.

- probe: setup in box A. Run the initial-state test, take file snapshots
  before and after replaying the reference, then run the final test under an
  audit hook. This derives the protected and hidden inputs and the files the
  reference changed. Setup runs again in a fresh box B to check that the
  protected inputs come out identical.
- no-op: an untouched agent box, graded in a separate box, must score 0.
- solution x3: the reference in a fresh agent box (hidden inputs deleted),
  graded in a separate box, must score 1 each time. If it scores 0 with the
  inputs hidden, it is retried with them visible but still protected
  (`tmax_validation.verdict`). If every attempt fails, the next successful
  recorded run is tried, up to 3.
- mutation: the first solution's collected artifacts, with every changed file
  emptied ("zero") or perturbed ("perturb"), must score 0.

A task's whole validation is about 12 short-lived containers. With
`--workers 6`, expect around 10 s of wall time per task, which is unmeasured.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.runtimes import ProgramResult, provision_runtime
from verifiers.v1.tasksets.harbor.env import HarborEnvConfig

from reliquary_terminal import TerminalEnv, taskset, tmax, tmax_select
from reliquary_terminal import tmax_validation as V
from reliquary_terminal.grading import TerminalTask

PACKAGE = Path(__file__).resolve().parent.parent
SOLVE_TIMEOUT = 900.0
TEST_TIMEOUT = 600.0
MAX_RUNS = 3
STABILITY = 3


def sh(*argv: str) -> str:
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout


# --------------------------------------------------------------------------
# Step 1: the base image.
# --------------------------------------------------------------------------


def build(repository: str, tag: str, push: bool) -> None:
    """Build (and push) the base image. Without a push the image is pinned by
    its local content id (`sha256:...`), which `docker run` accepts on the
    host that built it."""
    ref = f"{repository}:{tag}"
    started = time.monotonic()
    subprocess.run(["docker", "build", "-t", ref, str(PACKAGE / "reliquary_terminal" / "tmax_base")], check=True)
    build_seconds = round(time.monotonic() - started, 1)
    if push:
        subprocess.run(["docker", "push", ref], check=True)
        digest = sh("docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", ref).strip()
    else:
        digest = sh("docker", "image", "inspect", "--format", "{{.Id}}", ref).strip()
    record = PACKAGE / "reliquary_terminal" / "tmax_base" / "built"
    record.mkdir(exist_ok=True)
    for name in ("pip-freeze.txt", "dpkg.txt"):
        (record / name).write_text(sh("docker", "run", "--rm", "--network", "none", digest, "cat", f"/opt/reliquary-tmax/{name}"))
    size = sh("docker", "image", "inspect", "--format", "{{.Size}}", ref).strip()
    (record / "image.json").write_text(
        json.dumps({"image": digest, "tag": ref, "bytes": int(size), "build_seconds": build_seconds}, indent=1) + "\n"
    )
    print(digest)


# --------------------------------------------------------------------------
# Step 2: per task.
# --------------------------------------------------------------------------


def box_config(task: TerminalTask) -> vf.DockerConfig:
    return vf.DockerConfig(
        image=task.data.image,
        workdir=task.data.workdir,
        allow=[],
        cpu=task.data.resources.cpu,
        memory=task.data.resources.memory,
    )


async def run(box, script: str, timeout: float) -> ProgramResult:
    return await asyncio.wait_for(box.run(["bash", "-c", script], {}), timeout)


async def snapshot(box) -> V.Snapshot:
    result = await run(box, V.SNAPSHOT_COMMAND + " 2>/dev/null; true", 300)
    return V.parse_snapshot(result.stdout.encode("utf-8", errors="surrogateescape"))


async def hashes(box, paths: list[str]) -> dict[str, str]:
    if not paths:
        return {}
    script = "sha256sum -- " + " ".join(shlex.quote(p) for p in paths) + " 2>/dev/null; true"
    out = {}
    for line in (await run(box, script, 120)).stdout.splitlines():
        digest, _, path = line.partition("  ")
        out[path] = digest
    return out


async def replay(box, task_dir: Path) -> ProgramResult:
    await box.write("/tmp/reliquary-tmax-solve.sh", (task_dir / "solution" / "solve.sh").read_bytes())
    try:
        return await run(box, "cd /home/user && bash /tmp/reliquary-tmax-solve.sh </dev/null; rm -f /tmp/reliquary-tmax-solve.sh", SOLVE_TIMEOUT)
    except TimeoutError:
        return ProgramResult(124, "", "reference replay timed out")


def make_task(source, task_id, entry, config, image, root) -> TerminalTask:
    return TerminalTask(taskset.tmax_data(source, task_id, entry, 0, config, image, root=root))


async def probe(task: TerminalTask, base: V.Snapshot, details: dict, checks: V.Checks) -> dict | None:
    """Box A: setup, initial test, snapshots around the reference, audited
    final test. Box B: a second setup for determinism."""
    task_dir = Path(task.data.task_dir)
    async with provision_runtime(box_config(task), env=task.runtime_env()) as box:
        await box.prepare_setup()
        started = time.monotonic()
        try:
            await task.setup(box)
        except Exception as e:  # noqa: BLE001 - recorded as the task's verdict
            checks.setup_ok = False
            details["setup_error"] = str(e)[-1500:]
            return None
        checks.setup_ok = True
        details["setup_seconds"] = round(time.monotonic() - started, 2)
        # As a rollout runs it: the agent's phase starts after this.
        await box.prepare_execution([])
        # Before anything else runs, so the snapshot holds only what setup made.
        after_setup = await snapshot(box)
        initial = task_dir / "checks" / tmax.INITIAL_TEST
        await run(box, "mkdir -p /tmp/reliquary-tmax-checks /tmp/reliquary-tmax-probe", 60)
        if initial.is_file():
            await box.write(f"/tmp/reliquary-tmax-checks/{tmax.INITIAL_TEST}", initial.read_bytes())
            result = await run(box, f"cd /tmp/reliquary-tmax-checks && PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider {tmax.INITIAL_TEST}", TEST_TIMEOUT)
            checks.initial_ok = result.exit_code == 0
            if not checks.initial_ok:
                details["initial_tail"] = (result.stdout + result.stderr)[-1500:]
                return None
        solved = await replay(box, task_dir)
        details["replay_exit"] = solved.exit_code
        after_solution = await snapshot(box)
        await box.write("/tmp/reliquary-tmax-probe/reliquary_tmax_audit.py", V.AUDIT_PLUGIN.encode())
        await box.write(f"/tmp/reliquary-tmax-probe/{tmax.FINAL_TEST}", (task_dir / "tests" / tmax.FINAL_TEST).read_bytes())
        await run(
            box,
            "cd /tmp/reliquary-tmax-probe && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/reliquary-tmax-probe python3 -m pytest -q "
            f"-p reliquary_tmax_audit -p no:cacheprovider {tmax.FINAL_TEST} >/dev/null 2>&1; true",
            TEST_TIMEOUT,
        )
        try:
            audited = set(json.loads(await box.read(V.AUDIT_LOG)))
        except Exception:  # noqa: BLE001 - a test that crashed pytest: literals only
            audited = set()
        candidates = audited | V.literal_paths((task_dir / "tests" / tmax.FINAL_TEST).read_text())
        protected = V.protected_inputs(candidates, base, after_setup, after_solution)
        changed = V.changed_paths(after_setup, after_solution)
        first = await hashes(box, protected)
    async with provision_runtime(box_config(task), env=task.runtime_env()) as box:
        await box.prepare_setup()
        await task.setup(box)
        second = await hashes(box, protected)
    checks.deterministic = first == second
    if not checks.deterministic:
        details["nondeterministic"] = sorted(p for p in protected if first.get(p) != second.get(p))
        return None
    details["changed_files"] = len(changed)
    return {"protected": protected, "changed": changed}


async def episode(env: TerminalEnv, task: TerminalTask, solve: bool, artifacts: dict | None = None) -> tuple[float, dict | None, dict]:
    """One rollout without a model: setup, the reference (or nothing), then
    collection and grading in a separate box, as `HarborEnv` runs it.
    With `artifacts`, skip the agent box and grade those instead."""
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
    )
    info: dict = {}
    if artifacts is None:
        async with provision_runtime(box_config(task), env=task.runtime_env()) as box:
            await box.prepare_setup()
            await task.setup(box)
            await box.prepare_execution([])
            if solve:
                info["replay_exit"] = (await replay(box, Path(task.data.task_dir))).exit_code
            await task.graded_elsewhere().finalize(trace, box)
    else:
        trace.state.artifacts = artifacts
    trace.ok = True
    run_episode = vf.Episode(task=trace.task, traces=[trace])
    await env.finalize(task, run_episode)
    info["grading"] = trace.info.get("grading")
    return float(trace.reward or 0.0), trace.state.artifacts, info


async def validate(task_id: str, args, source, config, env: TerminalEnv, base: V.Snapshot) -> dict:
    checks = V.Checks()
    details: dict = {}
    runs = tmax.successful_runs(source, task_id)
    root = args.out / ".tasks"
    chosen, protected, artifacts = 0, [], None
    for run_index in range(min(MAX_RUNS, len(runs))):
        checks = V.Checks()
        details = {"run": run_index}
        entry = {"run": run_index, "protected": [], "hidden": []}
        probing = make_task(source, task_id, entry, config, args.image, root)
        found = await probe(probing, base, details, checks)
        if found is None:
            break  # setup, initial state or determinism: no other run helps
        protected = found["protected"]
        hidden = V.hidden_inputs(protected, probing.data.prompt)
        checks.hidden = hidden
        task = make_task(source, task_id, {"run": run_index, "protected": protected, "hidden": hidden}, config, args.image, root)
        checks.noop_rewards = [(await episode(env, task, solve=False))[0]]
        if checks.noop_rewards[0] > 0:
            break
        rewards, first_artifacts = [], None
        for attempt in range(STABILITY):
            try:
                reward, collected, info = await episode(env, task, solve=True)
            except RuntimeError as e:
                if "byte limit" in str(e):
                    checks.artifact_over_cap = True
                    break
                raise
            rewards.append(reward)
            details.setdefault("solution_grading", info.get("grading"))
            if attempt == 0:
                first_artifacts = collected
            if reward == 0:
                break  # one failure decides it; the retry below may still apply
        checks.solution_rewards = rewards
        if checks.artifact_over_cap:
            break
        if hidden and rewards and all(r == 0 for r in rewards):
            visible = make_task(source, task_id, {"run": run_index, "protected": protected, "hidden": []}, config, args.image, root)
            retry = []
            for attempt in range(STABILITY):
                reward, collected, _ = await episode(env, visible, solve=True)
                retry.append(reward)
                if attempt == 0:
                    first_artifacts = collected
                if reward == 0:
                    break
            checks.hidden_retry_rewards = retry
            if retry and all(r == 1 for r in retry):
                task = visible
        reasons, _ = V.verdict(checks)
        if reasons in (["solution_fails"],) and run_index + 1 < min(MAX_RUNS, len(runs)):
            continue
        if not reasons:
            for mode in ("zero", "perturb"):
                mutated, count = V.mutate_artifacts(first_artifacts, found["changed"], mode)
                details[f"mutated_{mode}"] = count
                if count:
                    checks.mutation_rewards[mode] = (await episode(env, task, solve=False, artifacts=mutated))[0]
            if not checks.mutation_rewards:
                details["mutation"] = "the reference changed nothing under /app or /home/user"
        chosen = run_index
        artifacts = first_artifacts
        break
    sizes = {root_: len(a or b"") for root_, a in (artifacts or {}).items()}
    details["artifact_bytes"] = sizes
    return V.result_record(task_id, args.image, chosen, protected, checks, details)


async def run_all(args) -> None:
    source = tmax.Source(args.source or tmax.download_source())
    manifest = tmax_select.load_manifest()
    pending = [tid for tid, e in manifest["tasks"].items() if e["status"] == "pending"]
    if args.tasks:
        pending = [t for t in pending if t in set(args.tasks)]
    pending.sort(key=tmax_select.order_key)
    if args.limit:
        pending = pending[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in pending if not (args.out / f"{t}.json").exists()]
    print(f"{len(pending)} pending selected, {len(todo)} to validate", flush=True)
    config = vf.taskset_config_type("reliquary-terminal")(
        id="reliquary-terminal", split="tmax", resource_multiplier=args.resource_multiplier
    )
    env = TerminalEnv(HarborEnvConfig(taskset=config, verifier_runtime=vf.DockerConfig(), verifier_retries=0))
    # The base image's own files, to tell them from what setup made.
    cpu, memory = taskset.TMAX_CPUS * args.resource_multiplier, taskset.TMAX_MEMORY_GB * args.resource_multiplier
    async with provision_runtime(vf.DockerConfig(image=args.image, workdir="/", allow=[], cpu=cpu, memory=memory)) as box:
        await box.prepare_setup()
        base = await snapshot(box)
    gate = asyncio.Semaphore(args.workers)
    done = 0

    async def one(task_id: str) -> None:
        nonlocal done
        async with gate:
            if (args.out / "STOP").exists():
                return
            started = time.monotonic()
            try:
                record = await validate(task_id, args, source, config, env, base)
            except Exception:  # noqa: BLE001 - an infrastructure failure is not a verdict
                (args.out / f"{task_id}.error").write_text(traceback.format_exc())
                return
            record["details"]["wall_seconds"] = round(time.monotonic() - started, 1)
            path = args.out / f"{task_id}.json"
            path.with_suffix(".partial").write_text(json.dumps(record, indent=1))
            path.with_suffix(".partial").rename(path)
            done += 1
            print(time.strftime("%H:%M:%S"), task_id, record["reasons"] or "kept", f"{done}/{len(todo)}", flush=True)

    await asyncio.gather(*(one(t) for t in todo))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--repository", required=True)
    b.add_argument("--no-push", action="store_true", help="pin by local image id; for a single-host pilot")
    b.add_argument("--tag", default=hashlib.sha256((PACKAGE / "reliquary_terminal" / "tmax_base" / "Dockerfile").read_bytes() + (PACKAGE / "reliquary_terminal" / "tmax_base" / "apt.txt").read_bytes() + (PACKAGE / "reliquary_terminal" / "tmax_base" / "requirements.txt").read_bytes()).hexdigest()[:12])
    r = sub.add_parser("run")
    r.add_argument("--image", required=True, help="the base image, pinned by digest")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--source", type=Path)
    r.add_argument("--tasks", nargs="*")
    r.add_argument("--limit", type=int)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument(
        "--resource-multiplier", type=float, default=1.0,
        help="scale the 2 CPU / 4 GB boxes (0.5 on a small host)",
    )
    args = parser.parse_args()
    if os.environ.get("RELIQUARY_TMAX_I_HAVE_A_BOX") != "1":
        sys.exit(
            "tmax_validate.py starts many containers. Run it on a dedicated container host, "
            "never on a production machine, with RELIQUARY_TMAX_I_HAVE_A_BOX=1."
        )
    if args.command == "build":
        build(args.repository, args.tag, push=not args.no_push)
    else:
        if "@sha256:" not in args.image and not args.image.startswith("sha256:"):
            sys.exit("--image must be pinned by digest (REPOSITORY@sha256:... or a local sha256:... id)")
        asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
