"""One Terminal-Bench 2.1 task under the env norm (docs/env-norm.md), graded offline.

Harbor grades a task in the box the agent worked in, with network: `tests/test.sh`
installs curl, uv and pytest from the internet, then runs the final-state tests. A
signed-episode sandbox has no network and grades in a pristine box that receives only
declared artifacts. This task bridges the two:

* The image is the task's upstream image plus inert files, `/opt/tb-verifier/
  overlay.tar.gz.<part>`: every file `test.sh`'s own installs add or change, recorded once by
  running `test.sh` with network on the untouched image (`scripts/build_verifier_images.py`).
* `setup` (the agent's box) deletes `/opt/tb-verifier`, so the agent works in the
  upstream image as published, and marks the time its work starts.
* `finalize` (the agent's box, its processes already stopped) captures the agent's
  filesystem: every file, link and directory whose inode changed since the mark
  (`find -cnewer`, ctime: a restored mtime cannot hide a change), as a tar with owners and
  modes, plus the list of every path in the box (so deletions travel too), both under
  `/logs/artifacts` (the only artifact root).
* `grading_setup` (a pristine box of the same image) lists its paths.
* `solved` (that box, the artifacts restored): unpacks the verifier overlay (what
  `test.sh`'s installs would have left), deletes the paths the agent deleted, unpacks the
  agent's files over `/`, stages `tests/` at `/tests` and runs `test.sh` from the image's
  WORKDIR with `UV_OFFLINE=1`: its network commands fail and change nothing, its tools
  are already there. The reward is `reward.txt` (or `reward.json`), as Harbor reads it.

What this cannot carry is a process: a task graded against a server or VM the agent left
running is excluded (`gate.EXCLUDED`). Trust is Harbor's: the agent's files are restored
as root over the whole filesystem, as in Harbor's same-box grading, so a reward is as
tamper-proof as upstream's (not more); the forced-draw audit is what binds the actions to
the model.
"""

from __future__ import annotations

import json
import time

import verifiers.v1 as vf
from verifiers.v1.runtimes import Runtime
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask
from verifiers.v1.trace import Trace

from reliquary_terminal_bench import images, source

SANDBOX_RUNTIME = "reliquary-sandbox"
"""`runtime.config.type` of a signed-episode sandbox's boxes."""
VERIFIER_DIR = images.VERIFIER_DIR
OVERLAY = f"{VERIFIER_DIR}/overlay.tar.gz"
"""Stored as `OVERLAY.<part>` files (one registry layer each), concatenated to unpack."""
ARTIFACTS = "/logs/artifacts"
STATE = f"{ARTIFACTS}/tb-state.tar.gz"
PATHS = f"{ARTIFACTS}/tb-paths.gz"
MARK_DIR = "/var/lib/reliquary-tb"
MARKER = f"{MARK_DIR}/work-start"
GRADE_DIR = "/var/lib/reliquary-tb-grade"
TESTS_ARCHIVE = "/tmp/reliquary-tb-tests.tgz"
TEST_STDOUT = "/logs/verifier/test-stdout.txt"
MAX_STATE_BYTES = 512 * 1024 * 1024
"""The captured files, compressed. A gateway serving this env sets its bridged env's
`defaults.max_artifact_bytes` to `MAX_ARTIFACT_BYTES` (the bridge's default, 32 MiB, is
below what some references leave; with network, references that install toolchains leave
150-480 MB)."""
MAX_ARTIFACT_BYTES = MAX_STATE_BYTES + 32 * 1024 * 1024
"""The state plus the path list (a few MB compressed): the gateway's artifact cap."""
MAX_REWARD_BYTES = 4096

GRADING_ENV = {
    # test.sh's installs find everything in the overlay: fail at once, never wait.
    "UV_OFFLINE": "1",
    "UV_NO_PROGRESS": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "DEBIAN_FRONTEND": "noninteractive",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/root",
}

_PRUNED = ("./proc", "./sys", "./dev", "./run", "./logs", "./tests", "." + VERIFIER_DIR,
           "." + MARK_DIR, "." + GRADE_DIR)
_WALK = (
    "walk() { cd / && find . -mindepth 1 \\( "
    + " -o ".join(f"-path {p}" for p in _PRUNED)
    + ' -o -path "./${work#/}" \\) -prune -o \\( -type f -o -type l -o -type d \\) "$@" -print0; }\n'
)
"""Every regular file, link and directory below `/` (relative, `./...`), less the pseudo
filesystems, Harbor's `/logs` and `/tests`, the verifier overlay and our own marks.
Sockets, FIFOs and devices never travel."""

SETUP_SCRIPT = f"""set -u
[ -f {OVERLAY}.aa ] || {{ echo "not this task's verifier image: no {OVERLAY}.aa" >&2; exit 3; }}
rm -rf -- {VERIFIER_DIR} || exit 4
mkdir -p -- {MARK_DIR} && rm -f -- {MARKER} && : > {MARKER} || exit 5
"""
"""The agent's box: the upstream image as published (the overlay removed), and the mark
its changes are measured from. A failure is ours (the image is not the task's)."""

CAPTURE_SCRIPT = r"""set -u
max=$1
""" + _WALK + f"""
for d in /logs {ARTIFACTS}; do
  if [ -L "$d" ] || {{ [ -e "$d" ] && [ ! -d "$d" ]; }}; then rm -f -- "$d" || exit 90; fi
  mkdir -p -- "$d" || exit 90
done
rm -rf -- {STATE} {PATHS} || exit 90
[ -f {MARKER} ] && [ ! -L {MARKER} ] || exit 3
work=$(mktemp -d /tmp/reliquary-tb.XXXXXX) || exit 4
export LC_ALL=C
walk > "$work/all" 2>/dev/null
[ -s "$work/all" ] || exit 5
walk -cnewer {MARKER} > "$work/changed" 2>/dev/null
tar --null --no-recursion -C / -czf "$work/state" -T "$work/changed" 2> "$work/tar.err"
rc=$?
[ $rc -le 1 ] || {{ tail -c 300 "$work/tar.err" >&2; exit 6; }}
[ "$(wc -c < "$work/state")" -le "$max" ] || exit 7
gzip -c < "$work/all" > "$work/paths" || exit 8
mv -f -- "$work/state" {STATE} && mv -f -- "$work/paths" {PATHS} || exit 90
cd / && rm -rf -- "$work"
"""
"""The agent's filesystem, from its box: the changed inodes as a tar (owners, modes,
links), and every path (for deletions). Exit 90: a planted path could not be cleared (the
extract fails, graded 0); 3-8: no state (graded 0 without running the tests)."""

CAPTURE_ERRORS = {3: "no_mark", 4: "no_tmp", 5: "walk_failed", 6: "tar_failed",
                  7: "state_too_large", 8: "paths_failed"}

PRISTINE_SCRIPT = r"""set -u
work=""
""" + _WALK + f"""
[ -f {OVERLAY}.aa ] || {{ echo "not this task's verifier image: no {OVERLAY}.aa" >&2; exit 3; }}
rm -rf -- {GRADE_DIR} && mkdir -p -- {GRADE_DIR} || exit 4
export LC_ALL=C
walk 2>/dev/null | sort -z > {GRADE_DIR}/pristine
[ -s {GRADE_DIR}/pristine ] || exit 5
"""
"""The pristine grading box's paths, before anything of the agent's reaches it (ours)."""

APPLY_SCRIPT = f"""set -u
export LC_ALL=C
cd /
cat {OVERLAY}.* | tar -xzpf - -C / || exit 10
gzip -dc {PATHS} | sort -z > {GRADE_DIR}/agent || exit 11
comm -z -23 {GRADE_DIR}/pristine {GRADE_DIR}/agent > {GRADE_DIR}/deleted || exit 12
xargs -0 -r rm -rf -- < {GRADE_DIR}/deleted
tar -xzpf {STATE} -C / --same-owner 2> {GRADE_DIR}/untar.err
rc=$?
[ $rc -le 1 ] || {{ tail -c 300 {GRADE_DIR}/untar.err >&2; exit 13; }}
rm -f -- {STATE} {PATHS}
exit 0
"""
"""The grading box made the agent's: the verifier overlay first (what test.sh's installs
leave, under the agent's files as Harbor's installs land over them), the agent's
deletions, then its files."""

STAGE_SCRIPT = (
    f"rm -rf /tests /logs/verifier && mkdir -p /tests /logs/verifier "
    f"&& tar --no-same-owner -xzf {TESTS_ARCHIVE} -C /tests && rm -f {TESTS_ARCHIVE}"
)


def run_script(workdir: str) -> str:
    """test.sh as Harbor runs it: by absolute path from the image's WORKDIR (created if
    the agent removed it, as `docker exec -w` would fail otherwise)."""
    return (f"mkdir -p {workdir} && cd {workdir} && "
            f"bash /tests/test.sh > {TEST_STDOUT} 2>&1")


class TerminalBenchData(HarborData):
    task_id: str
    upstream_image: str
    """The task's published image (`task.toml`), pinned in `images.UPSTREAM`."""
    tests_sha256: str
    """`source.tests_sha256`: what grading stages, bound into the task's hash."""
    split: str


def _on_sandbox(runtime) -> bool:
    return getattr(getattr(runtime, "config", None), "type", None) == SANDBOX_RUNTIME


def _tail(result) -> str:
    return (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip()[-300:]


class TerminalBenchTask(HarborTask):
    NEEDS_CONTAINER = True

    @property
    def key(self) -> str:
        return self.data.task_id

    async def setup(self, runtime: Runtime) -> None:
        result = await runtime.run(["sh", "-c", SETUP_SCRIPT], {})
        if result.exit_code:
            raise RuntimeError(f"{self.data.task_id}: setup failed (exit {result.exit_code}): "
                               f"{_tail(result)}")

    async def finalize(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["sh", "-c", CAPTURE_SCRIPT, "tb-capture",
                                    str(MAX_STATE_BYTES)], {})
        if result.exit_code == 90:
            raise RuntimeError(f"could not clear {ARTIFACTS}: {_tail(result)}")
        if result.exit_code:
            trace.info["state_error"] = CAPTURE_ERRORS.get(result.exit_code,
                                                           f"exit {result.exit_code}")
        if not _on_sandbox(runtime):
            # verifiers' own lifecycle: the declared artifact travels on the trace.
            await super().finalize(trace, runtime)

    async def grading_setup(self, runtime: Runtime) -> None:
        """The pristine grading box, before the agent's state reaches it."""
        result = await runtime.run(["sh", "-c", PRISTINE_SCRIPT], {})
        if result.exit_code:
            raise RuntimeError(f"{self.data.task_id}: listing the grading box failed "
                               f"(exit {result.exit_code}): {_tail(result)}")

    async def _present(self, runtime: Runtime, path: str) -> bool:
        result = await runtime.run(["sh", "-c", 'test -f "$1" && test ! -L "$1"', "t", path], {})
        return result.exit_code == 0

    async def _reward(self, runtime: Runtime) -> float:
        """Harbor's reading: `reward.json` (a number, or `{"reward": n}`), else
        `reward.txt`; anything else is 0."""
        for path in ("/logs/verifier/reward.json", "/logs/verifier/reward.txt"):
            try:
                raw = await runtime.read(path, max_bytes=MAX_REWARD_BYTES)
            except TimeoutError:
                raise
            except (OSError, vf.SandboxError):
                continue
            text = raw.decode("utf-8", "replace").strip()
            try:
                value = json.loads(text) if path.endswith(".json") else float(text or 0)
            except ValueError:
                return 0.0
            if isinstance(value, dict):
                value = value.get("reward", 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return 0.0
            value = float(value)
            return value if 0.0 <= value <= 1.0 else 0.0
        return 0.0

    async def _summary(self, runtime: Runtime) -> dict:
        info: dict = {}
        try:
            raw = await runtime.read("/logs/verifier/ctrf.json", max_bytes=4 * 1024 * 1024)
            summary = json.loads(raw).get("results", {}).get("summary", {})
            info["tests"] = {k: summary.get(k) for k in ("tests", "passed", "failed", "skipped")}
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
        try:
            tail = await runtime.run(["tail", "-c", "3000", TEST_STDOUT], {})
            info["output_tail"] = (tail.stdout or "")[-3000:]
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
        return info

    @vf.reward(weight=1.0)
    async def solved(self, runtime: Runtime, trace: Trace) -> float:
        """test.sh's verdict on the agent's final filesystem, offline."""
        start = time.monotonic()
        if not (await self._present(runtime, STATE) and await self._present(runtime, PATHS)):
            # No captured state: the agent's box could not be read back (its own doing).
            trace.record_metrics({"state_missing": 1.0})
            return 0.0
        applied = await runtime.run(["sh", "-c", APPLY_SCRIPT], {})
        if applied.exit_code:
            trace.record_metrics({"state_unappliable": 1.0})
            trace.info["terminal_bench"] = {"apply_exit": applied.exit_code,
                                            "apply_error": _tail(applied)}
            return 0.0
        await runtime.write(TESTS_ARCHIVE, source.tests_archive(self.data.task_id))
        staged = await runtime.run(["sh", "-c", STAGE_SCRIPT], {})
        if staged.exit_code:
            raise RuntimeError(f"staging /tests failed (exit {staged.exit_code}): "
                               f"{_tail(staged)}")
        began = time.monotonic()
        result = await runtime.run(["sh", "-c", run_script(self.data.workdir)], GRADING_ENV)
        reward = await self._reward(runtime)
        trace.record_metrics({
            "test_sh_exit_code": float(result.exit_code),
            "test_seconds": float(time.monotonic() - began),
            "grading_seconds": float(time.monotonic() - start),
        })
        trace.info["terminal_bench"] = await self._summary(runtime)
        return reward


def task_data(task_id: str, idx: int, split: str) -> TerminalBenchData:
    """The task's data from its `task.toml`: Harbor's resources and agent timeout as
    declared (the leaderboard forbids changing them), the image's WORKDIR, our verifier
    image. Scoring gets the bridge's most (810 s): test.sh's declared timeouts (360 to
    12,000 s) bound the tests' worst case, the gold check measures the real one."""
    toml = source.task_toml(task_id)
    environment, metadata = toml["environment"], toml.get("metadata", {})
    if environment.get("gpus"):
        raise ValueError(f"{task_id}: GPU tasks are not served")
    return TerminalBenchData(
        idx=idx,
        name=task_id,
        description=toml.get("task", {}).get("description") or None,
        prompt=source.instruction(task_id),
        image=images.image_ref(task_id),
        workdir=source.workdir(task_id),
        network_allow=[],
        timeout=vf.TaskTimeout(setup=120.0, agent=float(toml["agent"]["timeout_sec"]),
                               finalize=600.0, scoring=810.0),
        resources=vf.TaskResources(cpu=float(environment["cpus"]),
                                   memory=environment["memory_mb"] / 1024,
                                   disk=environment["storage_mb"] / 1024),
        artifacts=[vf.Artifact(source=ARTIFACTS, required=False)],
        difficulty=metadata.get("difficulty"),
        category=metadata.get("category"),
        tags=list(metadata.get("tags", [])),
        task_id=task_id,
        upstream_image=environment["docker_image"],
        tests_sha256=source.tests_sha256(task_id),
        split=split,
    )


__all__ = ["APPLY_SCRIPT", "CAPTURE_SCRIPT", "GRADING_ENV", "MAX_ARTIFACT_BYTES",
           "MAX_STATE_BYTES", "PATHS",
           "PRISTINE_SCRIPT", "SANDBOX_RUNTIME", "SETUP_SCRIPT", "STATE", "TerminalBenchData",
           "TerminalBenchTask", "task_data"]
