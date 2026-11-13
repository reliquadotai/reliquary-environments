"""One DeepSWE v1.1 task under the env norm (docs/env-norm.md).

* `setup` (the agent's box): checks the image is the task's (`/app` at the base commit;
  a mismatch is ours: the open fails) and records the files it ships untracked.
* `finalize` (the agent's box): everything the agent changed against the base,
  committed or not (a deviation from upstream, which grades `git diff <base> HEAD`, the
  committed work only: here a harness that never commits is graded on what it did, so
  scores compare across harnesses), written to `/logs/artifacts/model.patch` (the only
  declared artifact). The work tree is staged in a scratch repository of ours over the
  agent's objects, so no git setting, hook, attribute driver or redirected `.git` the
  agent left runs; the image's untracked files and toolchain directories are left out; a
  patch planted there is removed first.
* `grading_setup` (a pristine box of the image): checks `/app` is at the base commit.
* `solved` (the same box, the patch restored): refuses a patch of a shape no candidate
  needs (`scoring.patch_refusal`), stages `/tests` (`overlay`), runs the task's `test.sh`
  (`verifier.prepare` restores test files and applies the patch and the test patch, the
  suites run), then scores from the reports in the scoring process (`scoring`). A run whose
  only failures are timeouts, with every fail-to-pass test passing, is run twice more when
  the budget allows, and recovers only if both pass (PI-DeepSWE's rule).

Nothing of an episode is kept on the task: every hook acts through the box it is given.
"""

from __future__ import annotations

import time
import uuid

import verifiers.v1 as vf
from verifiers.v1.runtimes import Runtime
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask
from verifiers.v1.trace import Trace

from reliquary_deepswe import overlay, scoring

SANDBOX_RUNTIME = "reliquary-sandbox"
"""`runtime.config.type` of a signed-episode sandbox's boxes."""
WORKDIR = "/app"
PATCH_PATH = "/logs/artifacts/model.patch"
MAX_PATCH_BYTES = 16 * 1024 * 1024
TESTS_ARCHIVE = "/tmp/deepswe-tests.tgz"
TEST_STDOUT = "/tmp/deepswe-test-stdout.txt"
MAX_DISK_GB = 10.0
MAX_ATTEMPTS = 3
RETRY_BUDGET_S = 700.0
"""A timeout retry starts only when both retries, each as long as the first run, end
within this many seconds of the reward's start: the sandbox grants a grading step 810 s,
which also covers the restore and the scoring process's start."""

GRADING_ENV = {
    # PI-DeepSWE's: Go's build cache where test.sh expects it, tests re-run every attempt.
    "GOCACHE": "/app/.gocache",
    "GOFLAGS": "-count=1",
    # No box has network: a module fetch fails at once instead of waiting on a resolver.
    # Toolchains and checksums are left to Go's defaults: several images pin a newer Go
    # than they ship, with that toolchain and its sumdb entries in the module cache, which
    # Go uses offline (GOSUMDB=off or GOTOOLCHAIN=local would refuse it; measured).
    "GOPROXY": "off",
    "GIT_TERMINAL_PROMPT": "0",
}

_HARDENED_GIT = (
    'export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_ATTR_NOSYSTEM=1 '
    'GIT_NO_REPLACE_OBJECTS=1 GIT_NO_LAZY_FETCH=1 GIT_TERMINAL_PROMPT=0 '
    'HOME="$scratch.home" XDG_CONFIG_HOME="$scratch.home"\n'
)

HEAD_SCRIPT = (
    'scratch=/tmp/deepswe-head\n' + _HARDENED_GIT +
    'exec git -c safe.directory=* --git-dir="$1/.git" rev-parse --verify -q HEAD\n'
)
"""Prints the object name HEAD resolves to in `$1`'s repository; reads no object, runs
no hook."""

TOOLCHAIN_EXCLUDES = tuple(f"{d}/" for d in sorted(scoring.REFUSED_DIRS - {".git"})) + (
    "/target/",)
"""Never captured: dependency and build-cache directories the agent's commands create
(`GOCACHE=/app/.gocache` among them). The grader refuses a path there anyway."""

UNTRACKED_EXCLUDE = ".git/deepswe-image-untracked"
"""Under the workdir: `setup`'s exclude patterns for the files the image ships untracked
(in the agent's box; editing it only widens or narrows the agent's own diff)."""
MAX_UNTRACKED_BYTES = 4 * 1024 * 1024

_SCRATCH = (
    'rm -rf -- "$scratch" && mkdir -m 700 -- "$scratch" '
    '&& git init -q --bare --template= "$scratch" && mkdir -p -- "$scratch/info" || exit 4\n'
    "printf '%s\\n' " + " ".join(f"'{p}'" for p in TOOLCHAIN_EXCLUDES)
    + ' > "$scratch/info/exclude" || exit 4\n'
    f'x="$work/{UNTRACKED_EXCLUDE}"\n'
    'if [ -f "$x" ] && [ ! -L "$x" ]; then '
    f'head -c {MAX_UNTRACKED_BYTES} -- "$x" >> "$scratch/info/exclude"; fi\n'
    '[ -d "$work/.git/objects" ] && [ ! -L "$work/.git" ] || exit 3\n'
    'export GIT_DIR="$scratch" GIT_WORK_TREE="$work" '
    'GIT_ALTERNATE_OBJECT_DIRECTORIES="$work/.git/objects"\n'
    "G() { git -c core.bare=false -c core.hooksPath=/dev/null -c core.fsmonitor=false "
    "-c core.untrackedCache=false -c core.splitIndex=false -c safe.directory='*' \"$@\"; }\n"
    'b=$(G rev-parse --verify -q "$base^{commit}") && G read-tree "$b" && cd -- "$work" '
    '|| exit 5\n'
)
"""A scratch repository of ours over the agent's work tree: index at the base tree, the
agent's objects read as an alternate (data); no config, hook, fsmonitor or attribute
driver of the agent's runs. In-tree `.gitignore` files are read: they only hide paths."""

LIST_UNTRACKED_SCRIPT = (
    'set -u\nbase=$1 work=$2 scratch=$3\n' + _HARDENED_GIT + _SCRATCH
    + 'G ls-files --others --exclude-standard -z; rc=$?\ncd /\n'
    'rm -rf -- "$scratch" "$scratch.home"\nexit $rc\n'
)
"""The image's untracked, unignored files (NUL-separated), listed once in `setup`."""

CAPTURE_SCRIPT = r'''
set -u
base=$1 work=$2 out=$3 scratch=$4 max=$5
''' + _HARDENED_GIT + r'''
art=$(dirname -- "$out")
for d in "$(dirname -- "$art")" "$art"; do
  if [ -L "$d" ] || { [ -e "$d" ] && [ ! -d "$d" ]; }; then rm -f -- "$d" || exit 90; fi
  mkdir -p -- "$d" || exit 90
done
rm -rf -- "$out" || exit 90
[ ! -e "$out" ] && [ ! -L "$out" ] || exit 90
rm -f -- "$scratch.patch"
''' + _SCRATCH + r'''
G add -A && G diff --cached --binary --no-ext-diff --no-textconv --no-color "$b" -- \
  > "$scratch.patch"
rc=$?
cd /
rm -rf -- "$scratch" "$scratch.home"
if [ $rc -ne 0 ]; then rm -f -- "$scratch.patch"; exit 5; fi
if [ "$(wc -c < "$scratch.patch")" -gt "$max" ]; then rm -f -- "$scratch.patch"; exit 6; fi
mv -f -- "$scratch.patch" "$out" || exit 90
'''
"""Everything the agent changed against the base, committed or not: its work tree staged
over the base tree in the scratch repository (`git add -A`), then `git diff --cached
--binary <base>`. Files the image ships untracked and toolchain directories are left out.
Exit 90: a planted path could not be removed (the extract fails, graded 0); 3-6: no
patch (an empty submission)."""

CAPTURE_ERRORS = {3: "no_repository", 4: "scratch_failed", 5: "diff_failed",
                  6: "patch_too_large"}


def exclude_patterns(paths: list[str]) -> str:
    """Anchored gitignore patterns matching exactly these paths."""
    lines = []
    for path in paths:
        if "\n" in path or "\r" in path:
            raise ValueError(f"an untracked path with a line break: {path!r}")
        escaped = "".join("\\" + c if c in "\\*?[" else c for c in path)
        if escaped.endswith(" "):
            escaped = escaped[:-1] + "\\ "
        lines.append("/" + escaped + "\n")
    return "".join(lines)

STAGE_SCRIPT = (
    'rm -rf /tests /logs/verifier && mkdir -p /tests /logs/verifier '
    f'&& tar --no-same-owner -xzf {TESTS_ARCHIVE} -C /tests && rm -f {TESTS_ARCHIVE}'
)
RUN_SCRIPT = f'cd {WORKDIR} && bash /tests/test.sh > {TEST_STDOUT} 2>&1'


class DeepSWEData(HarborData):
    task_id: str
    language: str
    base_commit: str
    repository_url: str = ""
    tests_sha256: str
    """`overlay.tests_sha256`: what grading stages, bound into the task's hash."""
    split: str


def _on_sandbox(runtime) -> bool:
    return getattr(getattr(runtime, "config", None), "type", None) == SANDBOX_RUNTIME


class DeepSWETask(HarborTask):
    NEEDS_CONTAINER = True

    @property
    def key(self) -> str:
        return self.data.task_id

    async def _head(self, runtime: Runtime) -> str:
        result = await runtime.run(["sh", "-c", HEAD_SCRIPT, "deepswe-head", WORKDIR], {})
        return (result.stdout or "").strip() if result.exit_code == 0 else ""

    async def _check_base(self, runtime: Runtime, box: str) -> None:
        head = await self._head(runtime)
        # Three tasks name their base by an abbreviated hash (7 or 39 hex digits).
        if not (len(head) == 40 and head.startswith(self.data.base_commit)):
            raise RuntimeError(f"{self.data.task_id}: the {box} box's {WORKDIR} is at "
                               f"{head or 'nothing'}, not the base {self.data.base_commit}")

    async def setup(self, runtime: Runtime) -> None:
        """The agent's box is the task's image, untouched: `/app` at the base commit. The
        files the image ships untracked are recorded in the box (`UNTRACKED_EXCLUDE`), so
        the capture leaves them out: in the grading box they exist already, and a patch
        creating them would not apply."""
        await self._check_base(runtime, "agent")
        scratch = f"/tmp/deepswe-list-{uuid.uuid4().hex}"
        listing = await runtime.run(["sh", "-c", LIST_UNTRACKED_SCRIPT, "deepswe-list",
                                     self.data.base_commit, WORKDIR, scratch], {})
        if listing.exit_code:
            raise RuntimeError(f"{self.data.task_id}: listing the image's untracked files "
                               f"failed (exit {listing.exit_code}): "
                               f"{(listing.stderr or '').strip()[-300:]}")
        patterns = exclude_patterns([p for p in (listing.stdout or "").split("\0") if p])
        if len(patterns.encode()) > MAX_UNTRACKED_BYTES:
            raise RuntimeError(f"{self.data.task_id}: the image's untracked list is over "
                               f"{MAX_UNTRACKED_BYTES} bytes")
        await runtime.write(f"{WORKDIR}/{UNTRACKED_EXCLUDE}", patterns.encode())

    async def finalize(self, trace: Trace, runtime: Runtime) -> None:
        scratch = f"/tmp/deepswe-capture-{uuid.uuid4().hex}"
        result = await runtime.run(
            ["sh", "-c", CAPTURE_SCRIPT, "deepswe-capture", self.data.base_commit, WORKDIR,
             PATCH_PATH, scratch, str(MAX_PATCH_BYTES)], {})
        if result.exit_code == 90:
            raise RuntimeError(f"could not clear {PATCH_PATH}: "
                               f"{(result.stderr or '').strip()[-300:]}")
        if result.exit_code:
            trace.info["patch_error"] = CAPTURE_ERRORS.get(result.exit_code,
                                                           f"exit {result.exit_code}")
        if not _on_sandbox(runtime):
            # verifiers' own lifecycle: the declared artifact travels on the trace.
            await super().finalize(trace, runtime)

    async def grading_setup(self, runtime: Runtime) -> None:
        """The pristine grading box, before the agent's patch reaches it."""
        await self._check_base(runtime, "grading")

    async def _stage(self, runtime: Runtime) -> None:
        await runtime.write(TESTS_ARCHIVE, overlay.tests_archive(self.data.task_id))
        result = await runtime.run(["sh", "-c", STAGE_SCRIPT], {})
        if result.exit_code:
            raise RuntimeError(f"staging /tests failed (exit {result.exit_code}): "
                               f"{(result.stderr or '').strip()[-300:]}")

    async def _read(self, runtime: Runtime, path: str, cap: int) -> bytes | None:
        try:
            return await runtime.read(path, max_bytes=cap)
        except TimeoutError:
            raise  # the step deadline
        except (OSError, vf.SandboxError):
            return None  # missing, too large, or not a regular file

    async def _assess(self, runtime: Runtime) -> dict:
        config = overlay.effective_config(self.data.task_id)
        reports: dict[str, bytes | None] = {}
        for path in config["grade"]["reports"]:
            for location in scoring.report_locations(path):
                raw = await self._read(runtime, location, scoring.MAX_REPORT_BYTES)
                if raw is not None:
                    reports[path] = raw
                    break
        run_log = None
        if self.data.language == "go":
            raw = await self._read(runtime, f"{scoring.VERIFIER_DIR}/run.log",
                                   scoring.MAX_RUN_LOG_BYTES)
            run_log = raw.decode("utf-8", "replace") if raw is not None else None
        return scoring.assess_reports(config, self.data.language, reports, run_log)

    @vf.reward(weight=1.0)
    async def solved(self, runtime: Runtime, trace: Trace) -> float | dict[str, float]:
        """1 iff every fail-to-pass test passes and no pass-to-pass test fails."""
        if self._graded_elsewhere:
            return {}
        start = time.monotonic()
        try:
            patch = await runtime.read(PATCH_PATH, max_bytes=MAX_PATCH_BYTES)
        except FileNotFoundError:
            patch = b""
        except TimeoutError:
            raise
        except (OSError, vf.SandboxError):
            trace.record_metrics({"patch_unreadable": 1.0})
            return 0.0
        if not patch.strip():
            # The base state: every fail-to-pass test fails there by construction
            # (upstream's whitelists come from the oracle-vs-nop differential).
            trace.record_metrics({"empty_patch": 1.0})
            return 0.0
        refusal = scoring.patch_refusal(patch)
        if refusal is not None:
            trace.record_metrics({"patch_refused": 1.0})
            trace.info["deepswe"] = {"patch_refused": refusal}
            return 0.0

        attempts: list[tuple[dict, float, int]] = []
        for attempt in range(MAX_ATTEMPTS):
            await self._stage(runtime)
            began = time.monotonic()
            result = await runtime.run(["sh", "-c", RUN_SCRIPT], GRADING_ENV)
            assessment = await self._assess(runtime)
            attempts.append((assessment, time.monotonic() - began, result.exit_code))
            if attempt == 0:
                first = attempts[0][1]
                if not scoring.retry_eligible(assessment):
                    break
                if time.monotonic() - start + 2 * first > RETRY_BUDGET_S:
                    trace.record_metrics({"retry_skipped_for_budget": 1.0})
                    break
            elif not assessment["scores"].get("reward"):
                break
        recovered = len(attempts) == MAX_ATTEMPTS and all(
            a[0]["scores"].get("reward") == 1 for a in attempts[1:])
        selected, seconds, exit_code = attempts[-1] if recovered else attempts[0]
        scores = selected["scores"]
        trace.record_metrics({
            **{k: float(scores[k]) for k in scoring.METRIC_KEYS if k in scores},
            "timeout_retries": float(len(attempts) - 1),
            "retry_recovered": float(recovered),
            "test_sh_exit_code": float(exit_code),
            "test_seconds": float(seconds),
            "grading_seconds": float(time.monotonic() - start),
        })
        info = scoring.summary(selected)
        try:
            tail = await runtime.run(["tail", "-c", "3000", TEST_STDOUT], {})
            info["output_tail"] = (tail.stdout or "")[-3000:]
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
        trace.info["deepswe"] = info
        return float(scores.get("reward", 0))


def task_data(task_id: str, idx: int, split: str, toml: dict, prompt: str) -> DeepSWEData:
    """The task's data, from its `task.toml` (all 113 declare the same box: 2 CPUs, 8 GB,
    20 GB, no network; agent 5400 s), the disk capped at `MAX_DISK_GB`."""
    environment, metadata = toml["environment"], toml["metadata"]
    return DeepSWEData(
        idx=idx,
        name=task_id,
        description=metadata.get("display_title") or None,
        prompt=prompt,
        image=environment["docker_image"],
        workdir=WORKDIR,
        network_allow=[],
        timeout=vf.TaskTimeout(setup=120.0, agent=float(toml["agent"]["timeout_sec"]),
                               finalize=300.0, scoring=810.0),
        # Disk capped at 10 GiB: a session's budgets must cover the task's needs, and both
        # a gateway (episode_max_disk_bytes) and Catalyst's episode budgets stop at 10 GiB;
        # task.toml's 20 GB would refuse every session. It bounds the box's writable layer.
        resources=vf.TaskResources(cpu=float(environment["cpus"]),
                                   memory=environment["memory_mb"] / 1024,
                                   disk=min(environment["storage_mb"] / 1024, MAX_DISK_GB)),
        artifacts=[vf.Artifact(source=PATCH_PATH, required=False)],
        category=metadata.get("category"),
        tags=[metadata["language"]],
        task_id=task_id,
        language=metadata["language"],
        base_commit=metadata["base_commit_hash"],
        repository_url=metadata.get("repository_url", ""),
        tests_sha256=overlay.tests_sha256(task_id),
        split=split,
    )


__all__ = ["CAPTURE_SCRIPT", "DeepSWEData", "DeepSWETask", "GRADING_ENV", "PATCH_PATH",
           "SANDBOX_RUNTIME", "WORKDIR", "task_data"]
