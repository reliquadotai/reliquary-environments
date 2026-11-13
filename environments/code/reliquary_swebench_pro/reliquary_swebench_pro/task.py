"""One SWE-bench Pro V2 task under the env norm (docs/env-norm.md).

* `setup` (the agent's box): checks the image is the task's (`/app` at the base commit;
  a mismatch is ours: the open fails) and records the files it ships untracked.
* `finalize` (the agent's box): everything the agent changed against the base, committed
  or not, as Scale's locked protocol captures it (`git add -A; git diff --cached`), written
  to `/logs/artifacts/model.patch` (the only declared artifact). The work tree is staged in
  a scratch repository of ours over the agent's objects, so no git setting, hook,
  attribute driver or redirected `.git` the agent left runs; the image's untracked files
  and toolchain directories are left out; a patch planted there is removed first.
* `grading_setup` (a pristine box of the image): checks `/app` is at the base commit.
* `solved` (the same box, the patch restored): refuses a patch of a shape no candidate
  needs (`scoring.patch_refusal`), applies it as Scale's replay does (`git apply`, then
  `--3way`, then `patch --fuzz=3`), stages upstream's `tests/` as `/tests`, runs the
  task's `test.sh` unchanged (it restores test files, applies the test patch, runs the
  task's `run_script.sh`), then scores from the run script's logs in the scoring process
  (`scoring`). The box's own `reward.txt` is never read.

Nothing of an episode is kept on the task: every hook acts through the box it is given.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import io
import tarfile
import time
import uuid

import verifiers.v1 as vf
from verifiers.v1.runtimes import Runtime
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask
from verifiers.v1.trace import Trace

from reliquary_swebench_pro import scoring, source

SANDBOX_RUNTIME = "reliquary-sandbox"
"""`runtime.config.type` of a signed-episode sandbox's boxes."""
WORKDIR = "/app"
PATCH_PATH = "/logs/artifacts/model.patch"
MAX_PATCH_BYTES = 16 * 1024 * 1024
TESTS_ARCHIVE = "/tmp/swepro-tests.tgz"
TEST_STDOUT = "/tmp/swepro-test-stdout.txt"
_HARDENED_GIT = (
    'export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_ATTR_NOSYSTEM=1 '
    'GIT_NO_REPLACE_OBJECTS=1 GIT_NO_LAZY_FETCH=1 GIT_TERMINAL_PROMPT=0 '
    'HOME="$scratch.home" XDG_CONFIG_HOME="$scratch.home"\n'
)

HEAD_SCRIPT = (
    'scratch=/tmp/swepro-head\n' + _HARDENED_GIT +
    'exec git -c safe.directory=* --git-dir="$1/.git" rev-parse --verify -q HEAD\n'
)
"""Prints the object name HEAD resolves to in `$1`'s repository; reads no object, runs
no hook."""

TOOLCHAIN_EXCLUDES = tuple(f"{d}/" for d in sorted(scoring.REFUSED_DIRS - {".git"}))
"""Never captured: dependency and build-cache directories the agent's commands create
The grader refuses a path there anyway."""

UNTRACKED_EXCLUDE = ".git/swepro-image-untracked"
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
    f'rm -rf /tests {scoring.VERIFIER_DIR} && mkdir -p /tests {scoring.VERIFIER_DIR} '
    f'&& tar --no-same-owner -xzf {TESTS_ARCHIVE} -C /tests && rm -f {TESTS_ARCHIVE}'
)
REPLAY_PATCH = "/tmp/swepro-replay.patch"
APPLY_SCRIPT = (
    f'cd {WORKDIR} && cp -- {PATCH_PATH} {REPLAY_PATCH} && '
    f'{{ git apply --verbose {REPLAY_PATCH} || git apply --3way {REPLAY_PATCH} '
    f'|| patch --batch --fuzz=3 -p1 -i {REPLAY_PATCH}; }}'
)
"""Scale's `patch_replay.py` fallbacks, in order (`--batch`: never ask)."""
RUN_SCRIPT = f'cd {WORKDIR} && bash /tests/test.sh > {TEST_STDOUT} 2>&1'
GRADING_ENV = {"GIT_TERMINAL_PROMPT": "0"}


@functools.cache
def tests_archive(task_id: str) -> bytes:
    """Upstream's `tests/` as a reproducible tar.gz (fixed mtime and owner)."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, data in sorted(source.tests_files(task_id).items()):
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.uid, info.gid = len(data), 0, 0, 0
                info.mode = 0o755 if name.endswith(".sh") else 0o644
                tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def tests_sha256(task_id: str) -> str:
    """What grading stages and scores with: `/tests` and the parser, bound into the
    task's hash."""
    digest = hashlib.sha256()
    for name, data in sorted(source.tests_files(task_id).items()):
        digest.update(f"{name}\0{len(data)}\0".encode() + data)
    return digest.hexdigest()


class SweProData(HarborData):
    task_id: str
    language: str
    repo: str
    base_commit: str
    hard: bool = False
    tests_sha256: str
    """`tests_sha256`: what grading stages, bound into the task's hash."""
    split: str


def _on_sandbox(runtime) -> bool:
    return getattr(getattr(runtime, "config", None), "type", None) == SANDBOX_RUNTIME


class SweProTask(HarborTask):
    NEEDS_CONTAINER = True

    @property
    def key(self) -> str:
        return self.data.task_id

    async def _head(self, runtime: Runtime) -> str:
        result = await runtime.run(["sh", "-c", HEAD_SCRIPT, "swepro-head", WORKDIR], {})
        return (result.stdout or "").strip() if result.exit_code == 0 else ""

    async def _check_base(self, runtime: Runtime, box: str) -> None:
        head = await self._head(runtime)
        if head != self.data.base_commit:
            raise RuntimeError(f"{self.data.task_id}: the {box} box's {WORKDIR} is at "
                               f"{head or 'nothing'}, not the base {self.data.base_commit}")

    async def setup(self, runtime: Runtime) -> None:
        """The agent's box is the task's image, untouched: `/app` at the base commit. The
        files the image ships untracked are recorded in the box (`UNTRACKED_EXCLUDE`), so
        the capture leaves them out: in the grading box they exist already, and a patch
        creating them would not apply."""
        await self._check_base(runtime, "agent")
        scratch = f"/tmp/swepro-list-{uuid.uuid4().hex}"
        listing = await runtime.run(["sh", "-c", LIST_UNTRACKED_SCRIPT, "swepro-list",
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
        scratch = f"/tmp/swepro-capture-{uuid.uuid4().hex}"
        result = await runtime.run(
            ["sh", "-c", CAPTURE_SCRIPT, "swepro-capture", self.data.base_commit, WORKDIR,
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

    async def _read(self, runtime: Runtime, path: str, cap: int) -> bytes | None:
        try:
            return await runtime.read(path, max_bytes=cap)
        except TimeoutError:
            raise  # the step deadline
        except (OSError, vf.SandboxError):
            return None  # missing, too large, or not a regular file

    @vf.reward(weight=1.0)
    async def solved(self, runtime: Runtime, trace: Trace) -> float | dict[str, float]:
        """1 iff every fail-to-pass and pass-to-pass test is reported PASSED."""
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
            # The base state: every fail-to-pass test fails there (upstream's nop gate).
            trace.record_metrics({"empty_patch": 1.0})
            return 0.0
        refusal = scoring.patch_refusal(patch)
        if refusal is not None:
            trace.record_metrics({"patch_refused": 1.0})
            trace.info["swebench_pro"] = {"patch_refused": refusal}
            return 0.0
        applied = await runtime.run(["sh", "-c", APPLY_SCRIPT], GRADING_ENV)
        if applied.exit_code:
            trace.record_metrics({"patch_failed": 1.0})
            trace.info["swebench_pro"] = {
                "patch_failed": (applied.stderr or applied.stdout or "")[-1000:]}
            return 0.0

        await runtime.write(TESTS_ARCHIVE, tests_archive(self.data.task_id))
        staged = await runtime.run(["sh", "-c", STAGE_SCRIPT], {})
        if staged.exit_code:
            raise RuntimeError(f"staging /tests failed (exit {staged.exit_code}): "
                               f"{(staged.stderr or '').strip()[-300:]}")
        began = time.monotonic()
        result = await runtime.run(["sh", "-c", RUN_SCRIPT], GRADING_ENV)
        test_seconds = time.monotonic() - began
        stdout = await self._read(runtime, scoring.STDOUT_LOG, scoring.MAX_LOG_BYTES)
        stderr = await self._read(runtime, scoring.STDERR_LOG, scoring.MAX_LOG_BYTES)
        assessment = scoring.assess(self.data.task_id, stdout, stderr)
        trace.record_metrics({
            **{k: assessment[k] for k in scoring.METRIC_KEYS if k in assessment},
            "test_sh_exit_code": float(result.exit_code),
            "test_seconds": float(test_seconds),
            "grading_seconds": float(time.monotonic() - start),
        })
        info = scoring.summary(assessment)
        try:
            tail = await runtime.run(["tail", "-c", "3000", TEST_STDOUT], {})
            info["output_tail"] = (tail.stdout or "")[-3000:]
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
        trace.info["swebench_pro"] = info
        return assessment["reward"]


def task_data(task_id: str, idx: int, split: str) -> SweProData:
    """The task's data, from its `task.toml` and `instance.json` (all 642 declare the same
    box: 1 CPU, 4 GB, 10 GB; agent 3000 s). No box has network: upstream's agent runs
    offline, and its verifier's `public` network is not granted (`gate`)."""
    toml, row = source.task_toml(task_id), source.instance(task_id)
    environment, metadata = toml["environment"], toml["metadata"]
    if environment["docker_image"] != row["docker_image"]:
        raise ValueError(f"{task_id}: task.toml and the dataset name different images")
    return SweProData(
        idx=idx,
        name=task_id,
        description=None,
        prompt=source.instruction(task_id),
        image=environment["docker_image"],
        workdir=WORKDIR,
        network_allow=[],
        timeout=vf.TaskTimeout(setup=120.0, agent=float(toml["agent"]["timeout_sec"]),
                               finalize=300.0, scoring=810.0),
        resources=vf.TaskResources(cpu=float(environment["cpus"]),
                                   memory=environment["memory_mb"] / 1024,
                                   disk=environment["storage_mb"] / 1024),
        artifacts=[vf.Artifact(source=PATCH_PATH, required=False)],
        category=metadata.get("category"),
        difficulty=metadata.get("difficulty"),
        tags=[row["repo_language"]] + (["hard"] if row["hard"] else []),
        task_id=task_id,
        language=row["repo_language"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        hard=bool(row["hard"]),
        tests_sha256=tests_sha256(task_id),
        split=split,
    )


__all__ = ["APPLY_SCRIPT", "CAPTURE_SCRIPT", "GRADING_ENV", "PATCH_PATH", "RUN_SCRIPT",
           "SANDBOX_RUNTIME", "STAGE_SCRIPT", "SweProData", "SweProTask", "WORKDIR",
           "task_data", "tests_archive", "tests_sha256"]
