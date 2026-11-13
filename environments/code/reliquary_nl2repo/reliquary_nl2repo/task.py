"""One NL2Repo-Bench task under the env norm (docs/env-norm.md).

* `setup` (the agent's box): the task's image carries the hidden tests and upstream's
  packaging in `/workspace`, and some carry the reference itself (a source copy left in
  the image, a package installed from it, a pip cache). All of it goes: the image's
  reference copies (`hygiene.REFERENCE_PATHS`), pip caches, every distribution installed
  from a local directory (PEP 610 `dir_info`), and `/workspace`'s content. Then `start.md`
  is written to `/workspace`. A failure here is ours: the open fails.
* `finalize` (the agent's box): `/workspace` as `/logs/artifacts/workspace.tar.gz`
  (regular files, toolchain and cache directories left out, at most 24 MiB); the artifact
  directory is emptied first, so nothing planted there travels.
* `grading_setup` (a pristine box of the image): the same hygiene, `/workspace` kept (it
  holds the hidden tests); a box without test files is ours.
* `pass_rate` (the same box, the archive restored): the archive is read and rebuilt in the
  scoring process (`submission.sanitize`), extracted over `/workspace` without replacing
  any file the image has, then upstream's test commands run through bash, offline, within
  the budget, each pytest run writing a JUnit report. Reward = min(passed /
  `test_case_count`, 1), upstream's metric, counted from the reports.

Nothing of an episode is kept on the task: every hook acts through the box it is given.
"""

from __future__ import annotations

import re
import shlex
import time
from xml.etree.ElementTree import ParseError

import verifiers.v1 as vf
from verifiers.v1.runtimes import Runtime
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask
from verifiers.v1.trace import Trace

from reliquary_nl2repo import hygiene, submission

SANDBOX_RUNTIME = "reliquary-sandbox"
"""`runtime.config.type` of a signed-episode sandbox's boxes."""
WORKDIR = "/workspace"
ARCHIVE_PATH = "/logs/artifacts/workspace.tar.gz"
CLEAN_ARCHIVE = "/tmp/nl2repo-submission.tar"
INSTRUCTION = (
    "According to the start.md in the workspace, implement the entire project as per the "
    "requirements specified in the document, ensuring that the final product can be "
    "directly run in the current directory. The running requirements should comply with "
    "the <API Usage Guide> section of the document. Please complete this task step by step."
)
"""Upstream's prompt (openhands/openhands_app.py at the pin), unchanged."""

TEST_BUDGET_S = 660
"""All test commands together; the 810 s grading step also covers the restore, the
archive's rebuild and the reports' reads."""
JUNIT = "/tmp/nl2repo-junit-{}.xml"
LOG = "/tmp/nl2repo-cmd-{}.log"
EXIT = "/tmp/nl2repo-cmd-{}.exit"
MAX_LOG_TAIL = 3000

TOOLCHAIN_PRUNE = (".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache",
                   ".mypy_cache", "*.egg-info")

FINALIZE_SCRIPT = r'''
set -u
out=$1 max=$2 work=$3
art=$(dirname -- "$out")
for d in "$(dirname -- "$art")" "$art"; do
  if [ -L "$d" ] || { [ -e "$d" ] && [ ! -d "$d" ]; }; then rm -f -- "$d" || exit 90; fi
  mkdir -p -- "$d" || exit 90
done
find "$art" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + || exit 90
[ -z "$(ls -A -- "$art")" ] || exit 90
cd -- "$work" 2>/dev/null || exit 3
tmp=$(mktemp /tmp/nl2repo-ws.XXXXXX) || exit 4
find . -mindepth 1 \( -type d \( ''' + " -o ".join(f"-name '{d}'" for d in TOOLCHAIN_PRUNE) + r''' \
  -o -path ./build -o -path ./dist \) -prune \) -o -type f -print0 \
  | tar --null --no-recursion -T - -czf "$tmp" || { rm -f -- "$tmp"; exit 5; }
if [ "$(wc -c < "$tmp")" -gt "$max" ]; then rm -f -- "$tmp"; exit 6; fi
mv -f -- "$tmp" "$out" || exit 90
'''
"""The agent's workspace as a gzipped tar of regular files. Exit 90: the artifact
directory could not be cleared (the extract fails, graded 0); 3-6: no archive."""
CAPTURE_ERRORS = {3: "no_workspace", 4: "tmp_failed", 5: "archive_failed",
                  6: "workspace_too_large"}

CLEAR_WORKSPACE = (
    'mkdir -p /workspace && find /workspace -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + '
    '&& [ -z "$(ls -A /workspace)" ]'
)
CLEAR_TMP = "find /tmp -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + 2>/dev/null; exit 0"
"""Best effort: an image whose user is not root (pyautogui) keeps root's files there."""
HAS_TESTS = "find /workspace -type f -name '*.py' | head -n 1 | grep -q ."
EXTRACT_SCRIPT = (
    f"mkdir -p {WORKDIR} && tar -xf {CLEAN_ARCHIVE} -C {WORKDIR} --no-same-owner "
    f"--no-same-permissions --skip-old-files && rm -f {CLEAN_ARCHIVE}"
)

_PIP_INSTALL = re.compile(r"\b((?:python3?\s+-m\s+)?pip3?\s+install)\b")
_PYTEST = re.compile(r"\bpytest\b")
_SET = re.compile(r"^set ([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def plan_command(command: str, index: int) -> tuple[str, bool]:
    """(shell text, is_pytest) for upstream's command `index`.

    Upstream runs each command without a shell (`shlex.split`), which breaks seven tasks'
    commands (`&&`, redirects, `set X=Y`, `VAR=x cmd`); here every command runs in bash,
    `set X=Y` (Windows syntax) becomes an export, as Prime's adaptation does. pip runs
    offline with the image's build tools (`--no-build-isolation --no-deps`; `PIP_NO_INDEX`
    is set); a pytest run also writes a JUnit report."""
    match = _SET.match(command.strip())
    if match:
        return f"export {match.group(1)}={shlex.quote(match.group(2))}", False
    text = _PIP_INSTALL.sub(r"\1 --no-build-isolation --no-deps", command)
    is_pytest = bool(_PYTEST.search(command)) and not _PIP_INSTALL.search(command)
    if is_pytest:
        text = f"{text} --junitxml={JUNIT.format(index)}"
    return text, is_pytest


def test_script(commands: list[str], budget_s: int = TEST_BUDGET_S, workdir: str = WORKDIR) -> str:
    """Upstream's commands, in order, in one bash script: exports in the script's own
    shell, every other command in a child shell bounded by what is left of the budget."""
    lines = [
        f"cd {shlex.quote(workdir)} || exit 1",
        f'export PYTHONPATH="{workdir}:{workdir}/src${{PYTHONPATH:+:$PYTHONPATH}}"',
        "export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 "
        "PIP_ROOT_USER_ACTION=ignore PYTHONDONTWRITEBYTECODE=1",
        f"end=$(( $(date +%s) + {int(budget_s)} ))",
        "run() {",
        '  left=$(( end - $(date +%s) ))',
        f'  if [ "$left" -le 5 ]; then echo skipped > {EXIT.format("$1")}; return; fi',
        f'  timeout -k 5 "$left" bash -c "$2" > {LOG.format("$1")} 2>&1 < /dev/null',
        f'  echo $? > {EXIT.format("$1")}',
        "}",
    ]
    for index, command in enumerate(commands):
        text, _ = plan_command(command, index)
        if text.startswith("export "):
            lines.append(text)
        else:
            lines.append(f"run {index} {shlex.quote(text)}")
    return "\n".join(lines) + "\n"


_SUMMARY = re.compile(r"(\d+) passed")


def summary_passed(log: str) -> int:
    """Upstream's count, for diagnostics: every `N passed` on pytest's output lines."""
    return sum(int(m.group(1)) for line in log.splitlines() for m in _SUMMARY.finditer(line)
               if not re.search(r"\d+ xpassed", line[m.start():m.end() + 1]))


class NL2RepoData(HarborData):
    task_id: str
    spec_sha256: str
    """sha256 of `start.md`, which setup writes to the workspace: bound into the hash."""
    test_commands: list[str]
    test_files: list[str]
    test_case_count: int
    reference_paths: list[str]
    own_modules: list[str]
    """Import names of the project that look like pytest plugins (`hygiene.OWN_MODULES`)."""
    split: str


def _on_sandbox(runtime) -> bool:
    return getattr(getattr(runtime, "config", None), "type", None) == SANDBOX_RUNTIME


class NL2RepoTask(HarborTask):
    NEEDS_CONTAINER = True

    @property
    def key(self) -> str:
        return self.data.task_id

    async def _run(self, runtime: Runtime, script: str, *args: str, what: str,
                   name: str = "nl2repo") -> str:
        result = await runtime.run(["sh", "-c", script, name, *args], {})
        if result.exit_code:
            raise RuntimeError(f"{self.data.task_id}: {what} failed (exit {result.exit_code}): "
                               f"{((result.stderr or '') + (result.stdout or '')).strip()[-400:]}")
        return result.stdout or ""

    async def _hygiene(self, runtime: Runtime) -> None:
        await self._run(runtime, hygiene.REMOVE_SCRIPT, *self.data.reference_paths,
                        what="removing the image's reference copies")
        await self._run(runtime, hygiene.UNINSTALL_SCRIPT, what="uninstalling local-path packages",
                        name=hygiene.UNINSTALL_PY)

    async def setup(self, runtime: Runtime) -> None:
        """The agent's box: no reference anywhere, an empty workspace with the spec."""
        await self._hygiene(runtime)
        await self._run(runtime, CLEAR_TMP, what="emptying /tmp")
        await self._run(runtime, CLEAR_WORKSPACE, what="emptying /workspace")
        from reliquary_nl2repo import source
        await runtime.write(f"{WORKDIR}/start.md", source.spec(self.data.task_id))

    async def finalize(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(
            ["sh", "-c", FINALIZE_SCRIPT, "nl2repo-capture", ARCHIVE_PATH,
             str(submission.MAX_ARCHIVE_BYTES), WORKDIR], {})
        if result.exit_code == 90:
            raise RuntimeError(f"could not clear the artifact directory: "
                               f"{(result.stderr or '').strip()[-300:]}")
        if result.exit_code:
            trace.info["capture_error"] = CAPTURE_ERRORS.get(result.exit_code,
                                                             f"exit {result.exit_code}")
        if not _on_sandbox(runtime):
            await super().finalize(trace, runtime)

    async def grading_setup(self, runtime: Runtime) -> None:
        """The pristine grading box: the same hygiene, the hidden tests kept."""
        await self._hygiene(runtime)
        await self._run(runtime, HAS_TESTS, what="finding the hidden tests")

    async def _read(self, runtime: Runtime, path: str, cap: int) -> bytes | None:
        try:
            return await runtime.read(path, max_bytes=cap)
        except TimeoutError:
            raise
        except (OSError, vf.SandboxError):
            return None

    @vf.reward(weight=1.0)
    async def pass_rate(self, runtime: Runtime, trace: Trace) -> float | dict[str, float]:
        """min(passed / test_case_count, 1): upstream's per-task metric."""
        if self._graded_elsewhere:
            return {}
        start = time.monotonic()
        try:
            raw = await runtime.read(ARCHIVE_PATH, max_bytes=submission.MAX_ARCHIVE_BYTES)
        except FileNotFoundError:
            raw = None
        except TimeoutError:
            raise
        except (OSError, vf.SandboxError):
            trace.record_metrics({"archive_unreadable": 1.0})
            return 0.0
        if raw is None:
            # Nothing captured: graded as an empty workspace (the tests still run, so a
            # reference left in the image would show here, never silently).
            trace.record_metrics({"empty_workspace": 1.0})
            raw = submission.empty_archive()
        try:
            clean, kept = submission.sanitize(raw, self.data.test_files, self.data.own_modules)
        except submission.ArchiveRefused as exc:
            trace.record_metrics({"archive_refused": 1.0})
            trace.info["nl2repo"] = {"archive_refused": str(exc)[:300]}
            return 0.0
        await runtime.write(CLEAN_ARCHIVE, clean)
        extracted = await runtime.run(["sh", "-c", EXTRACT_SCRIPT], {})
        if extracted.exit_code:
            raise RuntimeError(f"extracting the workspace failed (exit {extracted.exit_code}): "
                               f"{(extracted.stderr or '').strip()[-300:]}")
        tests_began = time.monotonic()
        script = test_script(self.data.test_commands)
        await runtime.run(["bash", "-c", script], {})
        test_seconds = time.monotonic() - tests_began

        passed = reported = summary = 0
        report_errors: list[str] = []
        exits: list[str] = []
        tails: list[str] = []
        for index, command in enumerate(self.data.test_commands):
            _, is_pytest = plan_command(command, index)
            code = await self._read(runtime, EXIT.format(index), 64)
            exits.append((code or b"").decode("utf-8", "replace").strip() or "none")
            log = await self._read(runtime, LOG.format(index), 1024 * 1024)
            text = (log or b"").decode("utf-8", "replace")
            if text:
                tails.append(f"$ {command[:200]}\n{text[-MAX_LOG_TAIL:]}")
            if not is_pytest:
                continue
            summary += summary_passed(text)
            report = await self._read(runtime, JUNIT.format(index), submission.MAX_REPORT_BYTES)
            if report is None:
                report_errors.append(f"command {index}: no report")
                continue
            try:
                ok, total = submission.count_passed(report)
            except (ValueError, ParseError, RecursionError) as exc:
                report_errors.append(f"command {index}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            passed += ok
            reported += total
        count = self.data.test_case_count
        reward = min(passed, count) / count
        trace.record_metrics({
            "passed": float(passed), "reported": float(reported),
            "summary_passed": float(summary), "test_case_count": float(count),
            "report_errors": float(len(report_errors)),
            "files_kept": float(kept["kept"]),
            "files_dropped": float(sum(kept["dropped"].values())),
            "test_seconds": float(test_seconds),
            "grading_seconds": float(time.monotonic() - start),
        })
        trace.info["nl2repo"] = {"exits": exits, "dropped": kept["dropped"],
                                 "report_errors": report_errors[:20],
                                 "output_tail": "\n".join(tails)[-6000:]}
        return reward


def task_data(task_id: str, idx: int, split: str) -> NL2RepoData:
    """The task's data. Every box: 4 CPUs, 8 GB, 20 GB (Prime's adaptation's defaults:
    stable-baselines3, tsfresh, ydata-profiling and pytorch-grad-cam test heavily), no
    network."""
    from reliquary_nl2repo import source

    return NL2RepoData(
        idx=idx,
        name=task_id,
        prompt=INSTRUCTION,
        image=source.image(task_id),
        workdir=WORKDIR,
        network_allow=[],
        # Agent: upstream bounds nothing (OpenHands with no iteration limit); 2 h gives a
        # whole-repository build room, the order's turn and token budget binds first.
        timeout=vf.TaskTimeout(setup=300.0, agent=7200.0, finalize=300.0, scoring=810.0),
        resources=vf.TaskResources(cpu=4.0, memory=8.0, disk=20.0),
        artifacts=[vf.Artifact(source=ARCHIVE_PATH, required=False)],
        tags=["python"],
        task_id=task_id,
        spec_sha256=source.spec_sha256(task_id),
        test_commands=source.test_commands(task_id),
        test_files=source.test_files(task_id),
        test_case_count=source.test_case_count(task_id),
        reference_paths=list(hygiene.REFERENCE_PATHS.get(task_id, ())),
        own_modules=list(hygiene.OWN_MODULES.get(task_id, ())),
        split=split,
    )


__all__ = ["ARCHIVE_PATH", "FINALIZE_SCRIPT", "INSTRUCTION", "NL2RepoData", "NL2RepoTask",
           "SANDBOX_RUNTIME", "TEST_BUDGET_S", "WORKDIR", "plan_command", "task_data",
           "test_script"]
