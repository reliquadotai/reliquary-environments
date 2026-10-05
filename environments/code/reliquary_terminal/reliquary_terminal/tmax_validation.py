"""The pure parts of the box phase (docs/tmax.md, section 9).

`scripts/tmax_validate.py` drives the containers. Everything that decides
something lives here, so it can be tested without one: which files are a
task's protected and hidden inputs, which files the reference changed, how a
mutant of the reference state is made, and how a task's checks turn into
reason codes.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from reliquary_terminal import tmax

# A pytest plugin, written into the probe box, that records every file the
# final test opens and every absolute path it hands to a subprocess or exec.
# It runs inside the box's python3 (3.10), so standard library only.
AUDIT_LOG = "/tmp/reliquary-tmax-audit.json"
AUDIT_PLUGIN = '''
import json, os, re, sys

_SEEN = set()
_ABS = re.compile(r"(?<![\\w.~-])(/[\\w.+@%-]+(?:/[\\w.+@%-]+)*)")


def _add(value):
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    if not isinstance(value, str):
        return
    if os.path.isabs(value):
        _SEEN.add(os.path.normpath(value))
    else:
        for match in _ABS.findall(value):
            _SEEN.add(os.path.normpath(match))


def _hook(event, args):
    try:
        if event == "open":
            path = args[0]
            if isinstance(path, (str, bytes)):
                path = os.fsdecode(path)
                _SEEN.add(os.path.normpath(os.path.abspath(path)))
        elif event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn"):
            for arg in args[:2]:
                if isinstance(arg, (list, tuple)):
                    for item in arg:
                        _add(item)
                else:
                    _add(arg)
        elif event == "os.system":
            _add(args[0])
    except Exception:
        pass


sys.addaudithook(_hook)


def pytest_unconfigure(config):
    with open("''' + AUDIT_LOG + '''", "w") as handle:
        json.dump(sorted(_SEEN), handle)
'''

# Where the probe looks for files a setup made: everything but kernel and
# scratch filesystems, and this package's own staging.
SNAPSHOT_EXCLUDES = ("/proc", "/sys", "/dev", "/run", "/tmp", "/tests", "/logs", "/var/lib/reliquary-tmax")
SNAPSHOT_COMMAND = (
    "find / -xdev \\( "
    + " -o ".join(f"-path {p}" for p in SNAPSHOT_EXCLUDES)
    + " \\) -prune -o -type f -printf '%p\\0%s\\0%T@\\0'"
)
_LITERAL = re.compile(r"""['"](/(?:app|home|etc|opt|srv|var|data|usr/local|root|mnt|workspace)(?:/[^'"\s{}]*)?)['"]""")
ANSWER_WORDS = ("truth", "expected", "answer", "oracle", "golden", "secret", "solution", "reference")
# Mutants are made only of text files up to this size.
MAX_MUTATED_BYTES = 1024 * 1024


Snapshot = dict  # path -> (size, mtime)


def parse_snapshot(raw: bytes) -> Snapshot:
    """`SNAPSHOT_COMMAND`'s output: path, size, mtime, NUL-separated."""
    fields = raw.split(b"\0")
    out: Snapshot = {}
    for i in range(0, len(fields) - 2, 3):
        path = fields[i].decode("utf-8", errors="surrogateescape")
        out[path] = (int(fields[i + 1]), fields[i + 2].decode())
    return out


def literal_paths(test_source: str) -> set[str]:
    """The absolute path literals of a test file."""
    return {str(PurePosixPath(m)) for m in _LITERAL.findall(test_source)}


def protected_inputs(candidates: set[str], base: Snapshot, after_setup: Snapshot, after_solution: Snapshot) -> list[str]:
    """The files the final test reads that setup made (absent from the
    base image, or changed from it) and that the reference left untouched."""
    out = []
    for path in candidates:
        if path not in after_setup:
            continue
        if base.get(path) == after_setup[path]:
            continue  # the base image's own file, not the task's
        if after_solution.get(path) != after_setup[path]:
            continue  # the reference changed it: an output, not an input
        if any(path == p or path.startswith(p + "/") for p in SNAPSHOT_EXCLUDES):
            continue
        if "/__pycache__/" in path or path.endswith(".pyc"):
            continue  # bytecode the interpreter wrote for the test run itself
        out.append(path)
    return sorted(out)


def hidden_inputs(protected: list[str], instruction: str) -> list[str]:
    """The protected inputs the instruction names neither by path nor by
    file name: the agent is not told about them, so it does not see them."""
    return [p for p in protected if p not in instruction and PurePosixPath(p).name not in instruction]


def looks_like_answer(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    return any(word in name for word in ANSWER_WORDS)


def changed_paths(after_setup: Snapshot, after_solution: Snapshot, roots=tmax.ARTIFACT_ROOTS) -> list[str]:
    """The files under the artifact roots the reference created or changed."""
    out = []
    for path, meta in after_solution.items():
        if any(path.startswith(r + "/") for r in roots) and after_setup.get(path) != meta:
            out.append(path)
    return sorted(out)


def _perturb(content: bytes) -> bytes | None:
    """Every digit run's last digit changed, and the last non-empty line
    dropped; None for binary or oversized content."""
    if len(content) > MAX_MUTATED_BYTES or b"\0" in content:
        return None
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None
    text = re.sub(r"\d+", lambda m: m.group(0)[:-1] + str((int(m.group(0)[-1]) + 1) % 10), text)
    lines = text.splitlines(keepends=True)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            del lines[i]
            break
    return "".join(lines).encode()


def mutate_archive(archive: bytes | None, changed: list[str], mode: str) -> tuple[bytes | None, int]:
    """An artifact archive (one root, as verifiers collects it) with every
    changed file mutated: "zero" truncates it, "perturb" applies `_perturb`
    to text files, "control" rewrites the archive unchanged (the check that
    this path itself grades the reference at 1). Returns the archive and how
    many files were mutated."""
    if archive is None:
        return None, 0
    wanted = {p.lstrip("/") for p in changed}
    out = io.BytesIO()
    count = 0
    with tarfile.open(fileobj=io.BytesIO(archive)) as src, tarfile.open(fileobj=out, mode="w") as dst:
        for member in src.getmembers():
            data = src.extractfile(member).read() if member.isfile() else None
            if data is not None and member.name in wanted:
                if mode == "control":
                    mutated = None
                else:
                    mutated = b"" if mode == "zero" else _perturb(data)
                if mutated is not None and mutated != data:
                    data = mutated
                    count += 1
            if data is not None:
                dst.addfile(_resized(member, len(data)), io.BytesIO(data))
            else:
                dst.addfile(member)
    return out.getvalue(), count


def _resized(member: tarfile.TarInfo, size: int) -> tarfile.TarInfo:
    import copy

    clone = copy.copy(member)
    clone.size = size
    return clone


def mutate_artifacts(artifacts: dict, changed: list[str], mode: str) -> tuple[dict, int]:
    out, total = {}, 0
    for root, archive in artifacts.items():
        out[root], count = mutate_archive(archive, changed, mode)
        total += count
    return out, total


@dataclass
class Checks:
    """What the box phase measured for one task; `verdict` turns it into
    reason codes. `None` means not reached."""

    setup_ok: bool | None = None
    initial_ok: bool | None = None
    deterministic: bool | None = None
    noop_rewards: list[float] = field(default_factory=list)
    solution_rewards: list[float] = field(default_factory=list)
    hidden_retry_rewards: list[float] = field(default_factory=list)
    artifact_over_cap: bool = False
    mutation_rewards: dict[str, float] = field(default_factory=dict)
    # The unmutated archive, rewritten and graded the same way: must be 1,
    # or a 0 from a mutant proves nothing.
    mutation_control: float | None = None
    hidden: list[str] = field(default_factory=list)


def verdict(checks: Checks) -> tuple[list[str], list[str]]:
    """(reason codes, hidden inputs to ship). An empty reason list keeps
    the task. Checks stop at the first failure, so one reason at most from
    the box phase, except where noted."""
    if checks.setup_ok is False:
        return ["setup_failed"], []
    if checks.initial_ok is False:
        return ["initial_state_fails"], []
    if checks.deterministic is False:
        return ["setup_nondeterministic"], []
    if any(r > 0 for r in checks.noop_rewards):
        return ["noop_passes"], []
    if checks.artifact_over_cap:
        return ["artifact_cap"], []
    hidden = list(checks.hidden)
    rewards = checks.solution_rewards
    if hidden and rewards and all(r == 0 for r in rewards) and checks.hidden_retry_rewards:
        # The reference failed with the hidden inputs deleted. Visible but
        # protected, does it pass? Then the inputs were genuine inputs --
        # unless one is named like an answer, which the reference read.
        if all(r == 1 for r in checks.hidden_retry_rewards):
            if any(looks_like_answer(p) for p in hidden):
                return ["answer_needed_by_reference"], []
            hidden, rewards = [], checks.hidden_retry_rewards
        else:
            rewards = checks.hidden_retry_rewards
    if not rewards or all(r == 0 for r in rewards):
        return ["solution_fails"], []
    if any(r != 1 for r in rewards):
        return ["solution_unstable"], []
    if checks.mutation_rewards and checks.mutation_control is not None and checks.mutation_control != 1:
        return ["mutation_inconclusive"], []
    if any(r > 0 for r in checks.mutation_rewards.values()):
        return ["mutation_passes"], []
    return [], hidden


def result_record(task_id: str, base_image: str, run: int, protected: list[str], checks: Checks, details: dict) -> dict:
    reasons, hidden = verdict(checks)
    return {
        "task_id": task_id,
        "base_image": base_image,
        "reasons": reasons,
        "run": run,
        "protected": protected,
        "hidden": hidden,
        "checks": json.loads(json.dumps(checks.__dict__)),
        "details": details,
    }
