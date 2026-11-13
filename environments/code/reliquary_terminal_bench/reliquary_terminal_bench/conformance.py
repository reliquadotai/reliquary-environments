"""reliquary-terminal-bench's reference solutions and conformance cases, as plain data,
for a signed-episode sandbox's reference sweep (`reference_calls(split, index)`) and env
conformance suite (`conformance_cases(split)`). Nothing here imports a sandbox.

A reference does what Harbor's oracle agent does, through the `bash` tool: upload the
task's `solution/` to `/solution` (base64 chunks) and run `solve.sh` from the image's
WORKDIR. Some references run for many minutes: a sweep needs a per-call timeout as long
as the task's agent timeout.

Cases, on the goldens (small images, fast references; they must be in the gold check's
verified list before the cases are trusted): the reference (expect 1); the reference then
the state planted as an artifact by the agent and the work undone (expect 0: finalize
replaces planted artifacts); a no-op (expect 0); the reference with every file it wrote
restored to its old mtime (`touch -d @0`, expect 1: the capture reads ctimes); the
reference then its main output deleted (expect 0: deletions travel).
"""

from __future__ import annotations

import base64
import shlex

from reliquary_terminal_bench import source
from reliquary_terminal_bench.task import ARTIFACTS, STATE
from reliquary_terminal_bench.taskset import split_ids

CHUNK = 60_000
"""base64 characters per shell call."""
STAGE = "/tmp/reliquary-reference.b64"
GOLDENS: dict[str, str] = {
    # task id -> a file its reference writes and its tests require
    "fix-git": "/app/personal-site/_includes/about.md",
    "regex-log": "/app/regex.txt",
}


def _bash(command: str) -> list:
    return ["bash", {"command": command}]


def upload_calls(content: bytes, path: str) -> list:
    encoded = base64.b64encode(content).decode()
    calls = [_bash(f"rm -f {STAGE}")]
    calls += [_bash(f"printf %s {encoded[i:i + CHUNK]} >> {STAGE}")
              for i in range(0, len(encoded), CHUNK)]
    calls.append(_bash(f"base64 -d {STAGE} > {shlex.quote(path)} && rm -f {STAGE}"))
    return calls


def solve_calls(task_id: str) -> list:
    calls = upload_calls(source.solution_archive(task_id), "/tmp/reliquary-solution.tgz")
    calls.append(_bash(
        "rm -rf /solution && mkdir -p /solution && tar --no-same-owner -xzf "
        "/tmp/reliquary-solution.tgz -C /solution && rm -f /tmp/reliquary-solution.tgz && "
        f"cd {shlex.quote(source.workdir(task_id))} && bash /solution/solve.sh"))
    return calls


def reference_calls(split: str, index: int) -> list | None:
    """The calls that solve task `index` of `split` (every task ships `solve.sh`)."""
    ids = split_ids(split)
    if not 0 <= index < len(ids):
        return None
    return solve_calls(ids[index])


def conformance_cases(split: str) -> list[dict]:
    ids = split_ids(split)
    cases: list[dict] = []
    for task_id, output in GOLDENS.items():
        if task_id not in ids:
            continue
        index = ids.index(task_id)
        solve = solve_calls(task_id)
        cases += [
            {"name": f"{task_id}-reference", "index": index, "expect": 1.0, "calls": solve},
            {"name": f"{task_id}-noop", "index": index, "expect": 0.0,
             "calls": [_bash("true")]},
            {"name": f"{task_id}-old-mtimes", "index": index, "expect": 1.0,
             "calls": solve + [_bash(
                 "find / -xdev -newer /var/lib/reliquary-tb/work-start -type f "
                 "-not -path '/proc/*' -exec touch -h -d @0 {} + 2>/dev/null; true")]},
            {"name": f"{task_id}-output-deleted", "index": index, "expect": 0.0,
             "calls": solve + [_bash(f"rm -f {shlex.quote(output)}")]},
            {"name": f"{task_id}-planted-state", "index": index, "expect": 0.0,
             "calls": [_bash(f"mkdir -p {ARTIFACTS} && : > {STATE}")]},
        ]
    return cases


__all__ = ["GOLDENS", "conformance_cases", "reference_calls", "solve_calls", "upload_calls"]
