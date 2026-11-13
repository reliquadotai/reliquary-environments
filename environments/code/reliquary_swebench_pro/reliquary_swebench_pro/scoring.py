"""The verdict, computed in the scoring process from the run script's logs.

Upstream's `test.sh` runs the candidate's code as root in the grading box, then the task's
`parser.py` and the verdict there, and writes `/logs/verifier/reward.txt`. Neither is
trusted: once the candidate's code has run, anything it left running could rewrite them.
The scoring process reads the raw logs `test.sh` keeps (`run-script-stdout.txt`,
`run-script-stderr.txt`) with size caps, runs the task's own vendored `parser.py` on them
here (loaded from its pinned bytes, never from the box), so the `output.json` it writes is
the box's own, and applies upstream's rule: 1 iff every fail-to-pass and pass-to-pass test
is reported PASSED there.

What the candidate's code can still do while its tests run (print output that parses as
passing) is the residual every repository-test grader has: the logs are the evidence.
Test files it changed are restored by `test.sh` before the run.

Also here, decided on the patch before anything runs (`patch_refusal`): a patch with a
symlink or a submodule, or one touching a dependency or build-cache directory (where a
planted file is loaded by the test toolchain rather than by the task's code), is graded 0
and never applied.
"""

from __future__ import annotations

import ast
import functools
import json
import re
import sys
import tempfile
import types
from pathlib import Path, PurePosixPath

from reliquary_swebench_pro import source

VERIFIER_DIR = "/logs/verifier"
STDOUT_LOG = f"{VERIFIER_DIR}/run-script-stdout.txt"
STDERR_LOG = f"{VERIFIER_DIR}/run-script-stderr.txt"
MAX_LOG_BYTES = 32 * 1024 * 1024

# Directories whose files a toolchain loads on its own (interpreters, test runners,
# build caches): a patch has no business there, and a file planted there runs before or
# instead of the task's code. Tracked or not, the path is refused.
REFUSED_DIRS = frozenset({
    ".git", ".gocache", "node_modules", ".venv", "venv", "site-packages", "dist-packages",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".cargo", ".rustup", ".npm", ".pnpm-store",
    ".yarn", ".tox", ".nox",
})
_SPECIAL_MODE = re.compile(r"^(?:new file mode|new mode|old mode|deleted file mode|index [0-9a-f]+\.\.[0-9a-f]+) (120000|160000)\b",
                           re.MULTILINE)
_DIFF_GIT = re.compile(r'^diff --git "?a/(.*?)"? "?b/(.*?)"?$')


def patch_refusal(patch: bytes) -> str | None:
    """Why `patch` is graded 0 without being applied, or None."""
    text = patch.decode("utf-8", "replace")
    if _SPECIAL_MODE.search(text):
        return "symlink_or_submodule"
    for path in sorted(set(_patch_paths(text))):
        parts = PurePosixPath(path).parts
        if path.startswith("/") or ".." in parts:
            return f"path_outside_repository: {path[:200]}"
        if any(p in REFUSED_DIRS for p in parts):
            return f"refused_directory: {path[:200]}"
    return None


def _patch_paths(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        match = _DIFF_GIT.match(line)
        if match:
            out += [match.group(1), match.group(2)]
        elif line.startswith(("+++ ", "--- ")):
            # git's a/ b/ prefixes, or a plain diff's paths (`patch -p1` strips one level)
            name = line[4:].split("\t", 1)[0].strip()
            if name != "/dev/null":
                out.append(name.split("/", 1)[1] if "/" in name else name)
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            out.append(line.split(" ", 2)[2])
    return [p for p in out if p and p != "/dev/null"]


# --- the task's parser -------------------------------------------------------

@functools.lru_cache(maxsize=64)
def parser_module(task_id: str) -> types.ModuleType:
    """The task's `tests/parser.py`, loaded from its pinned bytes. Its `main` is never
    called; only `parse_test_output(stdout, stderr)` is."""
    name = f"_reliquary_swebench_pro_parser_{abs(hash(task_id)):x}"
    module = types.ModuleType(name)
    module.__file__ = f"<{task_id}/tests/parser.py>"
    sys.modules[name] = module  # dataclasses resolve the module by name
    try:
        exec(compile(source.read(task_id, "tests/parser.py"), module.__file__, "exec"),
             module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def parse(task_id: str, stdout: bytes, stderr: bytes) -> list[tuple[str, str]]:
    """(name, status) per entry of the `output.json` upstream's `parser.py` writes: its
    `parse_test_output`, then its own `export_to_json` (some tasks' keep the last status
    of a repeated name), into a private temporary file."""
    module = parser_module(task_id)
    results = module.parse_test_output(stdout.decode("utf-8", "replace"),
                                       stderr.decode("utf-8", "replace"))
    with tempfile.TemporaryDirectory(prefix="swepro-parse-") as tmp:
        output = Path(tmp) / "output.json"
        module.export_to_json(results, output)
        document = json.loads(output.read_text())
    return [(str(t.get("name", "")), str(t.get("status", ""))) for t in document.get("tests", [])]


def test_names(raw) -> set[str]:
    """Upstream's `parse_tests`: a JSON list, a Python literal list, or a list."""
    if isinstance(raw, str):
        try:
            return set(json.loads(raw))
        except json.JSONDecodeError:
            pass
        try:
            return set(ast.literal_eval(raw))
        except (ValueError, SyntaxError):
            return set()
    return set(raw) if isinstance(raw, list) else set()


def assess(task_id: str, stdout: bytes | None, stderr: bytes | None) -> dict:
    """Upstream's verdict on the logs (None: missing, the run never got there)."""
    config = source.config(task_id)
    f2p, p2p = test_names(config.get("fail_to_pass", [])), test_names(config.get("pass_to_pass", []))
    if stdout is None or stderr is None:
        return {"reward": 0.0, "evaluation_valid": 0.0, "f2p_total": float(len(f2p)),
                "p2p_total": float(len(p2p)), "f2p_passed": 0.0, "p2p_passed": 0.0,
                "missing": sorted(f2p | p2p), "failed": []}
    results = parse(task_id, stdout, stderr)
    passed = {name for name, status in results if status == "PASSED"}
    not_passed = {name for name, status in results if status in ("FAILED", "ERROR")}
    required = f2p | p2p
    return {
        "reward": float(required <= passed),
        "evaluation_valid": 1.0,
        "f2p_total": float(len(f2p)), "f2p_passed": float(len(f2p & passed)),
        "p2p_total": float(len(p2p)), "p2p_passed": float(len(p2p & passed)),
        # Upstream ignores a FAILED entry for a test also reported PASSED (issue #125):
        # counted, never scored.
        "required_also_failed": float(len(required & passed & not_passed)),
        "results": float(len(results)),
        "missing": sorted(required - passed - not_passed),
        "failed": sorted((required & not_passed) - passed),
    }


METRIC_KEYS = ("evaluation_valid", "f2p_total", "f2p_passed", "p2p_total", "p2p_passed",
               "required_also_failed", "results")


def summary(assessment: dict, limit: int = 20) -> dict:
    return {"failed": [n[:300] for n in assessment.get("failed", [])[:limit]],
            "missing": [n[:300] for n in assessment.get("missing", [])[:2 * limit]]}


__all__ = ["MAX_LOG_BYTES", "METRIC_KEYS", "REFUSED_DIRS", "STDERR_LOG", "STDOUT_LOG",
           "assess", "parse", "parser_module", "patch_refusal", "summary", "test_names"]
