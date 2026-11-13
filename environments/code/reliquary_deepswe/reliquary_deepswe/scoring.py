"""The verdict, computed in the scoring process from report bytes read out of the box.

The task's `test.sh` runs the candidate's code as root in the grading box, and its last
step (`grader.py grade`) writes a verdict there. That verdict is never read: once the
candidate's code has run, anything it left running could rewrite it, or the grader that
would rewrite it. After each run the sandbox kills every process of the box; the scoring
process then reads the raw test reports the task declares (`config.json`'s
`grade.reports`, and Go's `run.log`) with size caps, and applies PI-DeepSWE's scoring
rule itself (`verifier.assess` and its Go evidence helpers, the same code the box runs).

What the candidate's code can still do while its tests run (write a report that says
every test passed) is the residual every repository-test grader has: the reports are the
evidence. Test files it changed are restored before the run (`verifier.prepare`).

Also here, decided on the patch before anything runs (`patch_refusal`): a patch with a
symlink or a submodule, or one touching a dependency or build-cache directory (where a
planted file is loaded by the test toolchain rather than by the task's code), is graded
0 and never applied.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath

from reliquary_deepswe import verifier

VERIFIER_DIR = "/logs/verifier"
REPORTS_DIR = f"{VERIFIER_DIR}/reports"
RANK = {"passed": 0, "skipped": 1, "failed": 2}

MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_RUN_LOG_BYTES = 32 * 1024 * 1024

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


# --- patch -------------------------------------------------------------------

def patch_refusal(patch: bytes) -> str | None:
    """Why `patch` is graded 0 without being applied, or None. Paths are read as
    upstream's grader reads them (`patch_paths`), plus rename and copy targets."""
    text = patch.decode("utf-8", "replace")
    if _SPECIAL_MODE.search(text):
        return "symlink_or_submodule"
    paths = set(_patch_paths(text))
    for path in sorted(paths):
        parts = PurePosixPath(path).parts
        if path.startswith("/") or ".." in parts:
            return f"path_outside_repository: {path[:200]}"
        refused = next((p for p in parts if p in REFUSED_DIRS), None)
        if refused is not None or parts[:1] == ("target",):  # Cargo's build directory
            return f"refused_directory: {path[:200]}"
    return None


_DIFF_GIT = re.compile(r'^diff --git "?a/(.*?)"? "?b/(.*?)"?$')


def _patch_paths(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        match = _DIFF_GIT.match(line)
        if match:
            out += [match.group(1), match.group(2)]
        elif line.startswith(("+++ b/", "--- a/")):
            out.append(line[6:])
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            out.append(line.split(" ", 2)[2])
    return [p for p in out if p and p != "/dev/null"]


# --- reports (upstream's parsers, on bytes) ----------------------------------

def norm_status(raw) -> str:
    raw = str(raw or "").strip().lower()
    if raw == "passed":
        return "passed"
    if raw in ("skipped", "pending", "other"):
        return "skipped"
    return "failed"


def add(res: dict, nid: str, st: str, msg: str = "") -> None:
    """Upstream's `add`: worst status wins, the failing entry's message kept."""
    cur = res.get(nid)
    msg = msg or ""
    if cur is None or RANK[st] > RANK[cur[0]]:
        res[nid] = (st, msg if st != "passed" else "")
    elif RANK[st] == RANK[cur[0]] and st != "passed" and not cur[1] and msg:
        res[nid] = (st, msg)


def parse_ctrf(raw: bytes, cfg: dict) -> dict:
    """Upstream's `parse_ctrf` on bytes. Raises ValueError for a report that is not
    CTRF (PI-DeepSWE's `read_reports` check), so it is recorded as a report error."""
    doc = json.loads(raw)
    tests = doc["results"]["tests"]
    if not isinstance(tests, list):
        raise ValueError("report tests must be a list")
    res: dict = {}
    for tc in tests:
        if not isinstance(tc, dict):
            continue
        nm = str(tc.get("name") or "").strip()
        if not nm:
            continue
        su_raw = tc.get("suite")
        if isinstance(su_raw, list) and su_raw:
            su = str(su_raw[0]).strip()
        elif isinstance(su_raw, str):
            su = su_raw.strip()
        else:
            su = ""
        nid = f"{su}.{nm}" if (cfg.get("node_id") == "suite.name" and su) else nm
        st = norm_status(tc.get("status"))
        msg = ""
        if st != "passed":
            msg = str(tc.get("message") or tc.get("trace") or "").strip()
        add(res, nid, st, msg)
    return res


def _junit_status_msg(tc) -> tuple[str, str]:
    st = "passed"
    for ch in tc:
        tag = ch.tag.rsplit("}", 1)[-1]
        if tag in ("failure", "error"):
            parts = [(ch.get("message") or "").strip(), (ch.text or "").strip()]
            return "failed", "\n".join(p for p in parts if p).strip()
        if tag == "skipped":
            st = "skipped"
    return st, ""


def parse_junit(raw: bytes, cfg: dict) -> dict:
    """Upstream's `parse_junit` on bytes; malformed XML raises (a report error)."""
    if b"<!DOCTYPE" in raw[:4096].upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("a JUnit report with a DTD or entities is not read")
    root = ET.fromstring(raw)
    res: dict = {}
    for tc in root.iter("testcase"):
        cn = (tc.attrib.get("classname", "") or "").strip()
        nm = (tc.attrib.get("name", "") or "").strip()
        if not nm:
            continue
        nid = f"{cn}.{nm}" if cn else nm
        st, msg = _junit_status_msg(tc)
        add(res, nid, st, msg)
    return res


PARSERS = {"ctrf": parse_ctrf, "junit": parse_junit}


def report_locations(path: str) -> tuple[str, ...]:
    """Where a declared report may be after `test.sh`: its own path, or `reports/`
    (the frame moves every non-canonical file there when it ends)."""
    if not path.startswith(VERIFIER_DIR + "/"):
        return (path,)
    rel = path[len(VERIFIER_DIR) + 1:]
    return (path, f"{REPORTS_DIR}/{rel}")


def assess_reports(config: dict, language: str, reports: dict[str, bytes | None],
                   run_log: str | None) -> dict:
    """PI-DeepSWE's `grade`, on bytes: `reports` maps each declared report path to its
    bytes (None: missing or unreadable). Returns the assessment (`scores`, `tests`)."""
    grade = config["grade"]
    parse = PARSERS[grade.get("format", "ctrf")]
    parsed: dict = {}
    errors: list[str] = []
    for path in grade["reports"]:
        raw = reports.get(path)
        if raw is None:
            errors.append(f"{path}: missing")
            continue
        try:
            found = parse(raw, grade)
        except (ValueError, KeyError, TypeError, ET.ParseError, RecursionError) as exc:
            errors.append(f"{path}: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        for node_id, (status, message) in found.items():
            add(parsed, node_id, status, message)
    log = run_log or ""
    if language == "go":
        required = {i.strip() for i in config["f2p_node_ids"] + config["p2p_node_ids"]}
        for node_id, (status, message) in verifier.go_test_timeouts(log).items():
            if node_id in required:
                add(parsed, node_id, status, message)
    assessment = verifier.assess(config, parsed, language, (), errors)
    if language == "go":
        verifier.apply_go_package_failures(assessment, verifier.go_build_failures(log),
                                           verifier.go_package_crashes(log))
    return assessment


def retry_eligible(assessment: dict) -> bool:
    """PI-DeepSWE's rule: every fail-to-pass test passed and every remaining failure
    is an explicit timeout."""
    scores = assessment["scores"]
    failed = [r for r in assessment["tests"] if r["status"] != "passed"]
    return bool(scores.get("evaluation_valid") and scores.get("f2p") == 1 and failed
                and all(r["status"] == "failed" and verifier.TIMEOUT.search(r["message"] or "")
                        for r in failed))


METRIC_KEYS = ("f2p", "p2p", "f2p_total", "f2p_passed", "p2p_total", "p2p_passed",
               "missing_total", "family_matches", "partial", "triage")


def summary(assessment: dict, limit: int = 20) -> dict:
    """What the trace keeps of an assessment: failing and missing test ids, report
    errors and Go build failures, truncated."""
    tests = assessment.get("tests") or []
    return {
        "failed": [{"id": r["id"][:300], "message": str(r.get("message", ""))[:300]}
                   for r in tests if r["status"] not in ("passed", "missing")][:limit],
        "missing": [r["id"][:300] for r in tests if r["status"] == "missing"][:2 * limit],
        "report_errors": [str(e)[:300] for e in assessment.get("report_errors") or []][:limit],
        "build_failures": sorted(assessment.get("build_failures") or {})[:limit],
    }


__all__ = ["MAX_REPORT_BYTES", "MAX_RUN_LOG_BYTES", "METRIC_KEYS", "PARSERS", "REFUSED_DIRS",
           "assess_reports", "parse_ctrf", "parse_junit", "patch_refusal", "report_locations",
           "retry_eligible", "summary"]
