"""What a grading box receives as `/tests`: upstream's tests with PI-DeepSWE's overlay.

Upstream ships per task `tests/{test.sh,test.patch,grader.py,config.json,Dockerfile}`;
its verifier image is the task image plus that directory (`tests/Dockerfile`). The
grading box here is a fresh box of the task image, and `/tests` is staged into it:

* `grader.py` is this package's `verifier.py` (PI-DeepSWE's shared verifier, v17), and
  upstream's own grader travels as `upstream_grader.py`, which it imports;
* `metadata.json` is the task's `[metadata]` (the verifier reads the language and id);
* `test.sh` and `config.json` carry PI-DeepSWE's per-task corrections to the runner
  (`kcp-go-multiplexed-kcp-streams`: `TestMuxPriorityPreemption` excluded from execution
  and scoring; `eicrud-keyset-pagination-cursor`: open-file limit 64000 for MongoDB), and
  Go's runner keeps build events in `run.log` (tee before the filter) so build failures
  stay on record. Each edit must match exactly once, else the task is refused;
* `Dockerfile` is left out: nothing is built.

The corrections to fixtures inside the repository (Pest, SQLfmt, Testem, Adaptix,
Vulture, Mashumaro, Meriyah, Obsidian Linter, Ink, Eicrud, Onedump) are applied by
`verifier.prepare` in the box, as PI-DeepSWE applies them.

`effective_config` is the `config.json` the box receives, which the scoring process
also grades against. `tests_sha256` digests the whole staged directory and is part of
the task's data, so the episode's signed task hash binds what grading runs.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

from reliquary_deepswe import source

VERIFIER = Path(__file__).resolve().parent / "verifier.py"
EXECUTABLE = frozenset({"test.sh"})
LEFT_OUT = frozenset({"Dockerfile", "grader.py"})

KCP = "kcp-go-multiplexed-kcp-streams"
KCP_TEST = "TestMuxPriorityPreemption"
EICRUD = "eicrud-keyset-pagination-cursor"

_GO_FILTER = """| grep -v '"Action":"build-'"""
_GO_FILTER_THEN_TEE = re.compile(
    r"""\| grep -v '"Action":"build-'( \\\n[ \t]*| )\| tee -a "\$RUN_LOG" \|"""
)


class OverlayError(ValueError):
    """A correction did not apply to the pinned task: refused rather than half-applied."""


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise OverlayError(f"{what}: expected one occurrence of {old!r}")
    return text.replace(old, new, 1)


def corrected_test_sh(task_id: str, language: str, script: str) -> str:
    if task_id == KCP:
        script = _replace_once(script, '-run "^TestMux"',
                               f'-run "^TestMux" -skip "^{KCP_TEST}$"', task_id)
    if task_id == EICRUD:
        script = _replace_once(script, "set -uo pipefail\n",
                               "set -uo pipefail\nulimit -n 64000 || exit $?\n", task_id)
    if language == "go":
        filters = sum(_GO_FILTER in line for line in script.splitlines()
                      if not line.lstrip().startswith("#"))
        script, count = _GO_FILTER_THEN_TEE.subn(
            lambda m: f"""| tee -a "$RUN_LOG"{m[1]}{_GO_FILTER} |""", script)
        if count != filters:
            raise OverlayError(f"{task_id}: {filters - count} Go build-event filter(s) this "
                               "overlay cannot reorder")
    return script


@functools.cache
def effective_config(task_id: str) -> dict:
    config = json.loads(source.read(task_id, "tests/config.json"))
    if task_id == KCP:
        for key in ("f2p_node_ids", "p2p_node_ids"):
            kept = [i for i in config[key] if not i.strip().endswith(f".{KCP_TEST}")]
            if key == "f2p_node_ids" and len(kept) != len(config[key]) - 1:
                raise OverlayError(f"{task_id}: expected {KCP_TEST} once among the f2p ids")
            config[key] = kept
    return config


@functools.cache
def staged_tests(task_id: str) -> dict[str, bytes]:
    """`{relative path: bytes}` of the `/tests` a grading box receives."""
    metadata = source.task_toml(task_id)["metadata"]
    language = metadata["language"]
    staged: dict[str, bytes] = {}
    for rel in source.files_of(task_id):
        if not rel.startswith("tests/"):
            continue
        name = rel[len("tests/"):]
        if name in LEFT_OUT:
            continue
        staged[name] = source.read(task_id, rel)
    staged["test.sh"] = corrected_test_sh(task_id, language, staged["test.sh"].decode()).encode()
    if task_id == KCP:
        staged["config.json"] = json.dumps(effective_config(task_id), indent=1).encode()
    staged["upstream_grader.py"] = source.read(task_id, "tests/grader.py")
    staged["grader.py"] = VERIFIER.read_bytes()
    staged["metadata.json"] = json.dumps(metadata, sort_keys=True).encode()
    return staged


@functools.cache
def tests_sha256(task_id: str) -> str:
    digest = hashlib.sha256()
    for name, data in sorted(staged_tests(task_id).items()):
        digest.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


@functools.lru_cache(maxsize=4)
def tests_archive(task_id: str) -> bytes:
    """`staged_tests` as a gzip'd tar, byte-identical on every machine (sorted, no
    timestamps or owners)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, data in sorted(staged_tests(task_id).items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name in EXECUTABLE else 0o644
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


__all__ = ["EICRUD", "KCP", "OverlayError", "corrected_test_sh", "effective_config",
           "staged_tests", "tests_archive", "tests_sha256"]
