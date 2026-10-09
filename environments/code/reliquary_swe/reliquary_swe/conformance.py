"""reliquary-swe's conformance cases and reference solutions, as plain data, for a
signed-episode sandbox's env conformance suite (`conformance_cases(split)`) and reference
sweep (`reference_calls(split, index)`). Split names are the taskset's own: `train`
(SWE-smith, the default 20 images), `r2e`, `polyglot`; a sandbox must serve each under
that name. Nothing here imports a sandbox.

A reference applies the task's gold patch through the `bash` tool, in base64 chunks (one
tool call's arguments are bounded). A gold patch the env norm refuses (a binary file, a
symlink or a submodule: 8 R2E rows add a symlink) is no reference: the task has none.
Attacks on a golden task, each expected to grade 0: the gold patch plus a forced path
under `.venv` (refused path), plus a symlink, plus a binary file (refused shapes); the
gold patch planted in the artifact directory with the sources untouched (finalize
replaces it); a root conftest.py that reports every test as passed (restored from the
image by grading).
"""

from __future__ import annotations

import base64
import functools
import shlex
from importlib.resources import files

from reliquary_swe import corpus, grading

CHUNK = 60_000
"""base64 characters per shell call."""
STAGE = "/tmp/reliquary-reference.b64"
GOLDENS = {
    "r2e": (
        "c1bfa7352368b63f3a9b30c02f242408d07a7ab2",  # coveragepy
        "b5ec807edc83c8e7d1d12553d635ebe765e5c614",  # tornado
    ),
    "train": ("oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc",),
    "polyglot": ("format-code-task-002703",),  # Go, the smallest image (0.49 GB)
}
"""The goldens this package's own tests already run on real images."""
FORCING_CONFTEST = (
    "import pytest\n\n\n"
    "@pytest.hookimpl(hookwrapper=True)\n"
    "def pytest_runtest_makereport(item, call):\n"
    "    outcome = yield\n"
    "    outcome.get_result().outcome = 'passed'\n"
)


def _bash(command: str) -> list:
    return ["bash", {"command": command}]


def upload_calls(content: bytes, path: str) -> list:
    encoded = base64.b64encode(content).decode()
    calls = [_bash(f"rm -f {STAGE}")]
    calls += [
        _bash(f"printf %s {encoded[i:i + CHUNK]} >> {STAGE}")
        for i in range(0, len(encoded), CHUNK)
    ]
    calls.append(_bash(f"base64 -d {STAGE} > {shlex.quote(path)} && rm -f {STAGE}"))
    return calls


def apply_calls(patch: bytes, workdir: str) -> list:
    calls = upload_calls(patch, "/tmp/reliquary-reference.diff")
    calls.append(
        _bash(
            f"cd {shlex.quote(workdir)} && git apply /tmp/reliquary-reference.diff"
            " && rm -f /tmp/reliquary-reference.diff"
        )
    )
    return calls


def golden_cases(index: int, gold: bytes, workdir: str, *, python: bool) -> list[dict]:
    q = shlex.quote(workdir)
    solve = apply_calls(gold, workdir)
    cases = [
        {"name": f"reference_{index}", "index": index, "expect": 1.0, "calls": solve},
        {
            "name": f"gold_with_forced_ignored_path_{index}",
            "index": index,
            "expect": 0.0,
            "calls": [
                *solve,
                _bash(
                    f"cd {q} && mkdir -p .venv && echo 'import os' > "
                    ".venv/zz_reliquary.pth && git add -f .venv/zz_reliquary.pth"
                ),
            ],
        },
        {
            "name": f"gold_with_symlink_{index}",
            "index": index,
            "expect": 0.0,
            "calls": [
                *solve,
                _bash(
                    f"cd {q} && ln -s /etc/hostname zz_reliquary_link"
                    " && git add zz_reliquary_link"
                ),
            ],
        },
        {
            "name": f"gold_with_binary_file_{index}",
            "index": index,
            "expect": 0.0,
            "calls": [
                *solve,
                _bash(
                    f"cd {q} && printf '\\000\\001\\002\\377' > zz_reliquary.bin"
                    " && git add zz_reliquary.bin"
                ),
            ],
        },
        {
            "name": f"patch_planted_in_artifacts_{index}",
            "index": index,
            "expect": 0.0,
            "calls": [
                _bash("mkdir -p /logs/artifacts"),
                *upload_calls(gold, "/logs/artifacts/patch.diff"),
            ],
        },
    ]
    if python:
        cases.append(
            {
                "name": f"conftest_forcing_passes_{index}",
                "index": index,
                "expect": 0.0,
                "calls": upload_calls(FORCING_CONFTEST.encode(), f"{workdir}/conftest.py"),
            }
        )
    return cases


@functools.cache
def _ids(split: str) -> tuple[str, ...]:
    if split == "train":
        return tuple(i for _, i in corpus.swesmith_order(corpus.DEFAULT_SWESMITH_IMAGES))
    if split == "r2e":
        return tuple(corpus.r2e_instance_ids())
    return tuple(row.instance_id for row in corpus.load_polyglot_rows(None))


def _row(split: str, index: int) -> corpus.SweRow:
    if split == "train":
        return corpus.swesmith_row_at(corpus.DEFAULT_SWESMITH_IMAGES, index)
    if split == "r2e":
        return corpus.r2e_row_at(index)
    return corpus.load_polyglot_rows(None)[index]


def _gold(split: str, row: corpus.SweRow) -> bytes | None:
    """The row's reference patch, or None: none published (polyglot ships the ones
    recovered from an image under `references/`), or one the env norm refuses."""
    if split != "polyglot":
        gold = row.gold_patch.encode() if row.gold_patch else None
    else:
        shipped = files("reliquary_swe").joinpath("references", f"{row.instance_id}.diff")
        gold = shipped.read_bytes() if shipped.is_file() else None
    if gold is None or grading.patch_shape_violations(gold):
        return None
    return gold


def _golden_index(split: str, key: str) -> int:
    if split == "r2e":
        return _ids("r2e").index(corpus.r2e_row(key).instance_id)
    return _ids(split).index(key)


def reference_calls(split: str, index: int) -> list | None:
    """The calls solving task `index` of `split`, or None when it has no reference."""
    if split not in GOLDENS:
        return None
    row = _row(split, index)
    gold = _gold(split, row)
    return None if gold is None else apply_calls(gold, row.workdir)


def conformance_cases(split: str) -> list[dict]:
    """A reference and the declared attacks on each golden of `split` that has one."""
    if split not in GOLDENS:
        return []
    cases: list[dict] = []
    for key in GOLDENS[split]:
        index = _golden_index(split, key)
        row = _row(split, index)
        gold = _gold(split, row)
        if gold is not None:
            cases += golden_cases(index, gold, row.workdir, python=split != "polyglot")
    return cases


__all__ = ["CHUNK", "GOLDENS", "apply_calls", "conformance_cases", "golden_cases",
           "reference_calls", "upload_calls"]
