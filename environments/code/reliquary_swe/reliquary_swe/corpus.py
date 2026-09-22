"""SWE-bench rows, and nothing that needs a container.

Deliberately free of Docker and of verifiers runtimes: corpus questions — how
many instances, which repositories, what a task id looks like — must be
answerable on a machine that cannot host images.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass

from datasets import load_dataset

SPLITS = ("eval",)

# Pinned by revision, as every other environment here pins its data. An
# unpinned corpus makes two runs incomparable for a reason that never shows up
# in the metrics.
_SOURCES = {
    "eval": (
        "princeton-nlp/SWE-bench_Verified",
        "test",
        "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
    ),
}


@dataclass(frozen=True, slots=True)
class SweRow:
    instance_id: str
    repo: str
    base_commit: str
    version: str
    problem_statement: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    gold_patch: str
    test_patch: str


def _tests(raw: object) -> tuple[str, ...]:
    """SWE-bench stores the two test lists as JSON-encoded strings."""
    if isinstance(raw, str):
        return tuple(json.loads(raw))
    return tuple(raw or ())


@functools.lru_cache(maxsize=None)
def load_rows(split: str = "eval") -> tuple[SweRow, ...]:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r}; expected one of {SPLITS}")
    name, hf_split, revision = _SOURCES[split]
    dataset = load_dataset(name, split=hf_split, revision=revision or None)
    return tuple(
        SweRow(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            version=row["version"],
            problem_statement=row["problem_statement"],
            fail_to_pass=_tests(row["FAIL_TO_PASS"]),
            pass_to_pass=_tests(row["PASS_TO_PASS"]),
            gold_patch=row["patch"],
            test_patch=row["test_patch"],
        )
        for row in dataset
    )
