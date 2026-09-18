"""The problem corpus, its splits, and why it is a hundredth of its source.

17,171 competition maths problems — AIME and AMC papers, olympiad-style
problems, Chinese competition and textbook exercises — each with an answer
that upstream transformed into a single integer. Derived from `BytedTsinghua-SIA/DAPO-Math-17k`, licensed
Apache 2.0, and shipped inside the wheel rather than fetched: the corpus is
the task, and a problem that arrived differently is a different environment.

The published file holds 1,791,700 rows for 17,188 distinct problems, each
repeated a hundred times or more — a shape that suits a trainer reading the
file end to end and ruins an environment addressed by index. Indexing the raw
rows would serve one problem under a hundred indices, which a cooldown keyed
on the index lets a participant farm a hundred times over and a cooldown keyed
on the content answers by retiring ninety-nine indices in every hundred. What
ships here is the deduplicated set, and `VIRTUAL_LENGTH` is its size.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.resources
import json
from dataclasses import dataclass
from functools import lru_cache

CORPUS_FILE = "data/problems.jsonl.gz"
# The digest covers the decompressed body, not the gzip container: the same
# corpus recompressed at another level is the same corpus.
CORPUS_SHA256 = "ca4d0d6f2014aa932324938df5229b0e11a1f94d3a4bd9bfc4295ce4306bfa8e"

# What the source holds, and what survived the build. `scripts/build_corpus.py`
# is the derivation; these are pinned so that a rebuilt corpus which quietly
# drops more of itself fails loudly instead of becoming a smaller environment.
UPSTREAM_ROWS = 1_791_700
DISTINCT_PROBLEMS = 17_188
# Twelve problems whose copies disagree about the answer, and five whose answer
# is larger than a double carries exactly. Neither can be graded honestly.
AMBIGUOUS_PROBLEMS = 12
OVERSIZED_PROBLEMS = 5
VIRTUAL_LENGTH = DISTINCT_PROBLEMS - AMBIGUOUS_PROBLEMS - OVERSIZED_PROBLEMS

# The largest integer a double carries without loss, and the bound the build
# applies to every answer. Both sides of the comparison stay exact below it.
MAX_ANSWER_MAGNITUDE = (1 << 53) - 1

SPLITS = ("train", "eval", "qualification")
# Shares rather than thirds. 1,700 problems already measure a checkpoint to
# well inside a point, and every problem beyond that is worth more as training
# signal than as a wider evaluation.
_SPLIT_BOUNDS = ((80, "train"), (90, "eval"), (100, "qualification"))
_SPLIT_SALT = "reliquary_dapo_math_v1"


@dataclass(frozen=True, slots=True)
class Problem:
    """One problem and the integer its answer has to equal."""

    key: str
    problem: str
    answer: int


def _parse(body: bytes) -> tuple[Problem, ...]:
    """Every problem in `body`, in file order."""
    problems: list[Problem] = []
    for line in body.decode("utf-8").split("\n"):
        if not line:
            continue
        record = json.loads(line)
        answer = record["answer"]
        if not isinstance(answer, int) or isinstance(answer, bool):
            raise ValueError(f"problem {record['id']} has a non-integer answer")
        if abs(answer) > MAX_ANSWER_MAGNITUDE:
            raise ValueError(f"problem {record['id']} has an answer beyond the bound")
        problems.append(Problem(record["id"], record["problem"], answer))
    return tuple(problems)


@lru_cache(maxsize=1)
def load() -> tuple[Problem, ...]:
    """The packaged corpus, checked against its pin."""
    body = gzip.decompress(
        importlib.resources.files(__package__).joinpath(CORPUS_FILE).read_bytes()
    )
    digest = hashlib.sha256(body).hexdigest()
    if digest != CORPUS_SHA256:
        raise RuntimeError(f"corpus digest is {digest}, expected {CORPUS_SHA256}")

    problems = _parse(body)
    if len(problems) != VIRTUAL_LENGTH:
        raise RuntimeError(
            f"corpus holds {len(problems)} problems, expected {VIRTUAL_LENGTH}"
        )
    return problems


def split_of(key: str) -> str:
    """Which split a problem belongs to, taken from its identity.

    Hashing the problem's key rather than slicing the file keeps a problem in
    the split it has always been in when the shares are retuned, and keeps the
    three splits alike in difficulty whatever order the corpus arrives in —
    `test_splits_are_disjoint_and_alike` measures that rather than assuming it.
    """
    digest = hashlib.sha256(f"{_SPLIT_SALT}:{key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    for bound, split in _SPLIT_BOUNDS:
        if bucket < bound:
            return split
    raise AssertionError("split bounds must end at 100")


@lru_cache(maxsize=None)
def problems(split: str = "train") -> tuple[Problem, ...]:
    """The problems of one split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return tuple(problem for problem in load() if split_of(problem.key) == split)


__all__ = [
    "AMBIGUOUS_PROBLEMS",
    "CORPUS_FILE",
    "CORPUS_SHA256",
    "DISTINCT_PROBLEMS",
    "MAX_ANSWER_MAGNITUDE",
    "OVERSIZED_PROBLEMS",
    "SPLITS",
    "UPSTREAM_ROWS",
    "VIRTUAL_LENGTH",
    "Problem",
    "load",
    "problems",
    "split_of",
]
