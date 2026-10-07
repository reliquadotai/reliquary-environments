"""The problem corpus, its splits, and the order the train split is served in.

24,455 olympiad-style problems from the Art of Problem Solving forums, as
collected, solved and answer-checked by NVIDIA in `nvidia/Nemotron-Math-v2`
(revision 8e793210, the AoPS subset, CC BY 4.0). Every problem has a short
closed-form answer — an integer for 58% of them, otherwise a fraction, a surd,
a multiple of pi, an expression in the problem's letters, a tuple, a list of
solutions or a ratio — and no proofs. Shipped inside the wheel rather than
fetched: the corpus is the task, and a problem that arrived differently is a
different environment. NOTICE carries the attribution the licence asks for.

Difficulty comes from upstream's own measurements of gpt-oss-120b, in three
tiers: `T1` where its high-reasoning setting solves under half the time, `T2`
where the high setting is reliable and the low one is not, `T3` the rest.
Every reference is either the forum's own answer confirmed by at least one
high-reasoning solution, or the answer of at least half of them. The file is ordered
`T1`, `T2`, `T3`, each by identity, and the splits keep that order, so index 0
of the train split is its hardest problem.
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
CORPUS_SHA256 = "38a31f2b4ce5d49b4c25b50f18e240eb4738dbd066d1524021f56c6e55857566"

# What the prepared source holds and what survived the build. `scripts/
# build_corpus.py` is the derivation; these are pinned so that a rebuilt corpus
# which quietly drops more of itself fails loudly instead of becoming a smaller
# environment.
SOURCE_ROWS = 24_525
# References `grading.py` does not read, or reads and does not recognise as
# themselves: complex surds (`\\sqrt{-3}`), towers past any bound
# (`2^{2^{126}}`), mixed numbers (`7\\frac{1}{2}`), clock times, floors written
# as brackets, two-part answers. Dropped rather than patched.
UNGRADABLE_REFERENCES = 70
VIRTUAL_LENGTH = SOURCE_ROWS - UNGRADABLE_REFERENCES
TIER_COUNTS = {"T1": 1_501, "T2": 15_037, "T3": 7_917}

SPLITS = ("train", "eval", "qualification")
# This corpus exists to be distilled from, so nearly all of it is train. 754
# eval problems measure a policy to within a couple of points.
_SPLIT_BOUNDS = ((95, "train"), (98, "eval"), (100, "qualification"))
_SPLIT_SALT = "reliquary_hard_math_v1"


@dataclass(frozen=True, slots=True)
class Problem:
    """One problem and the reference its answer has to state."""

    key: str
    problem: str
    answer: str
    tier: str
    source: str


def _parse(body: bytes) -> tuple[Problem, ...]:
    """Every problem in `body`, in file order."""
    problems: list[Problem] = []
    for line in body.decode("utf-8").split("\n"):
        if not line:
            continue
        record = json.loads(line)
        answer = record["answer"]
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"problem {record['id']} has no reference")
        problems.append(
            Problem(record["id"], record["problem"], answer, record["tier"], record["source"])
        )
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
    splits alike in difficulty whatever order the corpus arrives in.
    """
    digest = hashlib.sha256(f"{_SPLIT_SALT}:{key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    for bound, split in _SPLIT_BOUNDS:
        if bucket < bound:
            return split
    raise AssertionError("split bounds must end at 100")


@lru_cache(maxsize=None)
def problems(split: str = "train") -> tuple[Problem, ...]:
    """The problems of one split, in file order: hardest tier first."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return tuple(problem for problem in load() if split_of(problem.key) == split)


__all__ = [
    "CORPUS_FILE",
    "CORPUS_SHA256",
    "SOURCE_ROWS",
    "SPLITS",
    "TIER_COUNTS",
    "UNGRADABLE_REFERENCES",
    "VIRTUAL_LENGTH",
    "Problem",
    "load",
    "problems",
    "split_of",
]
