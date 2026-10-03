"""The problem corpus, where it comes from, and why it is half of its source.

Science problems — physics, chemistry, computer science, biology, economics —
from the `science` subset of `PrimeIntellect/INTELLECT-3-RL`, kept only where
the reference answer is one number. Fetched rather than shipped: the upstream
card declares no licence, so this package redistributes the derivation and not
the rows. The parquet is pinned by revision and by digest, and the corpus
derived from it by digest and by size, so a rebuilt corpus that quietly drops
more of itself fails loudly instead of becoming a smaller environment.

Half of the source does not survive, on purpose. A reference such as
`Glucose and galactose` or `\\frac{kq}{r^2}` can only be graded by a judge
model, and this environment grades without one: an answer is a number, read by
`grading.split_number` with the same rules it applies to the policy's box, or
the problem is not here. So are the problems that point at a figure or table
the text does not carry, since nothing the policy can read answers them.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from reliquary_science.grading import split_number

SOURCE_REPOSITORY = "PrimeIntellect/INTELLECT-3-RL"
SOURCE_REVISION = "a9eb9183224cec63fe9a71e7a3a80cff14868624"
SOURCE_FILE = "science/train-00000-of-00001.parquet"
SOURCE_SHA256 = "2dd5a390ee42c0a7829552e83361d19a952b7e480786254ea38eee657fbc78da"
UPSTREAM_ROWS = 29_307

# The digest covers the derived corpus serialised by `corpus_body`, one JSON
# record per line in identity order — what the environment serves, rather than
# the parquet container it was read from.
CORPUS_SHA256 = "64b8527aed2ebc2d991fd6300389873afa425a875a2c60885317bb94123a1383"
NUMERIC_ANSWERS = 14_224
FIGURE_PROBLEMS = 79
VIRTUAL_LENGTH = NUMERIC_ANSWERS - FIGURE_PROBLEMS

SPLITS = ("train", "eval", "qualification")
# Shares rather than thirds, as in the sibling maths corpus: 1,400 problems
# measure a checkpoint well inside a point, and the rest is worth more trained.
_SPLIT_BOUNDS = ((80, "train"), (90, "eval"), (100, "qualification"))
_SPLIT_SALT = "reliquary_science_v1"

# A problem that leans on something it does not include. Written narrowly: the
# phrase "phase diagram" names a body of knowledge, "the figure" names a page.
_FIGURE = re.compile(
    r"\b(?:figure|fig\.|the graph|the plot)\b"
    r"|\b(?:shown|given|see|listed) (?:below|above)\b"
    r"|\b(?:table|diagram|circuit) (?:below|above)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Problem:
    """One problem, the number its answer has to state, and that number's unit."""

    key: str
    problem: str
    answer: float
    unit: str
    domain: str


def _record(problem: Problem) -> dict[str, object]:
    return {
        "answer": problem.answer,
        "domain": problem.domain,
        "id": problem.key,
        "problem": problem.problem,
        "unit": problem.unit,
    }


def corpus_body(corpus: tuple[Problem, ...]) -> bytes:
    """The bytes the pinned digest covers."""
    return "".join(
        json.dumps(_record(problem), sort_keys=True, ensure_ascii=False) + "\n"
        for problem in corpus
    ).encode("utf-8")


def derive(rows: list[dict[str, object]]) -> tuple[tuple[Problem, ...], dict[str, int]]:
    """The corpus a list of upstream rows yields, and what each rule dropped."""
    kept: dict[str, Problem] = {}
    conflicting: set[str] = set()
    numeric = figures = 0
    for row in rows:
        parsed = split_number(str(row["answer"]), relation=False)
        if parsed is None:
            continue
        numeric += 1
        text = str(row["question"]).strip()
        if _FIGURE.search(text):
            figures += 1
            continue
        value, unit = parsed
        info = row.get("info") or {}
        domain = str(info.get("domain") or "unknown") if isinstance(info, dict) else "unknown"
        # Whitespace is the one difference that is not a difference: one
        # problem laid out twice is one problem, served from one index.
        layout_free = " ".join(text.split())
        key = hashlib.sha256(layout_free.encode("utf-8")).hexdigest()[:16]
        problem = Problem(key, text, value, unit, domain)
        if key in kept and kept[key].answer != value:
            conflicting.add(key)
        kept.setdefault(key, problem)
    for key in conflicting:
        # Copies that disagree about the answer cannot be graded honestly:
        # which copy a grader read would decide the reward.
        del kept[key]
    corpus = tuple(sorted(kept.values(), key=lambda problem: problem.key))
    return corpus, {
        "rows": len(rows),
        "numeric": numeric,
        "figures": figures,
        "conflicting": len(conflicting),
        "kept": len(corpus),
    }


def _fetch() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            SOURCE_REPOSITORY,
            SOURCE_FILE,
            revision=SOURCE_REVISION,
            repo_type="dataset",
        )
    )


def read_source(path: Path) -> list[dict[str, object]]:
    """The upstream rows, checked against the pinned parquet digest."""
    import pyarrow.parquet as pq

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"source parquet digest is {digest}, expected {SOURCE_SHA256}")
    rows = pq.read_table(path, columns=["question", "answer", "info"]).to_pylist()
    if len(rows) != UPSTREAM_ROWS:
        raise RuntimeError(f"source holds {len(rows)} rows, expected {UPSTREAM_ROWS}")
    return rows


@lru_cache(maxsize=1)
def load() -> tuple[Problem, ...]:
    """The derived corpus, checked against its pin."""
    corpus, counts = derive(read_source(_fetch()))
    digest = hashlib.sha256(corpus_body(corpus)).hexdigest()
    if digest != CORPUS_SHA256:
        raise RuntimeError(f"corpus digest is {digest}, expected {CORPUS_SHA256}")
    if (counts["numeric"], counts["figures"], len(corpus)) != (
        NUMERIC_ANSWERS, FIGURE_PROBLEMS, VIRTUAL_LENGTH
    ):
        raise RuntimeError(f"corpus counts {counts} differ from the pins")
    return corpus


def split_of(key: str) -> str:
    """Which split a problem belongs to, taken from its identity.

    Hashing the key rather than slicing the file keeps a problem in the split
    it has always been in when the shares are retuned, and keeps the splits
    alike whatever order the corpus arrives in.
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
    "CORPUS_SHA256",
    "FIGURE_PROBLEMS",
    "NUMERIC_ANSWERS",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "SOURCE_SHA256",
    "SPLITS",
    "UPSTREAM_ROWS",
    "VIRTUAL_LENGTH",
    "Problem",
    "corpus_body",
    "derive",
    "load",
    "problems",
    "read_source",
    "split_of",
]
