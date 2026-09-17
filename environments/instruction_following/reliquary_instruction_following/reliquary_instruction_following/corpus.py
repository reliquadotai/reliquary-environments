"""The prompt corpus, its splits, and the verifiers it refuses to carry.

37,196 writing requests, each carrying one to three formal constraints —
"the 3rd of 4 paragraphs must start with the word crash", "your last word
must be contest", "do not use the words inevitable or part". 36,617 of them
are gradable; `_is_stable` says what the rest ask for. Derived from
`nvidia/Nemotron-Cascade-2-RL-data`, split `IF-RL`, licensed ODC-BY.

The file ships inside the wheel rather than being fetched. The corpus is the
task: a row that arrived differently, or not at all, is a different
environment, and a participant must not be able to discover that at grading
time.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.resources
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

CORPUS_FILE = "data/prompts.jsonl.gz"
# The digest covers the decompressed body, not the gzip container: the same
# corpus recompressed at another level is the same corpus.
CORPUS_SHA256 = "d685f4a7f435931538bc800d4b34a977c445949fc15fdc1c6b8b65bd9ac08093"
CORPUS_ROWS = 37196

# Four verifiers this environment will not grade, and the reason each is out.
# `langdetect` samples, so the three that consult it can answer differently in
# two processes about the same text, and the reward has to be a fact rather
# than a draw. `number_sentences` loads the Punkt pickle, which means a
# download at grading time. The corpus ships without all four, and `load`
# refuses a row that names one rather than taking that on trust.
EXCLUDED_INSTRUCTION_IDS = frozenset(
    {
        "change_case:english_capital",
        "change_case:english_lowercase",
        "language:response_language",
        "length_constraints:number_sentences",
    }
)

# Rows dropped by `_is_stable`, pinned so that a corpus which starts rejecting
# more of itself fails loudly instead of quietly becoming a smaller environment.
UNSTABLE_ROWS = 579

SPLITS = ("train", "eval", "qualification")
# Shares rather than thirds. The corpus is finite, and a few thousand prompts
# already measure a checkpoint to well inside a point; every row beyond that is
# worth more as training signal than as a wider evaluation.
_SPLIT_BOUNDS = ((80, "train"), (90, "eval"), (100, "qualification"))
_SPLIT_SALT = "reliquary_instruction_following_v1"

CONSTRAINT_COUNTS = (1, 2, 3)


@dataclass(frozen=True, slots=True)
class Constraint:
    """One formal requirement: an IFEvalG verifier id and its arguments."""

    instruction_id: str
    kwargs: dict[str, Any]

    @property
    def family(self) -> str:
        return self.instruction_id.split(":", 1)[0]


@dataclass(frozen=True, slots=True)
class Row:
    """One writing request and everything its answer will be measured against."""

    key: str
    prompt: str
    constraints: tuple[Constraint, ...]

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({constraint.family for constraint in self.constraints}))


def _is_stable(constraint: Constraint) -> bool:
    """Reject the one constraint shape whose checker rewrites itself.

    `ParagraphFirstWordCheck` redraws `nth_paragraph` at random whenever a row
    asks for a paragraph past the number of paragraphs it also demands, and 579
    rows ask for the 4th of 3. Two participants would then measure two
    different paragraphs, and neither is the one the prompt names — which no
    answer could satisfy anyway.
    """
    if constraint.instruction_id != "length_constraints:nth_paragraph_first_word":
        return True
    nth = constraint.kwargs.get("nth_paragraph") or 0
    total = constraint.kwargs.get("num_paragraphs") or 0
    return 0 < nth <= total


def _parse(body: bytes) -> tuple[Row, ...]:
    """Every gradable row in `body`, in file order."""
    rows: list[Row] = []
    # Split on "\n" and nothing else. 106 of these prompts contain a vertical
    # tab, form feed, NEL or line separator, which JSON keeps raw inside a
    # string and `str.splitlines` would cut the record in half.
    for line in body.decode("utf-8").split("\n"):
        if not line:
            continue
        record = json.loads(line)
        constraints = tuple(
            Constraint(instruction_id, dict(kwargs or {}))
            for instruction_id, kwargs in zip(
                record["instruction_id_list"], record["kwargs"], strict=True
            )
        )
        named = {constraint.instruction_id for constraint in constraints}
        excluded = sorted(named & EXCLUDED_INSTRUCTION_IDS)
        if excluded:
            raise ValueError(
                f"corpus row {record['id']} names excluded verifiers: {excluded}"
            )
        if all(_is_stable(constraint) for constraint in constraints):
            # The corpus numbers its rows; identity here is a name, not a count.
            rows.append(Row(str(record["id"]), record["prompt"], constraints))
    return tuple(rows)


@lru_cache(maxsize=1)
def load() -> tuple[Row, ...]:
    """The packaged corpus, checked against its pin."""
    body = gzip.decompress(
        importlib.resources.files(__package__).joinpath(CORPUS_FILE).read_bytes()
    )
    digest = hashlib.sha256(body).hexdigest()
    if digest != CORPUS_SHA256:
        raise RuntimeError(f"corpus digest is {digest}, expected {CORPUS_SHA256}")

    rows = _parse(body)
    if len(rows) != CORPUS_ROWS - UNSTABLE_ROWS:
        raise RuntimeError(
            f"corpus holds {len(rows)} gradable rows, expected "
            f"{CORPUS_ROWS - UNSTABLE_ROWS}"
        )
    return rows


def split_of(key: str) -> str:
    """Which split a row belongs to, taken from its identity.

    Hashing the row key rather than slicing the file is what keeps the mix of
    constraint families the same in all three splits whatever order the corpus
    arrives in — `test_splits_are_disjoint_and_unbiased` measures that rather
    than assuming it — and it keeps a row in the split it has always been in
    when the shares are retuned or a band is selected.
    """
    digest = hashlib.sha256(f"{_SPLIT_SALT}:{key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    for bound, split in _SPLIT_BOUNDS:
        if bucket < bound:
            return split
    raise AssertionError("split bounds must end at 100")


@lru_cache(maxsize=None)
def rows(
    split: str = "train", constraint_counts: tuple[int, ...] | None = None
) -> tuple[Row, ...]:
    """The rows of one split, optionally narrowed to a difficulty band."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    wanted = None if constraint_counts is None else frozenset(constraint_counts)
    if wanted is not None and not wanted <= set(CONSTRAINT_COUNTS):
        raise ValueError(f"constraint counts must be drawn from {CONSTRAINT_COUNTS}")
    return tuple(
        row
        for row in load()
        if split_of(row.key) == split
        and (wanted is None or len(row.constraints) in wanted)
    )


__all__ = [
    "CONSTRAINT_COUNTS",
    "CORPUS_ROWS",
    "CORPUS_SHA256",
    "EXCLUDED_INSTRUCTION_IDS",
    "SPLITS",
    "UNSTABLE_ROWS",
    "Constraint",
    "Row",
    "load",
    "rows",
    "split_of",
]
