"""What must never be trained on, and what this version cannot grade.

Held out: LiveCodeBench problems dated from 2024-08-01, the window the held-out
benchmark measures. A source with dates is cut on its date; taco and
primeintellect carry none, so every statement is also compared, by 13-gram
overlap, with every held-out LiveCodeBench statement.

Not gradable in v1: problems that accept several outputs. The comparator has
one expected output per test; such a problem would mark valid answers wrong.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from reliquary_competitive_code.sources.common import SourceRow, normalise

HELD_OUT_FROM = "2024-08-01"
CONTAMINATION_THRESHOLD = 0.10
NGRAM = 13
LCB_REPOSITORY = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"

_MULTI_ANSWER = re.compile(
    r"(print|output)\s+any\b"
    r"|any\s+(of\s+them|valid|correct|one\s+of\s+them)"
    r"|if\s+there\s+are\s+(several|multiple|many)"
    r"|(multiple|several)\s+(possible|correct|valid)\s+(answers|solutions)",
    re.IGNORECASE,
)

_TIE_BREAK = re.compile(
    r"\b(smallest|largest|lexicographically|minimum|maximum|minimal|maximal|first|last|earliest|latest|shortest|longest)\b",
    re.IGNORECASE,
)


def is_after_cutoff(row: SourceRow) -> bool:
    return row.contest_date is not None and row.contest_date[:10] >= HELD_OUT_FROM


def is_multi_answer(row: SourceRow) -> bool:
    # Split on sentence boundaries
    sentences = re.split(r"[.!?\n]+", row.statement)
    for sentence in sentences:
        # A sentence counts if it matches _MULTI_ANSWER and does NOT match _TIE_BREAK
        if _MULTI_ANSWER.search(sentence) and not _TIE_BREAK.search(sentence):
            return True
    return False


def _ngrams(statement: str) -> set[tuple[str, ...]]:
    words = normalise(statement).split()
    return {tuple(words[i : i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


class HeldOutIndex:
    def __init__(self, statements: Iterable[str]) -> None:
        self._grams: set[tuple[str, ...]] = set()
        for statement in statements:
            self._grams |= _ngrams(statement)

    def overlap(self, statement: str) -> float:
        grams = _ngrams(statement)
        if not grams:
            return 0.0
        return len(grams & self._grams) / len(grams)


def load_lcb_statements(root: Path) -> list[str]:
    statements = []
    for path in sorted(Path(root).glob("test*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if str(record.get("contest_date", ""))[:10] >= HELD_OUT_FROM:
                    statements.append(record["question_content"])
    return statements
