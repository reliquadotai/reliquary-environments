"""What must never be trained on, and what this version cannot grade.

Held out: LiveCodeBench problems dated from 2024-08-01, the window the held-out
benchmark measures. A source with dates is cut on its date; taco and
primeintellect carry none, so every statement is also compared, by 13-gram
overlap, with every held-out LiveCodeBench statement.

Not gradable in v1: problems that accept several outputs. The comparator has
one expected output per test; such a problem would mark valid answers wrong.
Two detectors: the statement says so ("print any of them"), or, the objective
one, a test whose input is an example of the statement expects an output other
than the one the statement shows for it (Codeforces 1433D: the example prints
one valid tree, the upstream test expects another). The second catches
statements that never say "any" and tie-break words that fool the first.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable
from pathlib import Path

from reliquary_competitive_code.judge import outputs_match
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

# "print any ..." frees the answer even when the sentence also names an
# optimum ("... with the minimum number of moves, print any of them").
_ANY_ANSWER = re.compile(r"(print|output)\s+any\b", re.IGNORECASE)

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
        # A sentence counts if it says "print any", or matches _MULTI_ANSWER
        # without naming a tie-break ("if there are several, print the smallest").
        if _ANY_ANSWER.search(sentence) or (
            _MULTI_ANSWER.search(sentence) and not _TIE_BREAK.search(sentence)
        ):
            return True
    return False


_MARKER_WORDS = frozenset(
    {"sample", "samples", "example", "examples", "input", "inputs", "output", "outputs", "for", "the", "of"}
)


def _marker_kind(line: str) -> str | None:
    """'input'/'output' for an example heading ("Input", "-----Sample Output 2-----",
    "Output for the Sample Input"), else None."""
    words = re.findall(r"[a-z]+|\d+|\S", line.lower())
    words = [w for w in words if w not in {"-", "#", ">", "*", ":", "=", "_"}]
    if not words or len(words) > 6 or any(not w.isdigit() and w not in _MARKER_WORDS for w in words):
        return None
    for word in words:
        if word.startswith("input"):
            return "input"
        if word.startswith("output"):
            return "output"
    return None


def _blocks(statement: str) -> list[tuple[str, str, str]]:
    """Each example heading with its text: (kind, up to the first blank line,
    up to the next heading). The short reading is the usual layout; the long
    one keeps a block that itself contains a blank line."""
    lines = statement.splitlines()
    heads = [(i, kind) for i, line in enumerate(lines) if (kind := _marker_kind(line))]
    blocks = []
    for n, (i, kind) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        body = lines[i + 1 : end]
        while body and not body[0].strip():
            body.pop(0)
        short = []
        for line in body:
            if not line.strip():
                break
            short.append(line)
        blocks.append((kind, "\n".join(short), "\n".join(body)))
    return blocks


def example_disagrees(row: SourceRow) -> bool:
    """True when a test feeds an example input of the statement and expects an
    output the statement does not show for it: the problem accepts several
    answers (or its test is wrong), and the comparator would mark a valid one
    wrong. An example with no matching test is no evidence either way."""
    expected_by_input = {}
    for test in row.tests:
        expected_by_input.setdefault(tuple(test.stdin.split()), []).append(test.stdout)
    blocks = _blocks(row.statement)
    for (kind, short, long), following in zip(blocks, blocks[1:]):
        if kind != "input" or following[0] != "output":
            continue
        expected = expected_by_input.get(tuple(short.split())) or expected_by_input.get(tuple(long.split()))
        if not expected or not following[2].strip():
            continue
        _, shown_short, shown_long = following
        # Some statements are HTML-escaped ("&lt;") where the tests are not.
        shorts = (shown_short, html.unescape(shown_short))
        longs = (shown_long, html.unescape(shown_long))
        for want in expected:
            width = len(want.split())
            readings = (*shorts, *(" ".join(text.split()[:width]) for text in longs))
            if not any(outputs_match(want, text) for text in readings):
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
