import json

from reliquary_competitive_code.build.filters import (
    HeldOutIndex,
    is_after_cutoff,
    is_multi_answer,
    load_lcb_statements,
)
from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources.common import SourceRow

HELD = (
    "Takahashi has N cards numbered 1 to N arranged in a row and wants to "
    "remove exactly K of them so that the remaining sum is maximal modulo M"
)


def _row(statement: str, date: str | None = None) -> SourceRow:
    return SourceRow("s", "u", statement, (TestCase("", ""),), ("print()",), date)


def test_dates_on_or_after_the_cutoff_are_excluded() -> None:
    assert is_after_cutoff(_row("x", "2024-08-01"))
    assert is_after_cutoff(_row("x", "2025-01-04"))
    assert not is_after_cutoff(_row("x", "2024-07-31"))
    assert not is_after_cutoff(_row("x", None))


def test_multiple_answer_statements_are_detected() -> None:
    assert is_multi_answer(_row("If there are several answers, print any of them."))
    assert is_multi_answer(_row("Output any valid permutation."))
    assert not is_multi_answer(_row("Print the number of ways modulo 10^9+7."))


def test_overlap_with_held_out_statements() -> None:
    index = HeldOutIndex([HELD])
    assert index.overlap("Story. " + HELD + " Constraints follow.") > 0.5
    assert index.overlap("Print the sum of two integers a and b given on one line of input") == 0.0
    assert index.overlap("too short") == 0.0


def test_lcb_loader_keeps_only_held_out_dates(tmp_path) -> None:
    lines = [
        {"question_content": "old problem", "contest_date": "2024-07-31T00:00:00"},
        {"question_content": "new problem", "contest_date": "2024-08-01T00:00:00"},
    ]
    (tmp_path / "test5.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert load_lcb_statements(tmp_path) == ["new problem"]
