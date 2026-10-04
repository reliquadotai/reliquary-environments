from reliquary_competitive_code.build.dedup import deduplicate
from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources.common import SourceRow

LONG = (
    "Vasya has an array of n integers and wants to split it into the minimum "
    "number of contiguous segments such that every segment has a sum that does "
    "not exceed s and every segment contains at most k elements print the answer"
)


def _row(statement, tests, refs, source="a", date=None) -> SourceRow:
    return SourceRow(source, f"{source}#{tests}", statement, tuple(TestCase(str(i), str(i)) for i in range(tests)), refs, date)


def test_exact_duplicates_merge_and_keep_the_most_tests() -> None:
    merged = deduplicate([
        _row(LONG, 3, ("r1",), "taco"),
        _row(LONG.upper() + "!", 10, ("r2", "r1"), "primeintellect", "2021-01-01"),
    ])
    assert len(merged) == 1
    assert merged[0].source == "primeintellect" and len(merged[0].tests) == 10
    assert merged[0].references == ("r1", "r2")
    assert merged[0].contest_date == "2021-01-01"


def test_near_duplicates_merge() -> None:
    near = LONG.replace("print the answer", "output the answer")
    assert len(deduplicate([_row(LONG, 3, ("a",)), _row(near, 4, ("b",))])) == 1


def test_different_problems_stay_apart() -> None:
    other = "Petya walks along a grid of h rows and w columns collecting coins on every cell he visits and must reach the bottom right corner print the maximum number of coins"
    assert len(deduplicate([_row(LONG, 3, ("a",)), _row(other, 3, ("b",))])) == 2


def test_output_order_is_deterministic() -> None:
    rows = [_row(LONG, 3, ("a",)), _row("Print one integer the sum of a and b for every test case given in the input lines", 2, ("b",))]
    assert deduplicate(rows) == deduplicate(list(reversed(rows)))
