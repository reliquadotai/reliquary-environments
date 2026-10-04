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
    assert merged[0].references == ("r2", "r1")
    assert merged[0].contest_date == "2021-01-01"


def test_near_duplicates_merge() -> None:
    near = LONG.replace("print the answer", "output the answer")
    assert len(deduplicate([_row(LONG, 3, ("a",)), _row(near, 4, ("b",))])) == 1


def test_different_problems_stay_apart() -> None:
    other = "Petya walks along a grid of h rows and w columns collecting coins on every cell he visits and must reach the bottom right corner print the maximum number of coins"
    assert len(deduplicate([_row(LONG, 3, ("a",)), _row(other, 3, ("b",))])) == 2


def test_output_order_and_references_are_deterministic() -> None:
    near = LONG.replace("print the answer", "output the answer")
    rows = [
        _row(LONG, 3, ("a",), "x"),
        _row(LONG.upper(), 3, ("b",), "y"),
        _row(LONG + "!", 3, ("c",), "w"),
        _row(near, 2, ("d",), "z"),
    ]
    assert len(deduplicate(rows)) == 1
    assert deduplicate(rows) == deduplicate(list(reversed(rows)))
    assert deduplicate(rows)[0].references == ("b", "c", "a", "d")


def test_references_come_lead_first() -> None:
    near = LONG.replace("print the answer", "output the answer")
    merged = deduplicate([_row(LONG, 9, ("L",)), _row(near, 2, ("X", "Y", "Z"))])
    assert len(merged) == 1 and merged[0].references[0] == "L"
    assert merged[0].references == ("L", "X", "Y", "Z")


def test_easy_and_hard_versions_stay_apart() -> None:
    easy = "This is the easy version of the problem. " + LONG
    hard = "This is the hard version of the problem. " + LONG
    assert len(deduplicate([_row(easy, 3, ("a",)), _row(hard, 3, ("b",))])) == 2


def test_different_constraint_numbers_stay_apart() -> None:
    small = LONG + " where 1 <= n <= 10"
    large = LONG + " where 1 <= n <= 3000"
    assert len(deduplicate([_row(small, 3, ("a",)), _row(large, 3, ("b",))])) == 2


def test_chain_of_near_duplicates_is_one_cluster() -> None:
    base = LONG + " " + LONG.replace("Vasya", "Misha").replace("segments", "pieces") + " " + LONG.replace("array", "sequence")
    b = base.replace("print the answer", "output the answer", 1)
    c = b.replace("minimum", "smallest", 1)
    assert len(deduplicate([_row(base, 3, ("a",)), _row(b, 4, ("b",)), _row(c, 5, ("c",))])) == 1


def test_formatting_of_numbers_does_not_split_duplicates() -> None:
    caret = LONG + " where n is at most 10^5 and s is at most 200 000"
    sup = LONG + " where n is at most 10<sup>5</sup> and s is at most 200000"
    assert len(deduplicate([_row(caret, 3, ("a",)), _row(sup, 4, ("b",))])) == 1


def test_shared_version_boilerplate_still_merges() -> None:
    text = "The only difference between easy and hard versions is the constraints on n. " + LONG
    assert len(deduplicate([_row(text, 3, ("a",)), _row(text.replace("print", "output"), 4, ("b",))])) == 1
