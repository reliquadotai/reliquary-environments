import json

from reliquary_competitive_code.build.filters import (
    HeldOutIndex,
    example_disagrees,
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


def test_tie_break_patterns_are_not_multi_answer() -> None:
    # Tie-break patterns should NOT be marked as multi-answer
    assert not is_multi_answer(_row("If there are several such numbers, print the smallest one."))
    assert not is_multi_answer(_row("If there are multiple such strings, find the lexicographically smallest one among them."))
    assert not is_multi_answer(_row("If there are several solutions, Vasya wants to find the smallest possible x."))
    assert not is_multi_answer(_row("find the smallest of them"))


def test_multi_answer_without_tie_break_still_counts() -> None:
    # Original positive case must still pass
    assert is_multi_answer(_row("If there are several answers, print any of them."))


def test_multi_answer_in_second_sentence_after_tie_break() -> None:
    # Multi-answer in second sentence after a tie-break sentence
    assert is_multi_answer(_row("If there are several such numbers, print the smallest one. Otherwise, print any valid arrangement."))


# Codeforces 1433D (problem 00d7ec4956b61896): any valid tree is accepted, and
# the upstream test for the statement's own example expects another tree.
_ROADS = (
    "Build n-1 roads.\n\n-----Input-----\n\nThe first line contains t.\n\n"
    "-----Output-----\n\nFor each test case, print NO or YES and the roads.\n\n"
    "-----Example-----\nInput\n2\n3\n1 2 2\n2\n1 1\n\nOutput\nYES\n1 2\n1 3\nNO\n"
)


def _with_tests(statement: str, *tests: tuple[str, str]) -> SourceRow:
    return SourceRow("s", "u", statement, tuple(TestCase(a, b) for a, b in tests), ("print()",))


def test_an_example_whose_test_expects_another_output_disagrees() -> None:
    row = _with_tests(_ROADS, ("2\n3\n1 2 2\n2\n1 1\n", "YES\n1 3\n1 2\nNO\n"))
    assert example_disagrees(row)


def test_an_example_whose_test_expects_the_same_output_agrees() -> None:
    # Spacing and the comparator's relaxations (yes/no case) do not count.
    row = _with_tests(_ROADS, ("2\n3\n1 2 2\n2 \n1 1", "yes\n1 2\n1 3\nno\n"), ("1\n2\n1 2\n", "YES\n1 2\n"))
    assert not example_disagrees(row)


def test_an_example_with_no_matching_test_is_not_evidence() -> None:
    row = _with_tests(_ROADS, ("1\n2\n1 2\n", "YES\n2 1\n"))
    assert not example_disagrees(row)


def test_blank_line_separated_examples_and_notes() -> None:
    # The primeintellect layout: blank lines after every marker, then a Note.
    statement = (
        "Print any permutation.\n\nExamples\n\nInput\n\n3\n\nOutput\n\n1 2 3\n\n"
        "Input\n\n2\n\nOutput\n\n2 1\n\nNote\n\nIn the first example 3 2 1 is also fine."
    )
    agree = _with_tests(statement, ("3\n", "1 2 3\n"), ("2\n", "2 1\n"))
    second_differs = _with_tests(statement, ("3\n", "1 2 3\n"), ("2\n", "1 2\n"))
    assert not example_disagrees(agree)
    assert example_disagrees(second_differs)


def test_sample_markers_with_numbers_and_an_output_spanning_a_blank_line() -> None:
    statement = (
        "Text.\n\nSAMPLE INPUT 1\n2\n4 5\n\nSAMPLE OUTPUT 1\n9\n\n20\n\n"
        "Explanation\n\n4 + 5 = 9."
    )
    assert not example_disagrees(_with_tests(statement, ("2\n4 5\n", "9\n\n20\n")))
    assert example_disagrees(_with_tests(statement, ("2\n4 5\n", "9\n\n21\n")))


def test_print_any_is_multi_answer_even_next_to_a_tie_break_word() -> None:
    # e95412556bebb480 and eafddd74faf6e693: the optimum is fixed, the witness is free.
    assert is_multi_answer(_row("If there are multiple solutions where the number of universal quantifiers is maximum, print any."))
    assert is_multi_answer(_row("In the second line, print any array obtained with the minimum number of moves."))
    assert is_multi_answer(_row("If there are multiple shortest good subsequences, output any of them."))


def test_html_escaped_examples_compare_unescaped() -> None:
    # ca64a3e31741a199: the statement shows "&lt;" where the test has "<".
    statement = "Restore the type.\n\nInput\n3\npair pair int int int\n\nOutput\npair&lt;pair&lt;int,int&gt;,int&gt;\n"
    assert not example_disagrees(_with_tests(statement, ("3\npair pair int int int\n", "pair<pair<int,int>,int>\n")))
    assert example_disagrees(_with_tests(statement, ("3\npair pair int int int\n", "pair<int,pair<int,int>>\n")))
