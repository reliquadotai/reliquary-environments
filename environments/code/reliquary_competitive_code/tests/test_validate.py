from reliquary_competitive_code.build.validate import (
    MIN_TESTS,
    Curated,
    Rejection,
    curate,
    select_tests,
    split_of,
    time_limit,
)
from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources.common import SourceRow, problem_id

SUM = "a, b = map(int, input().split())\nprint(a + b)\n"
TESTS = tuple(TestCase(f"{i} {i}\n", f"{2 * i}\n") for i in range(6))


def _row(refs, tests=TESTS) -> SourceRow:
    return SourceRow("deepcoder/taco", "taco#0", "Print a plus b.", tests, tuple(refs))


def test_a_passing_reference_curates_the_problem() -> None:
    curated = curate(_row([SUM]))
    assert isinstance(curated, Curated)
    assert curated.problem_id == problem_id("Print a plus b.")
    assert curated.reference == SUM and curated.tests == TESTS
    assert curated.time_limit_s == 1.0
    assert curated.origin == {"source": "deepcoder/taco", "upstream_id": "taco#0"}
    assert curated.split == split_of(curated.problem_id)


def test_the_first_passing_reference_is_kept_after_a_crashing_one() -> None:
    curated = curate(_row(["raise SystemExit(3)", SUM]))
    assert isinstance(curated, Curated) and curated.reference == SUM


def test_no_passing_reference_rejects() -> None:
    assert curate(_row(["raise ValueError"])) == Rejection(problem_id("Print a plus b."), "no_passing_reference")
    assert curate(_row(["print(0)", "raise ValueError"])).reason == "no_passing_reference"


def test_a_wrong_answer_beside_a_passing_reference_rejects() -> None:
    assert curate(_row([SUM, "print(0)"])).reason == "references_disagree"


def test_too_few_tests_rejects() -> None:
    assert curate(_row([SUM], TESTS[: MIN_TESTS - 1])).reason == "too_few_tests"


def test_too_few_tests_rejects_before_any_reference_runs() -> None:
    for tests in ((), TESTS[:4]):
        assert curate(_row(["import os"], tests)).reason == "too_few_tests"


def test_select_tests_of_nothing_is_empty() -> None:
    assert select_tests([], []) == ()


def test_slow_references_reject(monkeypatch) -> None:
    # Slow by construction, whatever the machine: a modest loop against a
    # per-test ceiling of 50 ms.
    from reliquary_competitive_code.build import validate

    monkeypatch.setattr(validate, "MAX_REFERENCE_TEST_S", 0.05)
    slow = "s = 0\nfor i in range(5_000_000):\n    s += i\n" + SUM
    assert curate(_row([slow])).reason == "reference_too_slow"


def test_time_limit_is_four_times_the_reference_within_bounds() -> None:
    assert time_limit(0.01) == 1.0
    assert time_limit(0.5) == 2.0
    assert time_limit(1.5) == 4.0


def test_select_tests_respects_the_cpu_budget_and_keeps_extremes() -> None:
    tests = [TestCase("x" * n, "1") for n in (5, 1, 9, 3)]
    assert select_tests(tests, [1.0, 1.0, 1.0, 1.0]) == (0, 1, 2, 3)
    kept = select_tests(tests, [3.0, 3.0, 3.0, 3.0])
    assert kept == (1, 2)  # largest then smallest input first, budget 8 s


def test_select_tests_prefers_the_largest_test_when_only_one_fits() -> None:
    tests = [TestCase("x" * n, "1") for n in (5, 1, 9, 3)]
    assert select_tests(tests, [5.0, 5.0, 5.0, 5.0]) == (2,)


def test_a_forbidden_import_reference_is_skipped_not_counted() -> None:
    curated = curate(_row(["import os\n" + SUM, SUM]))
    assert isinstance(curated, Curated) and curated.reference == SUM
    # Skipped references do not use up the references tried...
    curated = curate(_row(["import os\n" + SUM] * 3 + [SUM]))
    assert isinstance(curated, Curated) and curated.reference == SUM
    # ...and are not a disagreement.
    assert isinstance(curate(_row([SUM, "import subprocess\nprint(0)"])), Curated)


def test_splits_are_stable_and_roughly_95_25_25() -> None:
    splits = [split_of(f"{i:016x}") for i in range(4000)]
    assert splits == [split_of(f"{i:016x}") for i in range(4000)]
    assert 0.93 < splits.count("train") / 4000 < 0.97
    assert splits.count("eval") > 50 and splits.count("qualification") > 50
