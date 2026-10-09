import json

import pyarrow.parquet as pq

from reliquary_competitive_code.build import io
from reliquary_competitive_code.build.filters import HeldOutIndex
from reliquary_competitive_code.build.pipeline import build
from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources.common import SourceRow

SUM = "a, b = map(int, input().split())\nprint(a + b)\n"
TESTS = tuple(TestCase(f"{i} {i}\n", f"{2 * i}\n") for i in range(6))
HELD = "Takahashi has N cards numbered 1 to N arranged in a row and wants to remove exactly K of them so that the remaining sum is maximal modulo M"


def _rows():
    return [
        SourceRow("t", "1", "Print the sum of a and b given on a single line.", TESTS, (SUM,)),
        SourceRow("p", "2", "PRINT THE SUM OF A AND B, given on a single line!", TESTS, ()),
        SourceRow("t", "3", "Story. " + HELD, TESTS, (SUM,)),
        SourceRow("t", "4", "Print the sum. If there are several answers, print any of them.", TESTS, (SUM,)),
        SourceRow("t", "5", "Print a plus b, a newer problem.", TESTS, (SUM,), "2025-02-01"),
        SourceRow("t", "6", "Print the product of a and b given on a single line.", TESTS, ("print(0)",)),
        SourceRow(
            "t", "7", "Print a pair whose sum is n.\n\nInput\n4\n\nOutput\n1 3\n",
            (TestCase("4\n", "2 2\n"),), ("print(2, 2)",),
        ),
    ]


def test_build_applies_every_stage_and_reports_it() -> None:
    curated, report = build(_rows(), HeldOutIndex([HELD]), workers=2)
    # Rows 1 and 2 are the same problem; both have 6 tests, so the lead is the
    # larger (source, upstream_id): ("t", "1"). Row 2 brings no reference and
    # is not dropped for it, because duplicates pool references first.
    assert [c.origin["upstream_id"] for c in curated] == ["1"]
    assert report["loaded"] == 7
    assert report["dropped_example_mismatch"] == 1
    assert report["dropped_date"] == 1
    assert report["dropped_contaminated"] == 1
    assert report["dropped_multi_answer"] == 1
    assert report["after_dedup"] == 2
    assert report["dropped_no_passing_reference"] == 1
    assert report["curated"] == 1
    assert report["dropped_harness_overload"] == 0
    assert sum(report[f"split_{s}"] for s in ("train", "eval", "qualification")) == 1


def test_write_dataset_round_trips(tmp_path) -> None:
    curated, _ = build(_rows(), HeldOutIndex([HELD]), workers=1)
    digests = io.write_dataset(curated, tmp_path)
    assert set(digests) == {io.problems_file(s) for s in ("train", "eval", "qualification")} | {io.REFERENCES_FILE}
    split = curated[0].split
    table = pq.read_table(tmp_path / io.problems_file(split))
    row = table.to_pylist()[0]
    assert row["problem_id"] == curated[0].problem_id
    assert [TestCase(a, b) for a, b in json.loads(row["tests_json"])] == list(curated[0].tests)
    assert json.loads(row["origin_json"]) == curated[0].origin
    refs = pq.read_table(tmp_path / io.REFERENCES_FILE).to_pylist()
    assert refs == [{"problem_id": curated[0].problem_id, "code": SUM}]


def test_the_report_counts_overloaded_problems(monkeypatch) -> None:
    from reliquary_competitive_code.build import pipeline
    from reliquary_competitive_code.build.validate import Rejection

    monkeypatch.setattr(pipeline, "curate", lambda row: Rejection("x", "harness_overload"))
    curated, report = build(_rows(), HeldOutIndex([HELD]), workers=1)
    assert curated == [] and report["dropped_harness_overload"] == 2


def test_the_build_cli_refuses_an_overloaded_build(monkeypatch, tmp_path, capsys) -> None:
    import sys

    import pytest

    from reliquary_competitive_code.build import __main__ as cli

    monkeypatch.setattr(cli.deepcoder, "rows", lambda path: [])
    monkeypatch.setattr(cli, "load_lcb_statements", lambda path: [])
    monkeypatch.setattr(cli, "build", lambda rows, held, workers: ([], {"dropped_harness_overload": 3}))
    monkeypatch.setattr(sys, "argv", ["build", "--deepcoder", "d", "--lcb", "l", "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code not in (0, None)
    assert "harness_overload" in str(exited.value.code) + capsys.readouterr().err
    assert not (tmp_path / "out" / "problems-train.parquet").exists()


def test_a_disagreeing_example_drops_every_copy_of_the_problem() -> None:
    # a029962b1a30a621: one copy's tests contradict the statement's example,
    # another copy's tests never feed it; both copies are the same problem.
    statement = "Print a number with k digits and digital root d.\n\nInput\n4 4\n\nOutput\n5881\n"
    rows = [
        SourceRow("p", "1", statement, (TestCase("4 4\n", "4000\n"),), ("print(4000)",)),
        SourceRow("p", "2", statement.replace("Print a", "print   a"), (TestCase("1 0\n", "0\n"),), ("print(0)",)),
    ]
    curated, report = build(rows, HeldOutIndex([]), workers=1)
    assert curated == [] and report["dropped_example_mismatch"] == 2
