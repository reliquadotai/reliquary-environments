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
    ]


def test_build_applies_every_stage_and_reports_it() -> None:
    curated, report = build(_rows(), HeldOutIndex([HELD]), workers=2)
    # Rows 1 and 2 are the same problem; both have 6 tests, so the lead is the
    # larger (source, upstream_id): ("t", "1"). Row 2 brings no reference and
    # is not dropped for it, because duplicates pool references first.
    assert [c.origin["upstream_id"] for c in curated] == ["1"]
    assert report["loaded"] == 6
    assert report["dropped_date"] == 1
    assert report["dropped_contaminated"] == 1
    assert report["dropped_multi_answer"] == 1
    assert report["after_dedup"] == 2
    assert report["dropped_no_passing_reference"] == 1
    assert report["curated"] == 1
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
