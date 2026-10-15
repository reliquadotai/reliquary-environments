"""open-r1/codeforces (CC-BY-4.0): Codeforces problems with official and
generated tests, and open-r1/codeforces-submissions for accepted Python.

Kept (`drop_reason` is None): train split, contest before 2024-08-01 (the
held-out LiveCodeBench window), rating 1200 to 2400, stdin/stdout, not
interactive, no special checker (`generated_checker` empty: the default
token comparison judges it), executable, and tests (official complete or
generated). The statement is laid out as Codeforces shows it: legend, Input,
Output, Examples, Note.

Tests are the official ones plus the generated ones (validated upstream
against accepted solutions), so a problem may ship without a Python reference
(`upstream_tests`). References: up to `PER_PROBLEM` accepted Python 3
submissions, then Python 2 ones converted by 2to3.
"""

from __future__ import annotations

import ast
import datetime
from collections.abc import Iterable, Iterator

from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources.common import SourceRow

SOURCE = "open-r1/codeforces"
REPOSITORY = "open-r1/codeforces"
REVISION = "fbe3f6e903ee854eec2e69e9d96d0306cde59baf"
SUBMISSIONS_REPOSITORY = "open-r1/codeforces-submissions"
SUBMISSIONS_REVISION = "07e7bf31f284b0e90e02dfc7ab675d5821e1c068"
DATA_PATTERN = "data/train-*.parquet"
SUBMISSIONS_PATTERN = "data/train-*.parquet"
MIN_RATING, MAX_RATING = 1200, 2400
HELD_OUT_FROM = datetime.datetime(2024, 8, 1, tzinfo=datetime.timezone.utc).timestamp()
PER_PROBLEM = 5
METADATA_COLUMNS = [
    "id", "contest_id", "contest_start", "index", "time_limit", "title", "description",
    "input_format", "output_format", "interaction_format", "note", "examples", "rating",
    "official_tests_complete", "input_mode", "generated_checker", "executable", "generated_tests",
]


def generated_tests_file(contest_id: str) -> str:
    return f"generated_tests/test_cases_{int(contest_id):04d}.parquet"


def _examples(raw: object) -> list[dict]:
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    return [e for e in raw or () if isinstance(e, dict)]


def statement(record: dict) -> str:
    parts = [
        (record.get("description") or "").strip(), "",
        "Input", "", (record.get("input_format") or "").strip(), "",
        "Output", "", (record.get("output_format") or "").strip(), "",
        "Examples", "",
    ]
    for example in _examples(record.get("examples")):
        parts += ["Input", "", str(example.get("input", "")).strip(), "",
                  "Output", "", str(example.get("output", "")).strip(), ""]
    if record.get("note"):
        parts += ["Note", "", record["note"].strip()]
    return "\n".join(parts).strip()


def contest_date(record: dict) -> str | None:
    start = record.get("contest_start")
    if not start:
        return None
    return datetime.datetime.fromtimestamp(start, datetime.timezone.utc).strftime("%Y-%m-%d")


def drop_reason(record: dict) -> str | None:
    """Why a metadata row is not a candidate, or None."""
    if not record.get("contest_start") or record["contest_start"] >= HELD_OUT_FROM:
        return "date"
    if record.get("interaction_format") or record.get("input_mode") != "stdio":
        return "not_stdio"
    if record.get("generated_checker"):
        return "checker"
    if not record.get("executable") or not record.get("description"):
        return "not_executable"
    if not (record.get("official_tests_complete") or (record.get("generated_tests") or 0) > 0):
        return "no_tests"
    rating = record.get("rating")
    if rating is None or not MIN_RATING <= rating <= MAX_RATING:
        return "rating"
    return None


def candidate(record: dict) -> SourceRow:
    """The statement-only row of a kept metadata record (tests and references
    are attached once the problem survives selection)."""
    return SourceRow(
        source=SOURCE,
        upstream_id=record["id"],
        statement=statement(record),
        tests=(),
        references=(),
        contest_date=contest_date(record),
        upstream_tests=True,
        time_limit_s=float(record["time_limit"]) if record.get("time_limit") else None,
        rating=record.get("rating"),
    )


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n")


def official_tests(batches: Iterable[dict], wanted: set[str]) -> Iterator[tuple[str, list[TestCase]]]:
    """(id, tests) from records carrying `id` and `official_tests`."""
    for record in batches:
        if record["id"] in wanted:
            yield record["id"], [
                TestCase(_normalise_newlines(t["input"]), _normalise_newlines(t["output"]))
                for t in record.get("official_tests") or ()
                if isinstance(t.get("input"), str) and isinstance(t.get("output"), str)
            ]


def generated_test(record: dict) -> TestCase | None:
    stdin, stdout = record.get("input"), record.get("output")
    if not isinstance(stdin, str) or not isinstance(stdout, str):
        return None
    return TestCase(_normalise_newlines(stdin), _normalise_newlines(stdout))


def submission_language(language: str | None) -> str | None:
    """'py3', 'py2' or None, from Codeforces' language names ("Python 3",
    "PyPy 3-64", "Python 2", "PyPy 2")."""
    if not language or "Py" not in language:
        return None
    return "py2" if "2" in language.split("-")[0] else "py3"
