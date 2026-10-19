"""ByteDance-Seed/Code-Contests-Plus, `ccplus_1x` (CC-BY-4.0).

Problems from Codeforces, AtCoder, AIZU and CodeChef with tests made by a
generator under a validator, and the rates at which upstream's correct and
incorrect submissions were told apart. Kept: problems whose checker is one of
the testlib stock checkers our comparison reproduces:

- `ncmp` (signed integers), `wcmp` (tokens), `lcmp` (lines, which tokens
  relax), huge integers: token-exact, as ours.
- `rcmp6` (doubles, 1e-6 absolute or relative): our decimal tolerance.
- `nyesno` (several YES/NO, case-insensitive): our comparison already takes
  `yes`/`no` in any case, so these are kept with no change to the grader
  (a grader change would move the judge version of every live task). A
  problem under this checker whose expected outputs hold anything other than
  yes/no tokens is dropped (`yesno_checker_other_tokens`).

Dropped: `rcmp4` and `rcmp9` (a tolerance ours does not reproduce: 1e-4 would
mark valid answers wrong, 1e-9 accept wrong ones) and every custom checker.

Tests count as upstream-validated (`upstream_tests`) when every upstream
correct submission passed them (true positive rate 1.0). References: up to
`PER_PROBLEM` correct Python 3 submissions, then Python 2 ones by 2to3.
Codeforces problems get the contest date and rating open-r1 knows, so the
date cut and the 1200-2400 rating band apply to them as to open-r1.
"""

from __future__ import annotations

import re

from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources.common import SourceRow

SOURCE = "ccplus"
REPOSITORY = "ByteDance-Seed/Code-Contests-Plus"
REVISION = "96c850540fade31d384a25766461e0da6b08f5fc"
DATA_PATTERN = "ccplus_1x/part-*.parquet"
PER_PROBLEM = 5
METADATA_COLUMNS = ["source", "id", "title", "description", "time_limit", "checker", "true_positive_rate"]
TEST_COLUMNS = ["id", "test_cases", "correct_submissions"]
# The first Codeforces contest of the held-out window (1993: 2024-08-04);
# contests numbered from it are dropped when open-r1 has no date for them.
CODEFORCES_HELD_OUT_CONTEST = 1993

_SET_NAME = re.compile(r'setName\(\s*"([^"]*)"')
_EPS = re.compile(r"const\s+double\s+EPS\s*=\s*([0-9.eE+-]+)")
_YES_NO = frozenset({"yes", "no"})


def checker_kind(checker: str | None) -> str:
    """The stock checker a CC+ checker source is, or 'custom'."""
    if not checker or "registerTestlibCmd" not in checker:
        return "custom"
    match = _SET_NAME.search(checker)
    name = match.group(1) if match else ""
    if name.startswith("compare ordered sequences of signed int"):
        return "ncmp"
    if name == "compare sequences of tokens":
        return "wcmp"
    if name == "compare files as sequence of lines":
        return "lcmp"
    if name == "compare two signed huge integers":
        return "hcmp"
    if '("multiple " + YES + "/" + NO + " (case insensitive)")' in checker and 'YES = "YES"' in checker:
        return "nyesno"
    if name.startswith("compare two sequences of doubles"):
        eps = _EPS.search(checker)
        if eps:
            value = float(eps.group(1))
            return {1e-4: "rcmp4", 1e-6: "rcmp6", 1e-9: "rcmp9"}.get(value, "custom")
    return "custom"


GRADABLE_CHECKERS = frozenset({"ncmp", "wcmp", "lcmp", "hcmp", "nyesno", "rcmp6"})


def statement(record: dict) -> str:
    title = (record.get("title") or "").strip()
    body = (record.get("description") or "").strip()
    return f"{title}\n\n{body}".strip() if title else body


def upstream_id(record: dict) -> str:
    return f"{record['source']}/{record['id']}"


def codeforces_key(record: dict) -> str | None:
    """'1059/D' for Codeforces problem '1059_D' (open-r1's id), else None."""
    if record.get("source") != "Codeforces":
        return None
    contest, _, index = str(record["id"]).partition("_")
    return f"{contest}/{index}" if contest.isdigit() and index else None


def drop_reason(record: dict, codeforces_meta: dict[str, dict]) -> str | None:
    if not record.get("description"):
        return "no_statement"
    if checker_kind(record.get("checker")) not in GRADABLE_CHECKERS:
        return "checker"
    key = codeforces_key(record)
    if key is not None:
        meta = codeforces_meta.get(key)
        if meta is None:
            if int(key.split("/")[0]) >= CODEFORCES_HELD_OUT_CONTEST:
                return "date"
        else:
            date = meta.get("date")
            if date is None or date >= "2024-08-01":
                return "date"
            rating = meta.get("rating")
            if rating is not None and not 1200 <= rating <= 2400:
                return "rating"
    return None


def candidate(record: dict, codeforces_meta: dict[str, dict]) -> SourceRow:
    key = codeforces_key(record)
    meta = codeforces_meta.get(key) if key else None
    limit_ms = record.get("time_limit")
    return SourceRow(
        source=SOURCE,
        upstream_id=upstream_id(record),
        statement=statement(record),
        tests=(),
        references=(),
        contest_date=meta.get("date") if meta else None,
        upstream_tests=record.get("true_positive_rate") == 1.0,
        time_limit_s=limit_ms / 1000 if limit_ms else None,
        rating=meta.get("rating") if meta else None,
    )


def tests_of(record: dict) -> list[TestCase]:
    return [
        TestCase(t["input"].replace("\r\n", "\n"), t["output"].replace("\r\n", "\n"))
        for t in record.get("test_cases") or ()
        if isinstance(t.get("input"), str) and isinstance(t.get("output"), str)
    ]


def yes_no_only(tests: list[TestCase]) -> bool:
    return all(token.lower() in _YES_NO for test in tests for token in test.stdout.split())


def submissions(record: dict) -> tuple[list[str], list[str]]:
    """(Python 3 codes, Python 2 codes) among the correct submissions."""
    py3, py2 = [], []
    for sub in record.get("correct_submissions") or ():
        code, language = sub.get("code"), sub.get("language")
        if not isinstance(code, str) or not code.strip():
            continue
        if language == "py3":
            py3.append(code)
        elif language == "py2":
            py2.append(code)
    return py3, py2
