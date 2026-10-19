import datetime

from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources import codecontests_plus as ccplus
from reliquary_competitive_code.sources import openr1_codeforces as openr1
from reliquary_competitive_code.sources import py2

JULY_2024 = datetime.datetime(2024, 7, 1, tzinfo=datetime.timezone.utc).timestamp()


def _openr1(**changes) -> dict:
    record = {
        "id": "1500/C", "contest_id": "1500", "contest_start": JULY_2024, "index": "C", "time_limit": 2.0,
        "title": "T", "description": "Legend.", "input_format": "n", "output_format": "m",
        "interaction_format": None, "note": "Because.", "examples": [{"input": "1\n", "output": "2\n"}],
        "rating": 1600, "official_tests_complete": True, "input_mode": "stdio", "generated_checker": None,
        "executable": True, "generated_tests": 10,
    }
    record.update(changes)
    return record


def test_openr1_keeps_the_rated_stdio_problems_before_the_window() -> None:
    assert openr1.drop_reason(_openr1()) is None
    window = datetime.datetime(2024, 8, 1, tzinfo=datetime.timezone.utc).timestamp()
    assert openr1.drop_reason(_openr1(contest_start=window)) == "date"
    assert openr1.drop_reason(_openr1(interaction_format="Interact.")) == "not_stdio"
    assert openr1.drop_reason(_openr1(input_mode="file")) == "not_stdio"
    assert openr1.drop_reason(_openr1(generated_checker="int main(){}")) == "checker"
    assert openr1.drop_reason(_openr1(executable=False)) == "not_executable"
    assert openr1.drop_reason(_openr1(official_tests_complete=False, generated_tests=0)) == "no_tests"
    for rating in (None, 1100, 2500):
        assert openr1.drop_reason(_openr1(rating=rating)) == "rating"
    for rating in (1200, 2400):
        assert openr1.drop_reason(_openr1(rating=rating)) is None


def test_openr1_candidate_lays_the_statement_out_and_carries_the_limit() -> None:
    row = openr1.candidate(_openr1())
    assert row.statement == "Legend.\n\nInput\n\nn\n\nOutput\n\nm\n\nExamples\n\nInput\n\n1\n\nOutput\n\n2\n\nNote\n\nBecause."
    assert row.upstream_tests and row.time_limit_s == 2.0 and row.rating == 1600
    assert row.contest_date == "2024-07-01" and row.upstream_id == "1500/C"
    # The examples as the metadata dump gives them (a Python literal string).
    assert openr1.statement(_openr1(examples="[{'input': '1\\n', 'output': '2\\n'}]")) == row.statement


def test_openr1_languages() -> None:
    assert openr1.submission_language("Python 3") == "py3"
    assert openr1.submission_language("PyPy 3-64") == "py3"
    assert openr1.submission_language("Python 2") == "py2"
    assert openr1.submission_language("PyPy 2") == "py2"
    assert openr1.submission_language("GNU C++17") is None
    assert openr1.generated_tests_file("27") == "generated_tests/test_cases_0027.parquet"


NCMP = '#include "testlib.h"\nint main(int argc, char *argv[]) {\n    setName("compare ordered sequences of signed int%d numbers", 64);\n    registerTestlibCmd(argc, argv);\n}'
NYESNO = ('#include "testlib.h"\nconst string YES = "YES";\nconst string NO = "NO";\nint main(int argc, char *argv[]) {\n'
          '    setName("%s", ("multiple " + YES + "/" + NO + " (case insensitive)").c_str());\n    registerTestlibCmd(argc, argv);\n}')


def _rcmp(eps: str) -> str:
    return ('#include "testlib.h"\nconst double EPS = ' + eps + ';\nint main(int argc, char *argv[]) {\n'
            '    setName("compare two sequences of doubles, max absolute or relative  error = %.7f", EPS);\n'
            '    registerTestlibCmd(argc, argv);\n}')


def test_ccplus_keeps_only_the_checkers_our_comparison_reproduces() -> None:
    assert ccplus.checker_kind(NCMP) == "ncmp"
    assert ccplus.checker_kind(NYESNO) == "nyesno"
    assert ccplus.checker_kind(_rcmp("1E-6")) == "rcmp6"
    assert ccplus.checker_kind(_rcmp("1E-4")) == "rcmp4"
    assert ccplus.checker_kind(_rcmp("1E-9")) == "rcmp9"
    assert ccplus.checker_kind("int main() { return 0; }") == "custom"
    assert ccplus.checker_kind(None) == "custom"
    assert {"rcmp4", "rcmp9", "custom"}.isdisjoint(ccplus.GRADABLE_CHECKERS)


def _ccplus(**changes) -> dict:
    record = {"source": "AIZU", "id": "p01234", "title": "Twins", "description": "Print who.", "time_limit": 2000,
              "checker": NCMP, "true_positive_rate": 1.0}
    record.update(changes)
    return record


def test_ccplus_candidates_and_their_codeforces_dates_and_ratings() -> None:
    row = ccplus.candidate(_ccplus(), {})
    assert row.statement == "Twins\n\nPrint who." and row.upstream_id == "AIZU/p01234"
    assert row.time_limit_s == 2.0 and row.upstream_tests and row.contest_date is None
    assert not ccplus.candidate(_ccplus(true_positive_rate=0.9), {}).upstream_tests
    assert ccplus.drop_reason(_ccplus(), {}) is None
    assert ccplus.drop_reason(_ccplus(checker=_rcmp("1E-9")), {}) == "checker"
    meta = {"1059/D": {"date": "2018-10-01", "rating": 2200}, "1060/A": {"date": "2018-10-02", "rating": 800},
            "1061/A": {"date": "2024-08-10", "rating": 1500}}
    cf = lambda i: _ccplus(source="Codeforces", id=i)  # noqa: E731
    assert ccplus.drop_reason(cf("1059_D"), meta) is None
    assert ccplus.candidate(cf("1059_D"), meta).rating == 2200
    assert ccplus.drop_reason(cf("1060_A"), meta) == "rating"
    assert ccplus.drop_reason(cf("1061_A"), meta) == "date"
    assert ccplus.drop_reason(cf("1995_A"), meta) == "date"  # unknown to open-r1, in the window
    assert ccplus.drop_reason(cf("900_A"), meta) is None


def test_ccplus_yes_no_outputs_and_submissions() -> None:
    assert ccplus.yes_no_only([TestCase("1\n", "YES\nno\n"), TestCase("2\n", "Yes\n")])
    assert not ccplus.yes_no_only([TestCase("1\n", "YES 3\n")])
    record = {"correct_submissions": [{"code": "print(1)", "language": "py3"}, {"code": "print 1", "language": "py2"},
                                      {"code": "int main(){}", "language": "cpp"}, {"code": " ", "language": "py3"}]}
    assert ccplus.submissions(record) == (["print(1)"], ["print 1"])


def test_python2_conversion_reads_input_literally_and_leaves_python3_alone() -> None:
    code = py2.convert("n = input()\nfor i in xrange(n): print i * 2\n")
    assert "eval" not in code and "_py2_input()" in code and "range(n)" in code
    out = []
    exec(compile(code, "<t>", "exec"), {"input": lambda: "3", "print": out.append})
    assert out == [0, 2, 4]
    pair = py2.convert("a, b = input()\nprint a + b\n")
    out.clear()
    exec(compile(pair, "<t>", "exec"), {"input": lambda: "4, 5", "print": out.append})
    assert out == [9]
    refs, converted = py2.upgrade(("print(1)\n", "print 2\n"))
    assert refs == ("print(1)\n", "print(2)\n") and converted == frozenset({"print(2)\n"})
    assert not py2.looks_python2("print(1)") and py2.looks_python2("print 1") and py2.looks_python2("x = raw_input()")
    assert py2.convert("def (:\n") is None
