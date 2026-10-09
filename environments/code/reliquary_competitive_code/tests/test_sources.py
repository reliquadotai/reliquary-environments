import json

import pyarrow as pa
import pyarrow.parquet as pq

from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources import deepcoder
from reliquary_competitive_code.sources.common import normalise, problem_id

PI_STATEMENT = (
    "Solve the following coding problem using the programming language python:\n\n"
    "Add two numbers $a$ and $b$.\n\n"
    "The input will be stdin and you should print your solution to stdout\n\n\n"
    "Now solve the problem and return the code."
)


def test_normalise_ignores_formatting_and_latex() -> None:
    assert normalise("Add  $a$ and \\texttt{b}!") == normalise("add a and b")
    assert problem_id("Add a and b") == problem_id("ADD A, AND B.")
    assert len(problem_id("x")) == 16


def test_primeintellect_rows_lose_their_wrapper_and_fence() -> None:
    record = {
        "problem": PI_STATEMENT,
        "tests": json.dumps([{"type": "stdin_stdout", "input": "1 2\n", "output": "3\n"}]),
        "solutions": ["```python\na, b = map(int, input().split())\nprint(a + b)\n```"],
    }
    row = deepcoder.convert("primeintellect", "pi#0", record)
    assert row.statement == "Add two numbers $a$ and $b$."
    assert row.tests == (TestCase("1 2\n", "3\n"),)
    assert row.references == ("a, b = map(int, input().split())\nprint(a + b)\n",)
    assert row.source == "deepcoder/primeintellect" and row.contest_date is None


def test_taco_rows_keep_their_code_and_skip_function_tests() -> None:
    stdin_record = {
        "problem": "Print the sum.",
        "tests": json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]}),
        "solutions": ["print(sum(map(int, input().split())))"],
    }
    row = deepcoder.convert("taco", "taco#0", stdin_record)
    assert row.tests == (TestCase("1 2\n", "3\n"),)
    assert row.references == ("print(sum(map(int, input().split())))",)

    function_record = dict(stdin_record, tests=json.dumps({"fn_name": "f", "inputs": [[1]], "outputs": [[1]]}))
    assert deepcoder.convert("taco", "taco#1", function_record) is None


def test_malformed_tests_drop_the_row() -> None:
    base = {"problem": "p", "solutions": []}
    for tests in (
        "not json",
        json.dumps({"inputs": ["1"], "outputs": []}),
        json.dumps({"inputs": [["1"]], "outputs": [["1"]]}),
        json.dumps([{"type": "functional", "input": "1", "output": "1"}]),
        json.dumps([]),
    ):
        assert deepcoder.convert("taco", "x", dict(base, tests=tests)) is None


def test_rows_reads_both_configs(tmp_path) -> None:
    for config, record in (
        ("taco", {"problem": "Print the sum.", "tests": json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]}), "solutions": ["print(3)"]}),
        ("primeintellect", {"problem": PI_STATEMENT, "tests": json.dumps([{"type": "stdin_stdout", "input": "1 2\n", "output": "3\n"}]), "solutions": ["```python\nprint(3)\n```"]}),
    ):
        (tmp_path / config).mkdir()
        pq.write_table(pa.Table.from_pylist([record]), tmp_path / config / "train-00000-of-00001.parquet")
    rows = list(deepcoder.rows(tmp_path))
    assert [row.source for row in rows] == ["deepcoder/taco", "deepcoder/primeintellect"]
    assert rows[0].upstream_id == "taco/train-00000-of-00001.parquet#0"


def test_primeintellect_strips_every_real_suffix_variant() -> None:
    stdin_new = (
        "The input will be given via stdin and the output should be printed to stdout by your code.\n\n"
        "Now solve the problem by providing the code."
    )
    bare = "Now solve the problem and return the code."
    for suffix in (stdin_new, bare):
        record = {
            "problem": "Solve the following coding problem using the programming language python:\n\nAdd $a$ and $b$.\n\n" + suffix,
            "tests": json.dumps([{"type": "stdin_stdout", "input": "1 2\n", "output": "3\n"}]),
            "solutions": [],
        }
        assert deepcoder.convert("primeintellect", "pi#0", record).statement == "Add $a$ and $b$."
