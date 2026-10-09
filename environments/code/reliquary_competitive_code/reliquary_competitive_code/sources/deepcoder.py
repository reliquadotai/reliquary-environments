"""DeepCoder-Preview-Dataset: taco and primeintellect train configs.

Kept: problems whose every test is a stdin/stdout pair of strings. Dropped:
function-call tests (reliquary-code covers that contract), malformed tests.
`lcbv5/train` is not read in v1 because it carries no reference solution, and
`lcbv5/test` and `codeforces/test` are never read: their dates fall inside the
held-out LiveCodeBench window.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from reliquary_competitive_code.extraction import extract_program
from reliquary_competitive_code.judge import TestCase
from reliquary_competitive_code.sources.common import SourceRow

REPOSITORY = "agentica-org/DeepCoder-Preview-Dataset"
REVISION = "177913a7bd43791646ef6a43645caa3c871ab3db"
CONFIGS = ("taco", "primeintellect")

_PI_PREFIX = re.compile(r"^\s*Solve the following coding problem using the programming language python:\s*")
_PI_SUFFIX = re.compile(
    r"\s*(?:The input will be stdin and you should print your solution to stdout"
    r"|The input will be given via stdin and the output should be printed to stdout by your code\.)?"
    r"\s*Now solve the problem (?:and return|by providing) the code\.\s*$"
)


def rows(root: Path) -> Iterator[SourceRow]:
    for config in CONFIGS:
        for path in sorted((Path(root) / config).glob("train-*.parquet")):
            table = pq.read_table(path, columns=["problem", "tests", "solutions"])
            for offset, record in enumerate(table.to_pylist()):
                row = convert(config, f"{config}/{path.name}#{offset}", record)
                if row is not None:
                    yield row


def convert(config: str, upstream_id: str, record: dict) -> SourceRow | None:
    tests = _tests(record.get("tests"))
    if not tests:
        return None
    statement = record.get("problem") or ""
    if config == "primeintellect":
        statement = _PI_SUFFIX.sub("", _PI_PREFIX.sub("", statement))
    references = tuple(
        code
        for code in (_reference(config, solution) for solution in record.get("solutions") or ())
        if code
    )
    return SourceRow(
        source=f"deepcoder/{config}",
        upstream_id=upstream_id,
        statement=statement.strip(),
        tests=tests,
        references=references,
    )


def _tests(raw: object) -> tuple[TestCase, ...]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return ()
    if isinstance(data, dict):
        if data.get("fn_name"):
            return ()
        inputs, outputs = data.get("inputs") or [], data.get("outputs") or []
        if len(inputs) != len(outputs):
            return ()
        pairs = list(zip(inputs, outputs))
    elif isinstance(data, list):
        if any(not isinstance(item, dict) or item.get("type") != "stdin_stdout" for item in data):
            return ()
        pairs = [(item.get("input"), item.get("output")) for item in data]
    else:
        return ()
    if not pairs or any(not isinstance(a, str) or not isinstance(b, str) for a, b in pairs):
        return ()
    return tuple(TestCase(a, b) for a, b in pairs)


def _reference(config: str, solution: object) -> str | None:
    if not isinstance(solution, str) or not solution.strip():
        return None
    if config == "primeintellect":
        return extract_program(solution)
    return solution
