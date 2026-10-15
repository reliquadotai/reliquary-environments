"""The row every source adapter emits, and the identity of a problem."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from reliquary_competitive_code.judge import TestCase

_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_NON_WORD = re.compile(r"[^a-z0-9]+")


# How a curated problem's tests were vouched for (`origin["reference_kind"]`):
# a Python 3 reference passed every kept test; a Python 2 reference converted
# by 2to3 did; or no Python reference passed and the tests ship on the
# upstream's word (official tests, or tests the upstream validated against
# accepted solutions), the teacher's answers being graded at export.
PYTHON_REFERENCE = "python_reference"
PY2TO3 = "py2to3"
UPSTREAM_TESTS_ONLY = "upstream_tests_only"
REFERENCE_KINDS = (PYTHON_REFERENCE, PY2TO3, UPSTREAM_TESTS_ONLY)


@dataclass(frozen=True, slots=True)
class SourceRow:
    source: str
    upstream_id: str
    statement: str
    tests: tuple[TestCase, ...]
    references: tuple[str, ...]
    contest_date: str | None = None
    # v1.1 sources. `upstream_tests`: the tests are the judge's own or were
    # validated upstream, so the problem may ship without a Python reference.
    # `time_limit_s`: the upstream (C++-calibrated) limit. `converted`: the
    # references that 2to3 produced from Python 2.
    upstream_tests: bool = False
    time_limit_s: float | None = None
    rating: int | None = None
    converted: frozenset[str] = frozenset()


def normalise(statement: str) -> str:
    """Lower-case words only: the same problem formatted twice compares equal."""
    text = _LATEX_COMMAND.sub(" ", statement.lower())
    return " ".join(_NON_WORD.sub(" ", text).split())


def problem_id(statement: str) -> str:
    return hashlib.sha256(normalise(statement).encode("utf-8")).hexdigest()[:16]
