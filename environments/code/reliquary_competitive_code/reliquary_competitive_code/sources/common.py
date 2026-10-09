"""The row every source adapter emits, and the identity of a problem."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from reliquary_competitive_code.judge import TestCase

_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_NON_WORD = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class SourceRow:
    source: str
    upstream_id: str
    statement: str
    tests: tuple[TestCase, ...]
    references: tuple[str, ...]
    contest_date: str | None = None


def normalise(statement: str) -> str:
    """Lower-case words only: the same problem formatted twice compares equal."""
    text = _LATEX_COMMAND.sub(" ", statement.lower())
    return " ".join(_NON_WORD.sub(" ", text).split())


def problem_id(statement: str) -> str:
    return hashlib.sha256(normalise(statement).encode("utf-8")).hexdigest()[:16]
