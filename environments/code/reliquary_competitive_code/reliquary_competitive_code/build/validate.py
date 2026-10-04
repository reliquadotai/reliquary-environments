"""Keep a problem only if a known solution passes every test we keep.

A test the reference fails is wrong or impossible under our limits; the
problem is dropped rather than the test, because a problem kept with fewer
tests is silently weaker. Two references that disagree (one passes, another
gives a different answer) usually mean a problem with several valid outputs,
which v1 cannot grade, so that drops it too. A reference that imports a module
the task forbids is skipped: it is neither a reference tried nor a dissent,
since it tells nothing about the problem.

The reference's own CPU time then sets the limit (x4, between 1 and 4 s) and
the test budget (8 s of reference CPU, 1 MiB of test data). Only CPU time is
measured, so the limit does not depend on how loaded the build box is.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from reliquary_competitive_code.judge import TestCase, output_cap, outputs_match, run_test
from reliquary_competitive_code.sources.common import SourceRow, problem_id

VALIDATION_TIME_LIMIT_S = 10.0
TIME_FACTOR = 4.0
MIN_TIME_LIMIT_S = 1.0
MAX_TIME_LIMIT_S = 4.0
MAX_REFERENCE_TEST_S = 2.0
REFERENCE_CPU_BUDGET_S = 8.0
TEST_BYTES_BUDGET = 1 << 20
MIN_TESTS = 5
MAX_REFERENCES_TRIED = 3
_SPLIT_SALT = "reliquary_competitive_code_v1"


@dataclass(frozen=True, slots=True)
class Curated:
    problem_id: str
    split: str
    statement: str
    tests: tuple[TestCase, ...]
    time_limit_s: float
    reference: str
    origin: dict = field(hash=False, compare=True)


@dataclass(frozen=True, slots=True)
class Rejection:
    problem_id: str
    reason: str


def split_of(pid: str) -> str:
    bucket = int(hashlib.sha256(f"{_SPLIT_SALT}:{pid}".encode()).hexdigest(), 16) % 1000
    if bucket < 950:
        return "train"
    return "eval" if bucket < 975 else "qualification"


def time_limit(slowest_cpu_s: float) -> float:
    return min(MAX_TIME_LIMIT_S, max(MIN_TIME_LIMIT_S, round(TIME_FACTOR * slowest_cpu_s, 2)))


def select_tests(tests: Sequence[TestCase], timings: Sequence[float]) -> tuple[int, ...]:
    if not tests:
        return ()
    by_size = sorted(range(len(tests)), key=lambda i: (len(tests[i].stdin), i))
    rest = sorted(
        by_size[1:-1],
        key=lambda i: hashlib.sha256(f"{i}:{tests[i].stdin}".encode()).hexdigest(),
    )
    keep, cpu, size = [], 0.0, 0
    # Largest first (the test most likely to catch a slow or wrong program),
    # then smallest (the edge case), then the rest in a stable hashed order.
    for i in dict.fromkeys([by_size[-1], by_size[0], *rest]):
        cost = len(tests[i].stdin.encode()) + len(tests[i].stdout.encode())
        if cpu + timings[i] > REFERENCE_CPU_BUDGET_S or size + cost > TEST_BYTES_BUDGET:
            continue
        keep.append(i)
        cpu += timings[i]
        size += cost
    return tuple(sorted(keep))


def _timings(code: str, tests: Sequence[TestCase]) -> list[float] | str:
    timings = []
    for test in tests:
        result = run_test(
            code, test.stdin,
            time_limit_s=VALIDATION_TIME_LIMIT_S, output_cap=output_cap(test.stdout),
        )
        if result.status != "ok":
            return result.status
        if not outputs_match(test.stdout, result.stdout):
            return "wrong_answer"
        if result.cpu_seconds > MAX_REFERENCE_TEST_S:
            return "too_slow"
        timings.append(result.cpu_seconds)
    return timings


def curate(row: SourceRow) -> Curated | Rejection:
    pid = problem_id(row.statement)
    if len(row.tests) < MIN_TESTS:
        return Rejection(pid, "too_few_tests")
    reference, timings, wrong, slow, tried = None, None, False, False, 0
    for code in row.references:
        if tried == MAX_REFERENCES_TRIED:
            break
        outcome = _timings(code, row.tests)
        if outcome == "forbidden_import":
            continue
        tried += 1
        if isinstance(outcome, list):
            if reference is None:
                reference, timings = code, outcome
        elif outcome == "wrong_answer":
            wrong = True
        elif outcome == "too_slow":
            slow = True
    if reference is None:
        if slow:
            return Rejection(pid, "reference_too_slow")
        # Disagreement only means something beside a reference that passes.
        return Rejection(pid, "no_passing_reference")
    if wrong:
        return Rejection(pid, "references_disagree")
    kept = select_tests(row.tests, timings)
    if len(kept) < MIN_TESTS:
        return Rejection(pid, "too_few_tests")
    return Curated(
        problem_id=pid,
        split=split_of(pid),
        statement=row.statement,
        tests=tuple(row.tests[i] for i in kept),
        time_limit_s=time_limit(max(timings[i] for i in kept)),
        reference=reference,
        origin={"source": row.source, "upstream_id": row.upstream_id},
    )
