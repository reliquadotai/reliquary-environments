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
measured, so the limit does not depend on how loaded the build box is. A run
the host starved (`harness_overload`) is retried once; a second one rejects
the problem as `harness_overload`, so the dataset never depends on box load.

v1.1 sources (`Policy`) add a path for problems whose tests the upstream
vouches for (`SourceRow.upstream_tests`): when no Python reference passes, but
none gave a wrong answer either, the problem is kept as `upstream_tests_only`,
its tests chosen by the byte and count budget alone and its limit the
upstream (C++-calibrated) one times `python_time_factor`. A Python reference
that passes still validates the tests as in v1, and the limit is never below
that upstream-derived one. A reference that runs to a wrong answer drops the
problem even then: an accepted submission failing the judge's own tests means
the tests or the comparison are not what we think (a 2to3 conversion is the
exception, since its integer `/` may be wrong where the original was right).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from reliquary_competitive_code.judge import TestCase, output_cap, outputs_match, run_test
from reliquary_competitive_code.judge.runner import HARNESS_OVERLOAD
from reliquary_competitive_code.sources.common import (
    PY2TO3,
    PYTHON_REFERENCE,
    UPSTREAM_TESTS_ONLY,
    SourceRow,
    problem_id,
)

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
class Policy:
    """What a build keeps of a problem's tests and how it sets the limit.

    `test_bytes`: stdin plus expected stdout of the kept tests, at most.
    `reference_cpu_s`: the reference's CPU over the kept tests, at most.
    `max_tests`: kept tests, at most (None: no cap, as v1).
    `python_time_factor`: multiplies an upstream limit for Python; None
    ignores upstream limits and the upstream-tests path (v1).
    """

    test_bytes: int = TEST_BYTES_BUDGET
    reference_cpu_s: float = REFERENCE_CPU_BUDGET_S
    max_tests: int | None = None
    python_time_factor: float | None = None
    max_time_limit_s: float = MAX_TIME_LIMIT_S


V1_POLICY = Policy()


@dataclass(frozen=True, slots=True)
class Curated:
    problem_id: str
    split: str
    statement: str
    tests: tuple[TestCase, ...]
    time_limit_s: float
    reference: str
    origin: dict = field(hash=False, compare=True)
    reference_kind: str = PYTHON_REFERENCE
    # The reference's slowest CPU over the kept tests (None without one): what
    # the Python time factor is measured on. Not published.
    reference_cpu_s: float | None = field(default=None, compare=False)


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


def upstream_time_limit(upstream_s: float, policy: Policy) -> float:
    """An upstream limit made a Python one: x `python_time_factor`, between
    1 s and the policy's ceiling."""
    assert policy.python_time_factor is not None
    raw = round(policy.python_time_factor * upstream_s, 2)
    return min(policy.max_time_limit_s, max(MIN_TIME_LIMIT_S, raw))


def select_tests(
    tests: Sequence[TestCase], timings: Sequence[float], policy: Policy = V1_POLICY
) -> tuple[int, ...]:
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
        if cpu + timings[i] > policy.reference_cpu_s or size + cost > policy.test_bytes:
            continue
        if policy.max_tests is not None and len(keep) == policy.max_tests:
            break
        keep.append(i)
        cpu += timings[i]
        size += cost
    return tuple(sorted(keep))


def _timings(code: str, tests: Sequence[TestCase]) -> list[float] | str:
    timings = []
    for test in tests:
        for _ in range(2):  # one retry when the host starved the run
            result = run_test(
                code, test.stdin,
                time_limit_s=VALIDATION_TIME_LIMIT_S, output_cap=output_cap(test.stdout),
            )
            if result.status != HARNESS_OVERLOAD:
                break
        if result.status != "ok":
            return result.status
        if not outputs_match(test.stdout, result.stdout):
            return "wrong_answer"
        if result.cpu_seconds > MAX_REFERENCE_TEST_S:
            return "too_slow"
        timings.append(result.cpu_seconds)
    return timings


def curate(row: SourceRow, policy: Policy = V1_POLICY) -> Curated | Rejection:
    pid = problem_id(row.statement)
    if len(row.tests) < MIN_TESTS:
        return Rejection(pid, "too_few_tests")
    upstream = policy.python_time_factor is not None and row.upstream_tests
    if upstream and row.time_limit_s is None:
        return Rejection(pid, "no_upstream_time_limit")
    reference, timings, wrong, original_wrong, slow, tried = None, None, False, False, False, 0
    for code in row.references:
        if tried == MAX_REFERENCES_TRIED:
            break
        outcome = _timings(code, row.tests)
        if outcome == "forbidden_import":
            continue
        if outcome == HARNESS_OVERLOAD:
            return Rejection(pid, HARNESS_OVERLOAD)
        tried += 1
        if isinstance(outcome, list):
            if reference is None:
                reference, timings = code, outcome
        elif outcome == "wrong_answer":
            wrong = True
            original_wrong = original_wrong or code not in row.converted
        elif outcome == "too_slow":
            slow = True
    origin = {"source": row.source, "upstream_id": row.upstream_id}
    if reference is None:
        if upstream:
            if original_wrong:
                return Rejection(pid, "reference_wrong_answer")
            kept = select_tests(row.tests, [0.0] * len(row.tests), policy)
            if len(kept) < MIN_TESTS:
                return Rejection(pid, "too_few_tests")
            return Curated(
                problem_id=pid,
                split=split_of(pid),
                statement=row.statement,
                tests=tuple(row.tests[i] for i in kept),
                time_limit_s=upstream_time_limit(row.time_limit_s, policy),
                reference="",
                origin=origin,
                reference_kind=UPSTREAM_TESTS_ONLY,
            )
        if slow:
            return Rejection(pid, "reference_too_slow")
        # Disagreement only means something beside a reference that passes.
        return Rejection(pid, "no_passing_reference")
    if wrong:
        return Rejection(pid, "references_disagree")
    kept = select_tests(row.tests, timings, policy)
    if len(kept) < MIN_TESTS:
        return Rejection(pid, "too_few_tests")
    slowest = max(timings[i] for i in kept)
    limit = time_limit(slowest)
    if policy.python_time_factor is not None and row.time_limit_s is not None:
        limit = max(limit, upstream_time_limit(row.time_limit_s, policy))
    return Curated(
        problem_id=pid,
        split=split_of(pid),
        statement=row.statement,
        tests=tuple(row.tests[i] for i in kept),
        time_limit_s=limit,
        reference=reference,
        origin=origin,
        reference_kind=PY2TO3 if reference in row.converted else PYTHON_REFERENCE,
        reference_cpu_s=slowest,
    )


__all__ = ["V1_POLICY", "Curated", "Policy", "Rejection", "curate", "select_tests", "split_of", "time_limit"]
