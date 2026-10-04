"""Run a program on stdin/stdout tests and decide whether it passes."""

from reliquary_competitive_code.judge.compare import outputs_match
from reliquary_competitive_code.judge.runner import (
    RunResult,
    TestCase,
    Verdict,
    judge,
    output_cap,
    run_test,
)

__all__ = ["RunResult", "TestCase", "Verdict", "judge", "output_cap", "outputs_match", "run_test"]
