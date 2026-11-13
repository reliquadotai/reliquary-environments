"""Terminal-Bench 2.1 as a standalone Reliquary environment, evaluation only.

verifiers pairs a taskset with its Env and Harness by scanning `__all__`: this package
exports one Taskset and neither (grading needs no Env subclass under the env norm). The
two functions are a signed-episode sandbox's conformance and reference hooks.
"""

from reliquary_terminal_bench.conformance import conformance_cases, reference_calls
from reliquary_terminal_bench.task import TerminalBenchData, TerminalBenchTask
from reliquary_terminal_bench.taskset import TerminalBenchConfig, TerminalBenchTaskset

__all__ = [
    "TerminalBenchConfig",
    "TerminalBenchData",
    "TerminalBenchTask",
    "TerminalBenchTaskset",
    "conformance_cases",
    "reference_calls",
]
