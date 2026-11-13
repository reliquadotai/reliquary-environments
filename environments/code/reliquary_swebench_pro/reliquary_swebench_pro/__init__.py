"""SWE-bench Pro V2 (Scale AI, public set) as a standalone Reliquary environment,
evaluation only.

verifiers pairs a taskset with its Env and Harness by scanning `__all__`: this package
exports one Taskset and neither (grading needs no Env subclass under the env norm). The
two functions are a signed-episode sandbox's conformance and reference hooks.
"""

from reliquary_swebench_pro.conformance import conformance_cases, reference_calls
from reliquary_swebench_pro.task import SweProData, SweProTask
from reliquary_swebench_pro.taskset import SweProConfig, SweProTaskset

__all__ = [
    "SweProConfig",
    "SweProData",
    "SweProTask",
    "SweProTaskset",
    "conformance_cases",
    "reference_calls",
]
