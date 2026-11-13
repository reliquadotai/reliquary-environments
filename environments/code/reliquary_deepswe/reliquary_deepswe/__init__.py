"""DeepSWE v1.1 (Datacurve) as a standalone Reliquary environment, evaluation only.

verifiers pairs a taskset with its Env and Harness by scanning `__all__`: this package
exports one Taskset and neither (grading needs no Env subclass under the env norm). The
two functions are a signed-episode sandbox's conformance and reference hooks.
"""

from reliquary_deepswe.conformance import conformance_cases, reference_calls
from reliquary_deepswe.task import DeepSWEData, DeepSWETask
from reliquary_deepswe.taskset import DeepSWEConfig, DeepSWETaskset

__all__ = [
    "DeepSWEConfig",
    "DeepSWEData",
    "DeepSWETask",
    "DeepSWETaskset",
    "conformance_cases",
    "reference_calls",
]
