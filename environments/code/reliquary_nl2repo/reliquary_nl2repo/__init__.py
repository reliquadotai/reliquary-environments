"""NL2Repo-Bench as a standalone Reliquary environment, evaluation only.

verifiers pairs a taskset with its Env and Harness by scanning `__all__`: this package
exports one Taskset and neither (grading needs no Env subclass under the env norm). The
two functions are a signed-episode sandbox's conformance and reference hooks.
"""

from reliquary_nl2repo.conformance import conformance_cases, reference_calls
from reliquary_nl2repo.task import NL2RepoData, NL2RepoTask
from reliquary_nl2repo.taskset import NL2RepoConfig, NL2RepoTaskset

__all__ = [
    "NL2RepoConfig",
    "NL2RepoData",
    "NL2RepoTask",
    "NL2RepoTaskset",
    "conformance_cases",
    "reference_calls",
]
