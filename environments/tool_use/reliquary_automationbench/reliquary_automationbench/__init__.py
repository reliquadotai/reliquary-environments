"""AutomationBench (zapier/AutomationBench) as a deterministic, replayable episode environment."""

from reliquary_automationbench.corpus import DOMAINS, SPLITS, TASK_COUNT
from reliquary_automationbench.environment import (
    ENVIRONMENT,
    MAX_TURNS,
    AutomationBenchEnvironment,
)

__all__ = [
    "DOMAINS",
    "ENVIRONMENT",
    "MAX_TURNS",
    "SPLITS",
    "TASK_COUNT",
    "AutomationBenchEnvironment",
    "AutomationBenchTaskset",
]


def __getattr__(name: str) -> object:
    """Import the Verifiers surface only when something asks for it: replay and
    grading need none of it."""
    if name == "AutomationBenchTaskset":
        from reliquary_automationbench.taskset import AutomationBenchTaskset

        return AutomationBenchTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
