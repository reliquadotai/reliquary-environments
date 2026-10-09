"""Terminal-Bench 2.1 and MiMo-V2.6's terminal tasks as one Reliquary environment."""

# Exactly one subclass each of Env, Taskset and Harness is exported: verifiers
# pairs them with this taskset by scanning `__all__`. `TerminalEnv` grades
# every `train` task in a fresh box (see `verifiers.v1.tasksets.harbor.env`);
# `TerminalHarness` is the default harness of `reliquary-terminal`. The two
# functions are a signed-episode sandbox's conformance and reference hooks (plain data).
from verifiers.v1.tasksets.harbor.env import HarborEnvConfig

from reliquary_terminal.conformance import conformance_cases, reference_calls
from reliquary_terminal.env import TerminalEnv
from reliquary_terminal.grading import TerminalTask
from reliquary_terminal.harness import TerminalHarness, TerminalHarnessConfig
from reliquary_terminal.taskset import TerminalConfig, TerminalTaskset

__all__ = [
    "HarborEnvConfig",
    "TerminalConfig",
    "TerminalEnv",
    "TerminalHarness",
    "TerminalHarnessConfig",
    "TerminalTask",
    "TerminalTaskset",
    "conformance_cases",
    "reference_calls",
]
