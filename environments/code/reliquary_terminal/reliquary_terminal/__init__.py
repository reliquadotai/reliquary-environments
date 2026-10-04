"""Terminal-Bench 2.1 and MiMo-V2.6's terminal tasks as one Reliquary environment."""

# `HarborEnv` is exported so verifiers pairs it with this taskset: it is what
# grades a task declaring a separate verifier -- every `train` task -- in a
# fresh box (see `verifiers.v1.tasksets.harbor.env`).
from verifiers.v1.tasksets.harbor.env import HarborEnv, HarborEnvConfig

# `TerminalHarness` is exported so it is `reliquary-terminal`'s default
# harness: verifiers picks a taskset package's own Harness subclass.
from reliquary_terminal.harness import TerminalHarness, TerminalHarnessConfig
from reliquary_terminal.taskset import TerminalConfig, TerminalTaskset

__all__ = [
    "HarborEnv",
    "HarborEnvConfig",
    "TerminalConfig",
    "TerminalHarness",
    "TerminalHarnessConfig",
    "TerminalTaskset",
]
