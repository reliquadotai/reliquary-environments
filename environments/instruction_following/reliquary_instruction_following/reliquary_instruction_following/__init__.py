from reliquary_instruction_following.corpus import EXCLUDED_INSTRUCTION_IDS
from reliquary_instruction_following.environment import (
    ENVIRONMENT,
    InstructionFollowingEnvironment,
)

__all__ = ["InstructionFollowingTaskset"]


def __getattr__(name: str) -> object:
    """Import the Verifiers surface only when something asks for it.

    Verifiers is a large dependency and a training-side one. Replay, grading
    and the corpus need none of it, and a validator that only has to reproduce
    a reward should not have to install it to do so.
    """
    if name == "InstructionFollowingTaskset":
        from reliquary_instruction_following.taskset import InstructionFollowingTaskset

        return InstructionFollowingTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
