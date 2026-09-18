from reliquary_dapo_math.corpus import VIRTUAL_LENGTH
from reliquary_dapo_math.environment import ENVIRONMENT, DapoMathEnvironment

__all__ = ["DapoMathTaskset"]


def __getattr__(name: str) -> object:
    """Import the Verifiers surface only when something asks for it.

    Verifiers is a large dependency and a training-side one. Replay, grading
    and the corpus need none of it, and a validator that only has to reproduce
    a reward should not have to install it to do so.
    """
    if name == "DapoMathTaskset":
        from reliquary_dapo_math.taskset import DapoMathTaskset

        return DapoMathTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
