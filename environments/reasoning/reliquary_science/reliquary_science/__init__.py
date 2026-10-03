from reliquary_science.corpus import VIRTUAL_LENGTH
from reliquary_science.environment import ENVIRONMENT, ScienceEnvironment

__all__ = ["ScienceTaskset"]


def __getattr__(name: str) -> object:
    """Import the Verifiers surface only when something asks for it.

    Verifiers is a large dependency and a training-side one. Replay, grading
    and the corpus need none of it, and a validator that only has to reproduce
    a reward should not have to install it to do so.
    """
    if name == "ScienceTaskset":
        from reliquary_science.taskset import ScienceTaskset

        return ScienceTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
