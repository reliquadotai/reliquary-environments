from reliquary_general.corpus import BLOCKS, MODES, RENDERERS, SPLITS, segments
from reliquary_general.environment import (
    ENVIRONMENT,
    GENERATION_SYSTEM,
    GeneralPromptsEnvironment,
    Ungraded,
)

__all__ = [
    "BLOCKS",
    "ENVIRONMENT",
    "GENERATION_SYSTEM",
    "MODES",
    "RENDERERS",
    "SPLITS",
    "GeneralPromptsEnvironment",
    "GeneralTaskset",
    "Ungraded",
    "segments",
]


def __getattr__(name: str) -> object:
    """Import the Verifiers surface only when something asks for it.

    Replay, grading and the corpus need none of Verifiers, and a validator that
    only has to reproduce a verdict should not have to install it to do so.
    """
    if name == "GeneralTaskset":
        from reliquary_general.taskset import GeneralTaskset

        return GeneralTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
