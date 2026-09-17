from reliquary_telecom_solo.corpus import SPLITS, TASK_COUNT
from reliquary_telecom_solo.environment import (
    ENVIRONMENT,
    MAX_TURNS,
    STOP_TOOL,
    TelecomSoloEnvironment,
)

__all__ = ["TelecomSoloTaskset"]


def __getattr__(name: str) -> object:
    """Import the Verifiers surface only when something asks for it.

    Verifiers is a large dependency and a training-side one. Replay, grading
    and the corpus need none of it, and a validator that only has to reproduce
    a reward should not have to install it to do so.
    """
    if name == "TelecomSoloTaskset":
        from reliquary_telecom_solo.taskset import TelecomSoloTaskset

        return TelecomSoloTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
