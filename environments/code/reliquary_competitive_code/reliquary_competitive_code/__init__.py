"""Competitive programming graded on hidden stdin/stdout tests."""

__all__ = ["CompetitiveCodeTaskset"]


def __getattr__(name: str) -> object:
    """Import the Verifiers surface only when something asks for it.

    Replay, grading and the build need none of Verifiers, and a validator that
    only has to reproduce a reward should not have to install it to do so.
    """
    if name == "CompetitiveCodeTaskset":
        from reliquary_competitive_code.taskset import CompetitiveCodeTaskset

        return CompetitiveCodeTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
