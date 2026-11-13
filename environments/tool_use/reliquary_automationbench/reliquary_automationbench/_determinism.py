"""The clock and the random source the vendored upstream code runs on.

Upstream mints ids with `uuid4()` and `random.choices` (about a hundred sites)
and stamps records with `datetime.now()` (about three hundred), so two runs of
the same actions differ in every id and timestamp an agent reads back. The
vendoring script rewrites each of those calls to the functions here, which
read a per-episode context: a `random.Random` seeded from the task, the
episode seed and the call's position, and a clock pinned to the task's
scenario time. The same actions then give the same ids, the same timestamps
and the same transcript, in any process.

A call outside an episode context raises: a nondeterministic value that
slipped in unnoticed would surface as a replay mismatch on a validator, far
from its cause.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import random as _random
import uuid as _uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class _Context:
    rng: _random.Random
    clock: datetime  # timezone-aware, UTC
    ticks: int = field(default=0)


_CONTEXT: contextvars.ContextVar[_Context | None] = contextvars.ContextVar(
    "reliquary_automationbench_context", default=None
)


def _current() -> _Context:
    context = _CONTEXT.get()
    if context is None:
        raise RuntimeError("AutomationBench world code ran outside an episode context")
    return context


@contextlib.contextmanager
def episode_context(*, label: str, scenario_time: datetime, offset_seconds: int) -> Iterator[None]:
    """Run world code with a seeded random source and a pinned clock.

    `label` names the stream (task, episode seed, call position); the clock
    reads `scenario_time + offset_seconds`, plus one microsecond per reading
    so that two records created in one call are still ordered.
    """
    if scenario_time.tzinfo is None:
        raise ValueError("scenario time must be timezone-aware")
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:16], "big")
    token = _CONTEXT.set(
        _Context(
            rng=_random.Random(seed),
            clock=scenario_time.astimezone(timezone.utc) + timedelta(seconds=offset_seconds),
        )
    )
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def _instant() -> datetime:
    context = _current()
    value = context.clock + timedelta(microseconds=context.ticks)
    context.ticks += 1
    return value


def _det_now(tz: tzinfo | None = None) -> datetime:
    """`datetime.now`: naive (as on a host whose zone is UTC) unless a zone is given."""
    value = _instant()
    return value.replace(tzinfo=None) if tz is None else value.astimezone(tz)


def _det_utcnow() -> datetime:
    return _instant().replace(tzinfo=None)


def _det_today() -> date:
    return _instant().date()


def _det_time() -> float:
    return _instant().timestamp()


def _det_timestamp(value: datetime) -> float:
    """`value.timestamp()`, reading a naive value as UTC rather than the host's zone."""
    return (value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)).timestamp()


class _RandomModule:
    """The slice of the `random` module upstream calls, on the episode's stream."""

    @staticmethod
    def choices(population: Sequence[T], weights: Any = None, *, cum_weights: Any = None, k: int = 1) -> list[T]:
        return _current().rng.choices(population, weights, cum_weights=cum_weights, k=k)

    @staticmethod
    def randint(a: int, b: int) -> int:
        return _current().rng.randint(a, b)

    @staticmethod
    def choice(sequence: Sequence[T]) -> T:
        return _current().rng.choice(sequence)

    @staticmethod
    def random() -> float:
        return _current().rng.random()


class _UuidModule:
    """`uuid` with `uuid4` on the episode's stream; everything else is the real module."""

    UUID = _uuid.UUID

    @staticmethod
    def uuid4() -> _uuid.UUID:
        return _uuid.UUID(int=_current().rng.getrandbits(128), version=4)


random = _RandomModule()
uuid = _UuidModule()
uuid4 = _UuidModule.uuid4

__all__ = [
    "_det_now",
    "_det_time",
    "_det_timestamp",
    "_det_today",
    "_det_utcnow",
    "episode_context",
    "random",
    "uuid",
    "uuid4",
]
