"""Deterministic counter-stream RNG shared by the task generators.

Vendored from reliquadotai/reliquary at f4a6a23ad883,
`reliquary/environment/records_tasks.py`, with only the imports it
needs. The core copy stays where it is: it is a consensus artifact,
never moved or rewritten, only versioned forward here.
"""

from __future__ import annotations

import hashlib
from typing import TypeVar

_T = TypeVar("_T")


class HashCounterRng:
    """Small SHA-256 counter stream with stable rejection sampling."""

    __slots__ = ("_seed", "_counter", "_buffer")

    def __init__(self, seed: bytes):
        self._seed = bytes(seed)
        self._counter = 0
        self._buffer = b""

    def _take(self, count: int) -> bytes:
        while len(self._buffer) < count:
            block = hashlib.sha256(
                self._seed + self._counter.to_bytes(8, "big")
            ).digest()
            self._counter += 1
            self._buffer += block
        result, self._buffer = self._buffer[:count], self._buffer[count:]
        return result

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper bound must be positive")
        byte_count = max(1, (upper.bit_length() + 7) // 8)
        limit = (1 << (8 * byte_count)) - ((1 << (8 * byte_count)) % upper)
        while True:
            candidate = int.from_bytes(self._take(byte_count), "big")
            if candidate < limit:
                return candidate % upper

    def choice(self, values: tuple[_T, ...] | list[_T]) -> _T:
        return values[self.randbelow(len(values))]

    def shuffle(self, values: list[_T]) -> None:
        for index in range(len(values) - 1, 0, -1):
            other = self.randbelow(index + 1)
            values[index], values[other] = values[other], values[index]


__all__ = ["HashCounterRng"]
