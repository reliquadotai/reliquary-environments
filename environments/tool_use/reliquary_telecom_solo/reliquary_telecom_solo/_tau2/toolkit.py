"""The toolkit base every telecom tool hangs off, and the frozen clock.

Vendored from τ²-bench: `src/tau2/environment/toolkit.py` (the `is_tool`
decorator, `ToolType`, and the parts of `ToolKitBase` the solo telecom path
reaches), `src/tau2/environment/db.py` (`DB`), `src/tau2/utils/utils.py`
(`get_dict_hash`), `src/tau2/utils/pydantic_utils.py` (`BaseModelNoExtra`) and
`src/tau2/domains/telecom/utils.py` (`get_now`, `get_today`).

Changes from upstream:

- `ToolKitType`, the metaclass that collects decorated methods, is replaced by
  a walk over the class at first use. The metaclass rebuilt a closure per
  subclass and its `super()` chain misbehaved under multiple inheritance;
  neither the behaviour nor the speed is needed for two flat toolkits.
- Discoverable tools, tool statistics, `update_db` and `get_db_json_schema` are
  gone with the code paths that used them. `update_db` was the only caller of
  `update_pydantic_model_with_dict`, which reached for `addict`: no telecom
  task carries `initialization_data`, so no task could reach it.
- `loguru` is gone. What it logged was informational, and a logger is a
  configuration a participant could differ on.

The clock is frozen in upstream source rather than being made frozen here, and
that is the single most important property this package inherits: every
`get_today()` in a telecom tool answers 2025-02-25, so a bill period computed
in a rollout today is the same bill period computed by a validator next month.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

TOOL_ATTR = "__tool__"
TOOL_TYPE_ATTR = "__tool_type__"
MUTATES_STATE_ATTR = "__mutates_state__"


def get_now() -> datetime.datetime:
    """The moment every telecom tool believes it is."""
    # assume now is 2025-02-25 12:08:00
    return datetime.datetime(2025, 2, 25, 12, 8, 0)


def get_today() -> datetime.date:
    """The day every telecom tool believes it is."""
    # assume today is 2025-02-25
    return datetime.date(2025, 2, 25)


class BaseModelNoExtra(BaseModel):
    model_config = ConfigDict(extra="forbid")


def get_dict_hash(obj: dict) -> str:
    """Upstream's database hash, kept byte-for-byte compatible."""
    hash_string = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(hash_string.encode()).hexdigest()


class DB(BaseModelNoExtra):
    """Base class for the two telecom databases."""

    def get_hash(self) -> str:
        return get_dict_hash(self.model_dump())


class ToolType(str, Enum):
    """What a tool does, from the caller's point of view.

    Only `WRITE` is load-bearing here: upstream skips non-mutating calls when
    it replays a trajectory, and this package reports the split so a caller can
    tell a diagnostic call from one that changed the world.
    """

    READ = "read"
    WRITE = "write"
    THINK = "think"
    GENERIC = "generic"


def is_tool(tool_type: ToolType = ToolType.READ, mutates_state: bool | None = None):
    """Mark a method as callable by the agent.

    A method without this decorator is reachable only from inside the package —
    which is what keeps the assertion functions, and the `break_*` helpers that
    set a scenario up, out of the agent's hands.
    """
    if mutates_state is None:
        mutates_state = tool_type == ToolType.WRITE

    def decorator(func):
        setattr(func, TOOL_ATTR, True)
        setattr(func, TOOL_TYPE_ATTR, tool_type)
        setattr(func, MUTATES_STATE_ATTR, mutates_state)
        return func

    return decorator


class ToolKitBase:
    """A bundle of decorated methods over one database."""

    def __init__(self, db: Any = None) -> None:
        self.db = db

    @classmethod
    def _tool_names(cls) -> tuple[str, ...]:
        """Every decorated method on the class, in definition order.

        Definition order rather than sorted: the order reaches the agent as the
        order of the tool list, and a set iteration order would be a second
        thing two processes could disagree about.
        """
        names: list[str] = []
        for klass in reversed(cls.__mro__):
            for name, method in vars(klass).items():
                if isinstance(method, property):
                    method = method.fget
                if getattr(method, TOOL_ATTR, False) and name not in names:
                    names.append(name)
        return tuple(names)

    @property
    def tools(self) -> dict[str, Callable]:
        return {name: getattr(self, name) for name in self._tool_names()}

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_names()

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        if not self.has_tool(tool_name):
            raise ValueError(f"Tool '{tool_name}' not found.")
        return getattr(self, tool_name)(**kwargs)

    def tool_type(self, tool_name: str) -> ToolType:
        return getattr(getattr(self, tool_name), TOOL_TYPE_ATTR)

    def tool_mutates_state(self, tool_name: str) -> bool:
        return getattr(getattr(self, tool_name), MUTATES_STATE_ATTR, True)

    def get_db_hash(self) -> str:
        return get_dict_hash(self.db.model_dump())


__all__ = [
    "BaseModelNoExtra",
    "DB",
    "ToolKitBase",
    "ToolType",
    "get_dict_hash",
    "get_now",
    "get_today",
    "is_tool",
]
