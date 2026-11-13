"""AutomationBench's 800 public tasks, frozen as upstream builds them.

`data/tasks.jsonl.gz` is written by `scripts/export_tasks.py` from
zapier/AutomationBench at `UPSTREAM_COMMIT`: each row is a task as upstream's
builders return it (noise injected, seeded by `example_id`), captured before
Hugging Face `Dataset` normalisation. The digest covers the decompressed body.

Splits, each in upstream's domain order and, within a domain, upstream's task
order:

| split | rows | what it is |
| --- | --- | --- |
| `eval` | 600 | the six scored domains, 100 each, in `DOMAINS` order: domain d is `[100*d, 100*d + 100)` |
| `qualification` | 200 | upstream's `simple` domain: one- and two-step tasks, never scored by upstream |
| `<domain>` | 100 / 200 | one domain alone, for local runs |

There is no `train` split: AutomationBench is a held-out benchmark here.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.resources
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

UPSTREAM = "zapier/AutomationBench"
UPSTREAM_VERSION = "1.0.6"
UPSTREAM_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"

TASKS_FILE = "data/tasks.jsonl.gz"
TASKS_SHA256 = "9b342647bd15d26019c2a740f67907f6f653cad269d064ce6a89691480d5dca0"
TASK_COUNT = 800

DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")
SIMPLE = "simple"
SPLITS = ("eval", "qualification", *DOMAINS, SIMPLE)
DOMAIN_SIZE = 100

# 503 of the 800 tasks seed no `meta.current_time`, and upstream then reads the
# host's wall clock. A pinned replay cannot: those tasks run at the moment the
# pinned upstream commit was made, which is when its authors last ran them.
DEFAULT_SCENARIO_TIME = datetime(2026, 8, 4, 21, 42, 13, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class Task:
    domain: str
    example_id: int
    name: str
    system: str
    user: str
    initial_state: dict[str, Any]
    assertions: tuple[dict[str, Any], ...]
    zapier_tools: tuple[str, ...]

    @property
    def scenario_time(self) -> datetime:
        raw = (self.initial_state.get("meta") or {}).get("current_time")
        if not raw:
            return DEFAULT_SCENARIO_TIME
        value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _row(raw: dict[str, Any]) -> Task:
    roles = [message["role"] for message in raw["prompt"]]
    if roles != ["system", "user"]:
        raise RuntimeError(f"{raw['info']['task_name']}: prompt roles {roles}")
    info = raw["info"]
    return Task(
        domain=raw["domain"],
        example_id=int(raw["example_id"]),
        name=info["task_name"],
        system=raw["prompt"][0]["content"],
        user=raw["prompt"][1]["content"],
        initial_state=info.get("initial_state") or {},
        assertions=tuple(info.get("assertions") or ()),
        zapier_tools=tuple(info.get("zapier_tools") or ()),
    )


@lru_cache(maxsize=1)
def tasks() -> tuple[Task, ...]:
    body = gzip.decompress(importlib.resources.files(__package__).joinpath(TASKS_FILE).read_bytes())
    digest = hashlib.sha256(body).hexdigest()
    if digest != TASKS_SHA256:
        raise RuntimeError(f"{TASKS_FILE} digest is {digest}, expected {TASKS_SHA256}")
    rows = tuple(_row(json.loads(line)) for line in body.decode("utf-8").splitlines())
    if len(rows) != TASK_COUNT or len({task.name for task in rows}) != TASK_COUNT:
        raise RuntimeError("the task file does not hold 800 distinctly named tasks")
    return rows


@lru_cache(maxsize=None)
def rows(split: str) -> tuple[Task, ...]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split == "eval":
        selected = tuple(task for domain in DOMAINS for task in tasks() if task.domain == domain)
        if len(selected) != DOMAIN_SIZE * len(DOMAINS):
            raise RuntimeError("eval does not hold 100 tasks per domain")
        return selected
    domain = SIMPLE if split == "qualification" else split
    return tuple(task for task in tasks() if task.domain == domain)


__all__ = [
    "DEFAULT_SCENARIO_TIME",
    "DOMAINS",
    "SPLITS",
    "TASK_COUNT",
    "Task",
    "UPSTREAM_COMMIT",
    "rows",
    "tasks",
]
