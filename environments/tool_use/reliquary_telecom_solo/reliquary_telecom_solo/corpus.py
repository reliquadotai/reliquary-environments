"""The 2,285 tickets, the world each one starts in, and the policy above both.

Everything the environment reads ships inside the wheel and is checked against
a digest at load. The corpus is the task: a ticket that arrived differently, a
starting database with one more bill in it, or a policy paragraph that changed
between a rollout and its verification would each make the same index address a
different problem, and a participant must not be able to discover that at
grading time.

Sources, all from τ²-bench `data/tau2/domains/telecom/`:

| shipped as | upstream | what it is |
| --- | --- | --- |
| `tasks.json.gz` | `tasks_full.json` | the 2,285 tickets and their assertions |
| `db.toml` | `db.toml` | the carrier's records before the episode |
| `user_db.toml` | `user_db.toml` | the phone before the episode |
| `main_policy_solo.md` | `main_policy_solo.md` | what the agent may do |
| `tech_support_workflow_solo.md` | `tech_support_workflow_solo.md` | how to diagnose |
| `tool_schemas.json` | generated | the 44 tools, as the agent sees them |
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.resources
import json
import re
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

DATA_DIR = "data"

# Each digest covers the bytes the loader actually parses — for the task file,
# the decompressed body rather than the gzip container, so that the same corpus
# recompressed at another level is still the same corpus.
TASKS_FILE = f"{DATA_DIR}/tasks.json.gz"
TASKS_SHA256 = "37e562e1ae3242577407e1303b1548bc64e7ea68e37d36173e6747990ceaf8a4"
TASK_COUNT = 2285

DB_FILE = f"{DATA_DIR}/db.toml"
DB_SHA256 = "562d647ef9d7df8df91eafd8ee76036e707c8f9e32aaedc4a7fde06975aea2c0"
USER_DB_FILE = f"{DATA_DIR}/user_db.toml"
USER_DB_SHA256 = "4886107ea0c16f8e16d74bef91cfb557de61822afe9361f5a9b241806aea2e4c"

MAIN_POLICY_FILE = f"{DATA_DIR}/main_policy_solo.md"
MAIN_POLICY_SHA256 = "781296ae5419169cb64aac0c4fa2aa6129c78f3c1592f64abd80df6fbabb7042"
WORKFLOW_POLICY_FILE = f"{DATA_DIR}/tech_support_workflow_solo.md"
WORKFLOW_POLICY_SHA256 = (
    "b174d1f9705b5d7df49468daf99560a5633f86d23e10499a1166a7753706a412"
)

TOOL_SCHEMAS_FILE = f"{DATA_DIR}/tool_schemas.json"
TOOL_SCHEMAS_SHA256 = (
    "92124e2900cbdc2820c4944494d8146daf862b73cd37a66365c51ae61d9276d9"
)
TOOL_COUNT = 44

# The two reward components telecom tasks actually name. τ² defines three more
# — DB, COMMUNICATE, NL_ASSERTION — and no telecom task uses any of them, which
# is the reason this domain can be graded without a model: DB compares hashes
# against a replayed reference trajectory, COMMUNICATE greps prose the agent
# said to a user who is not here, and NL_ASSERTION is an LLM judge.
REWARD_TYPES = ("ENV_ASSERTION", "ACTION")

SPLITS = ("train", "eval", "qualification")
# Shares rather than thirds, and drawn inside each task family rather than over
# the corpus as a whole. The families are lopsided — 1,984 MMS tickets against
# 47 service tickets — and 10% of 47 taken by a hash could land anywhere
# between 2 and 9, which would move the family mix of the small splits by
# several points. Slicing each family separately makes the mix exact.
_SPLIT_BOUNDS = ((80, "train"), (90, "eval"), (100, "qualification"))
_SPLIT_SALT = "reliquary_telecom_solo_v1"

_FAMILY = re.compile(r"^\[(?P<family>[^\]]+)\]")


def _resource(name: str) -> Any:
    return importlib.resources.files(__package__).joinpath(name)


def _checked(name: str, expected: str, *, gzipped: bool = False) -> bytes:
    body = _resource(name).read_bytes()
    if gzipped:
        body = gzip.decompress(body)
    digest = hashlib.sha256(body).hexdigest()
    if digest != expected:
        raise RuntimeError(f"{name} digest is {digest}, expected {expected}")
    return body


@dataclass(frozen=True, slots=True)
class EnvCall:
    """One call on a toolkit, named rather than invoked.

    Both the scenario setup and the marking are lists of these. `env_type`
    says which toolkit — the carrier's or the phone's — and `func_name` is
    resolved by attribute lookup, not through the tool registry, which is what
    lets a scenario call `unseat_sim_card` and an assertion call
    `assert_can_send_mms` while neither is a tool the agent can reach.
    """

    env_type: str
    func_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EnvAssertion(EnvCall):
    """One predicate over the finished world, and the answer it wants."""

    assert_value: bool = True


@dataclass(frozen=True, slots=True)
class GoldAction:
    """One call from the reference solution.

    For 2,253 of the 2,285 tasks this is a path, not a requirement: the reward
    reads the world the agent left behind, and any route there scores the same.
    For the other 32 the task names `ACTION` in its reward basis, and then each
    entry here has to be matched by something the agent actually called.
    """

    action_id: str
    requestor: str
    name: str
    arguments: dict[str, Any]
    compare_args: tuple[str, ...] | None

    def matched_by(self, name: str, arguments: dict[str, Any]) -> bool:
        """Upstream's `Action.compare_with_tool_call`, argument-for-argument.

        An empty `compare_args` means the name alone decides — which is how the
        32 transfer tasks are written, since the summary the agent hands over
        is prose and prose cannot be compared for equality.
        """
        if self.name != name:
            return False
        compare = (
            tuple(arguments.keys()) if self.compare_args is None else self.compare_args
        )
        if not compare:
            return True
        return {key: value for key, value in arguments.items() if key in compare} == {
            key: value for key, value in self.arguments.items() if key in compare
        }


@dataclass(frozen=True, slots=True)
class Task:
    """One telecom ticket, in the only shape solo mode needs."""

    key: str
    ticket: str
    setup: tuple[EnvCall, ...]
    assertions: tuple[EnvAssertion, ...]
    gold: tuple[GoldAction, ...]
    reward_basis: tuple[str, ...]

    @property
    def family(self) -> str:
        """The bracketed prefix of the task id: what kind of fault this is."""
        match = _FAMILY.match(self.key)
        if match is None:
            raise ValueError(f"task id has no family prefix: {self.key}")
        return match.group("family")


def _env_call(record: dict[str, Any]) -> EnvCall:
    return EnvCall(
        env_type=record["env_type"],
        func_name=record["func_name"],
        arguments=dict(record.get("arguments") or {}),
    )


def _assertion(record: dict[str, Any]) -> EnvAssertion:
    return EnvAssertion(
        env_type=record["env_type"],
        func_name=record["func_name"],
        arguments=dict(record.get("arguments") or {}),
        assert_value=bool(record.get("assert_value", True)),
    )


def _gold(record: dict[str, Any]) -> GoldAction:
    compare = record.get("compare_args")
    return GoldAction(
        action_id=record["action_id"],
        requestor=record.get("requestor", "assistant"),
        name=record["name"],
        arguments=dict(record.get("arguments") or {}),
        compare_args=None if compare is None else tuple(compare),
    )


def _parse(record: dict[str, Any]) -> Task:
    ticket = (record.get("ticket") or "").strip()
    if not ticket:
        # Solo mode works from the ticket and nothing else, so a task without
        # one has no statement of the problem. Telecom is the only τ² domain
        # where this never happens; it is checked rather than assumed.
        raise ValueError(f"task {record['id']} carries no ticket")
    criteria = record.get("evaluation_criteria") or {}
    basis = tuple(criteria.get("reward_basis") or ())
    unknown = sorted(set(basis) - set(REWARD_TYPES))
    if unknown:
        # A basis this package cannot compute would score every answer the
        # same, which looks from the outside exactly like a policy that solved
        # nothing.
        raise ValueError(f"task {record['id']} names reward types {unknown}")
    initial = record.get("initial_state") or {}
    if initial.get("initialization_data") or initial.get("message_history"):
        # Neither is reachable here: the first needs `update_pydantic_model_
        # with_dict`, which this package does not vendor, and the second is a
        # conversation, which solo mode does not have.
        raise ValueError(f"task {record['id']} carries an unsupported initial state")
    return Task(
        key=record["id"],
        ticket=ticket,
        setup=tuple(
            _env_call(call) for call in (initial.get("initialization_actions") or [])
        ),
        assertions=tuple(
            _assertion(item) for item in (criteria.get("env_assertions") or [])
        ),
        gold=tuple(_gold(item) for item in (criteria.get("actions") or [])),
        reward_basis=basis,
    )


@lru_cache(maxsize=1)
def tasks() -> tuple[Task, ...]:
    """The packaged tasks, in file order, checked against their pin."""
    records = json.loads(_checked(TASKS_FILE, TASKS_SHA256, gzipped=True))
    parsed = tuple(_parse(record) for record in records)
    if len(parsed) != TASK_COUNT:
        raise RuntimeError(f"corpus holds {len(parsed)} tasks, expected {TASK_COUNT}")
    if len({task.key for task in parsed}) != len(parsed):
        raise RuntimeError("corpus holds duplicate task ids")
    return parsed


@lru_cache(maxsize=1)
def databases() -> tuple[dict[str, Any], dict[str, Any]]:
    """The carrier's records and the phone, as they stand before any setup."""
    return (
        tomllib.loads(_checked(DB_FILE, DB_SHA256).decode("utf-8")),
        tomllib.loads(_checked(USER_DB_FILE, USER_DB_SHA256).decode("utf-8")),
    )


@lru_cache(maxsize=1)
def policy() -> str:
    """The two solo policy documents, joined the way upstream joins them.

    The solo variants, not their dual-control siblings: the dual-control ones
    tell the agent to ask the user to do things, and in solo mode there is
    nobody to ask. The workflow document rather than the manual, because
    upstream's `TELECOM_TECH_SUPPORT_POLICY_MANUAL_SOLO_PATH` points at the
    dual-control manual — a copy-paste in upstream's own path table — so the
    manual has no solo variant to ship.
    """
    main = _checked(MAIN_POLICY_FILE, MAIN_POLICY_SHA256).decode("utf-8")
    workflow = _checked(WORKFLOW_POLICY_FILE, WORKFLOW_POLICY_SHA256).decode("utf-8")
    return (
        "<main_policy>\n"
        + main
        + "\n</main_policy>\n"
        + "<tech_support_policy>\n"
        + workflow
        + "\n</tech_support_policy>"
    )


@lru_cache(maxsize=1)
def tool_schemas() -> tuple[dict[str, Any], ...]:
    """The 44 tools the agent is offered, frozen rather than reflected.

    Generated once from the vendored toolkits with upstream's own schema code,
    then shipped as data. Reflecting it at import time would have meant a
    third-party docstring parser in the runtime dependency list, and a tool
    description that changes with a library upgrade is a prompt that changes
    with a library upgrade. `test_tool_schemas_match_the_toolkits` is what
    keeps the frozen copy honest.
    """
    schemas = json.loads(_checked(TOOL_SCHEMAS_FILE, TOOL_SCHEMAS_SHA256))
    if len(schemas) != TOOL_COUNT:
        raise RuntimeError(f"{len(schemas)} tool schemas, expected {TOOL_COUNT}")
    return tuple(schemas)


def split_of(key: str) -> str:
    """Which split a task belongs to, taken from its identity and its family.

    Hashing the task id rather than slicing the file keeps a task in the split
    it has always been in whatever order the corpus arrives in, and the family
    prefix bounds the draw so that the mix is preserved exactly rather than on
    average — `test_splits_preserve_the_family_mix` measures it.
    """
    try:
        position, total = _family_order()[key]
    except KeyError:
        raise ValueError(f"unknown task id: {key}") from None
    for bound, split in _SPLIT_BOUNDS:
        if position < (total * bound) // 100:
            return split
    raise AssertionError("split bounds must end at 100")


@lru_cache(maxsize=1)
def _family_order() -> dict[str, tuple[int, int]]:
    """Each task's rank within its family, and how big that family is."""
    families: dict[str, list[str]] = {}
    for task in tasks():
        families.setdefault(task.family, []).append(task.key)
    ranks: dict[str, tuple[int, int]] = {}
    for keys in families.values():
        ordered = sorted(
            keys,
            key=lambda key: hashlib.sha256(
                f"{_SPLIT_SALT}:{key}".encode("utf-8")
            ).digest(),
        )
        for position, key in enumerate(ordered):
            ranks[key] = (position, len(ordered))
    return ranks


@lru_cache(maxsize=None)
def rows(split: str = "train") -> tuple[Task, ...]:
    """The tasks of one split, in corpus order."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return tuple(task for task in tasks() if split_of(task.key) == split)


__all__ = [
    "DB_SHA256",
    "EnvAssertion",
    "EnvCall",
    "GoldAction",
    "REWARD_TYPES",
    "SPLITS",
    "TASKS_SHA256",
    "TASK_COUNT",
    "TOOL_COUNT",
    "Task",
    "USER_DB_SHA256",
    "databases",
    "policy",
    "rows",
    "split_of",
    "tasks",
    "tool_schemas",
]
