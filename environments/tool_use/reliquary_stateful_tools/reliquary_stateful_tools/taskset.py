"""Native Verifiers v1 and synchronous compatibility surfaces for stateful tools."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal

from pydantic import Field
import verifiers.v1 as vf


GENERATOR_VERSION = "reliquary-stateful-tools-generator-v2"
TASK_COUNT = 1 << 31
MAX_ACTION_BYTES = 16 * 1024
FAMILIES = ("address_update", "refund", "support_note")
SPLITS = ("train", "eval", "qualification")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _strict_json_object(value: str) -> dict[str, Any]:
    try:
        if len(value.encode("utf-8")) > MAX_ACTION_BYTES:
            raise ValueError("tool arguments exceed the byte limit")
    except UnicodeError as error:
        raise ValueError("tool arguments are not valid UTF-8") from error

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-finite JSON number: {constant}")

    result = json.loads(
        value,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(result, dict):
        raise ValueError("tool arguments must be a JSON object")
    return result


def _action_fits(action: Mapping[str, Any]) -> bool:
    try:
        encoded = json.dumps(
            action,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return False
    return len(encoded) <= MAX_ACTION_BYTES


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "search_customers",
        "description": "Find customers whose email exactly matches the supplied email.",
        "parameters": _object({"email": {"type": "string"}}, ["email"]),
    },
    {
        "name": "get_customer",
        "description": "Get one customer by customer_id.",
        "parameters": _object(
            {"customer_id": {"type": "string"}}, ["customer_id"]
        ),
    },
    {
        "name": "list_orders",
        "description": "List a customer's orders, newest first.",
        "parameters": _object(
            {"customer_id": {"type": "string"}}, ["customer_id"]
        ),
    },
    {
        "name": "get_order",
        "description": "Get one order and its refundable amount.",
        "parameters": _object({"order_id": {"type": "string"}}, ["order_id"]),
    },
    {
        "name": "update_shipping_address",
        "description": "Update the shipping address for one pending order.",
        "parameters": _object(
            {
                "order_id": {"type": "string"},
                "address": {"type": "string"},
            },
            ["order_id", "address"],
        ),
    },
    {
        "name": "create_refund",
        "description": "Create an exact refund for one delivered order.",
        "parameters": _object(
            {
                "order_id": {"type": "string"},
                "amount_cents": {"type": "integer", "minimum": 1},
            },
            ["order_id", "amount_cents"],
        ),
    },
    {
        "name": "add_support_note",
        "description": "Add an audit-visible note to a customer account.",
        "parameters": _object(
            {
                "customer_id": {"type": "string"},
                "note": {"type": "string"},
            },
            ["customer_id", "note"],
        ),
    },
)


class _ToolFailure(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _normalize(index: int) -> int:
    return int(index) % TASK_COUNT


def _build_task(index: int, split: str) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    index = _normalize(index)
    digest = hashlib.sha256(
        f"{GENERATOR_VERSION}:{split}:{index}".encode("ascii")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:16], "big"))
    family = FAMILIES[index % len(FAMILIES)]
    customer_number = rng.randrange(100_000, 999_999)
    other_number = rng.randrange(100_000, 999_999)
    while other_number == customer_number:
        other_number = rng.randrange(100_000, 999_999)
    customer_id = f"cus_{customer_number}"
    other_customer_id = f"cus_{other_number}"
    order_number = rng.randrange(100_000, 999_999)
    other_order_number = rng.randrange(100_000, 999_999)
    while other_order_number == order_number:
        other_order_number = rng.randrange(100_000, 999_999)
    order_id = f"ord_{order_number}"
    other_order_id = f"ord_{other_order_number}"
    email = f"customer{customer_number}@example.test"
    refund_cents = rng.randrange(12, 90) * 100
    note = f"verified-request-{rng.randrange(1000, 9999)}"
    address = (
        f"{rng.randrange(10, 999)} Cedar Street, "
        f"Unit {rng.randrange(1, 40)}, Testville"
    )
    refund_id = f"ref_{order_id[4:]}"
    task_id = hashlib.sha256(
        f"reliquary_stateful_tools_v2:{GENERATOR_VERSION}:{split}:{index}".encode()
    ).hexdigest()

    rows = {
        "customers": [
            [customer_id, email, "Primary Customer"],
            [
                other_customer_id,
                f"other{other_number}@example.test",
                "Distractor",
            ],
        ],
        "orders": [
            [
                order_id,
                customer_id,
                "delivered" if family == "refund" else "pending",
                "12 Old Road, Testville",
                refund_cents,
                2,
            ],
            [
                other_order_id,
                other_customer_id,
                "pending",
                "99 Unrelated Avenue, Testville",
                2500,
                1,
            ],
        ],
    }

    if family == "address_update":
        prompt = (
            f"Find the customer with email {email}. Update the shipping address "
            f"of their pending order to '{address}'. Add the support note "
            f"'{note}', then reply with the customer ID and order ID."
        )
        expected = {
            "address": address,
            "note": note,
            "final_terms": [customer_id, order_id],
        }
        actions = [
            {"tool": "search_customers", "arguments": {"email": email}},
            {"tool": "list_orders", "arguments": {"customer_id": customer_id}},
            {
                "tool": "update_shipping_address",
                "arguments": {"order_id": order_id, "address": address},
            },
            {
                "tool": "add_support_note",
                "arguments": {"customer_id": customer_id, "note": note},
            },
            {"final": f"Updated {order_id} for {customer_id}."},
        ]
    elif family == "refund":
        prompt = (
            f"Find the customer with email {email}. Refund exactly "
            f"{refund_cents} cents on their latest delivered order, add the "
            f"support note '{note}', then reply with the refund ID and order ID."
        )
        expected = {
            "refund_id": refund_id,
            "refund_cents": refund_cents,
            "note": note,
            "final_terms": [refund_id, order_id],
        }
        actions = [
            {"tool": "search_customers", "arguments": {"email": email}},
            {"tool": "list_orders", "arguments": {"customer_id": customer_id}},
            {"tool": "get_order", "arguments": {"order_id": order_id}},
            {
                "tool": "create_refund",
                "arguments": {"order_id": order_id, "amount_cents": refund_cents},
            },
            {
                "tool": "add_support_note",
                "arguments": {"customer_id": customer_id, "note": note},
            },
            {"final": f"Created {refund_id} for {order_id}."},
        ]
    else:
        prompt = (
            f"Find the customer with email {email}, inspect their latest order, "
            f"add the exact support note '{note}', then reply with the customer "
            "ID, order ID, and current order status. Do not change the order."
        )
        expected = {
            "note": note,
            "status": "pending",
            "final_terms": [customer_id, order_id, "pending"],
        }
        actions = [
            {"tool": "search_customers", "arguments": {"email": email}},
            {"tool": "list_orders", "arguments": {"customer_id": customer_id}},
            {"tool": "get_order", "arguments": {"order_id": order_id}},
            {
                "tool": "add_support_note",
                "arguments": {"customer_id": customer_id, "note": note},
            },
            {"final": f"{customer_id} {order_id} is pending."},
        ]

    return {
        "id": task_id,
        "prompt": prompt,
        "tools": [dict(tool) for tool in TOOLS],
        "metadata": {
            "family": family,
            "generator_version": GENERATOR_VERSION,
            "generator_index": index,
            "split": split,
            "difficulty": 1 + index % 3,
        },
        "private": {
            "rows": rows,
            "target_customer_id": customer_id,
            "target_order_id": order_id,
            "other_customer_id": other_customer_id,
            "other_order_id": other_order_id,
            "expected": expected,
            "reference_actions": actions,
        },
    }


def _initial_state(task: Mapping[str, Any], seed: int) -> dict[str, Any]:
    rows = task["private"]["rows"]
    return {
        "seed": int(seed),
        "customers": {
            row[0]: {"customer_id": row[0], "email": row[1], "name": row[2]}
            for row in rows["customers"]
        },
        "orders": {
            row[0]: {
                "order_id": row[0],
                "customer_id": row[1],
                "status": row[2],
                "shipping_address": row[3],
                "refundable_cents": row[4],
                "created_seq": row[5],
            }
            for row in rows["orders"]
        },
        "refunds": {},
        "notes": [],
        "mutations": [],
        "invalid_actions": 0,
        "final_response": None,
        "closed": False,
    }


def _snapshot(state: Mapping[str, Any]) -> dict[str, list[list[Any]]]:
    return {
        "customers": [
            [row["customer_id"], row["email"], row["name"]]
            for row in sorted(state["customers"].values(), key=lambda row: row["customer_id"])
        ],
        "orders": [
            [
                row["order_id"],
                row["customer_id"],
                row["status"],
                row["shipping_address"],
                row["refundable_cents"],
                row["created_seq"],
            ]
            for row in sorted(state["orders"].values(), key=lambda row: row["order_id"])
        ],
        "refunds": [
            [row["refund_id"], row["order_id"], row["amount_cents"]]
            for row in sorted(state["refunds"].values(), key=lambda row: row["refund_id"])
        ],
        "notes": [
            [row["note_id"], row["customer_id"], row["note"]]
            for row in state["notes"]
        ],
    }


def _expected_snapshot(task: Mapping[str, Any]) -> dict[str, list[list[Any]]]:
    private = task["private"]
    expected = private["expected"]
    family = task["metadata"]["family"]
    order_id = private["target_order_id"]
    orders = [list(row) for row in private["rows"]["orders"]]
    if family == "address_update":
        next(row for row in orders if row[0] == order_id)[3] = expected["address"]
    refunds = (
        [[expected["refund_id"], order_id, expected["refund_cents"]]]
        if family == "refund"
        else []
    )
    return {
        "customers": sorted(
            [list(row) for row in private["rows"]["customers"]],
            key=lambda row: row[0],
        ),
        "orders": sorted(orders, key=lambda row: row[0]),
        "refunds": refunds,
        "notes": [[1, private["target_customer_id"], expected["note"]]],
    }


class StatefulToolsEnvironment:
    """Synchronous, JSON-shaped ABI used by Reliquary replay and local tests."""

    name = "reliquary_stateful_tools_v2"
    max_turns = 8
    validator_authoritative_reward = True

    def __init__(self, split: str = "train") -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split

    def __len__(self) -> int:
        return TASK_COUNT

    def task(self, index: int) -> dict[str, Any]:
        task = _build_task(index, self.split)
        return {key: task[key] for key in ("id", "prompt", "tools", "metadata")}

    def reset(self, index: int, seed: int) -> dict[str, Any]:
        return {"state": _initial_state(_build_task(index, self.split), seed), "events": []}

    @staticmethod
    def _arguments(arguments: Mapping[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
        if not isinstance(arguments, Mapping) or set(arguments) != set(required):
            raise _ToolFailure("INVALID_ARGUMENTS", f"expected arguments {required}")
        return dict(arguments)

    def call(
        self,
        state: dict[str, Any],
        tool: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            if state.get("closed"):
                raise _ToolFailure("CLOSED", "episode state is closed")
            if tool == "search_customers":
                args = self._arguments(arguments, ("email",))
                email = args["email"]
                if not isinstance(email, str):
                    raise _ToolFailure("INVALID_ARGUMENTS", "email must be a string")
                rows = sorted(
                    (
                        dict(row)
                        for row in state["customers"].values()
                        if row["email"] == email
                    ),
                    key=lambda row: row["customer_id"],
                )
                value = {"customers": rows}
            elif tool == "get_customer":
                args = self._arguments(arguments, ("customer_id",))
                customer_id = args["customer_id"]
                if not isinstance(customer_id, str):
                    raise _ToolFailure("INVALID_ARGUMENTS", "customer_id must be a string")
                row = state["customers"].get(customer_id)
                value = {"customer": None if row is None else dict(row)}
            elif tool == "list_orders":
                args = self._arguments(arguments, ("customer_id",))
                customer_id = args["customer_id"]
                if not isinstance(customer_id, str):
                    raise _ToolFailure("INVALID_ARGUMENTS", "customer_id must be a string")
                rows = sorted(
                    (
                        {
                            key: row[key]
                            for key in (
                                "order_id",
                                "status",
                                "shipping_address",
                                "refundable_cents",
                            )
                        }
                        for row in state["orders"].values()
                        if row["customer_id"] == customer_id
                    ),
                    key=lambda row: state["orders"][row["order_id"]]["created_seq"],
                    reverse=True,
                )
                value = {"orders": rows}
            elif tool == "get_order":
                args = self._arguments(arguments, ("order_id",))
                order_id = args["order_id"]
                if not isinstance(order_id, str):
                    raise _ToolFailure("INVALID_ARGUMENTS", "order_id must be a string")
                row = state["orders"].get(order_id)
                value = {
                    "order": None
                    if row is None
                    else {
                        key: row[key]
                        for key in (
                            "order_id",
                            "customer_id",
                            "status",
                            "shipping_address",
                            "refundable_cents",
                        )
                    }
                }
            elif tool == "update_shipping_address":
                args = self._arguments(arguments, ("order_id", "address"))
                order_id, address = args["order_id"], args["address"]
                if not isinstance(order_id, str) or not isinstance(address, str):
                    raise _ToolFailure("INVALID_ARGUMENTS", "order_id and address must be strings")
                if not 1 <= len(address) <= 512:
                    raise _ToolFailure("INVALID_ARGUMENTS", "address must contain 1..512 characters")
                row = state["orders"].get(order_id)
                if row is None:
                    raise _ToolFailure("NOT_FOUND", "order not found")
                if row["status"] != "pending":
                    raise _ToolFailure("INVALID_STATE", "only pending orders may be updated")
                if row["shipping_address"] == address:
                    return {
                        "ok": True,
                        "updated": True,
                        "order_id": order_id,
                        "idempotent": True,
                    }
                row["shipping_address"] = address
                state["mutations"].append(["update_address", order_id])
                value = {"updated": True, "order_id": order_id}
            elif tool == "create_refund":
                args = self._arguments(arguments, ("order_id", "amount_cents"))
                order_id, amount = args["order_id"], args["amount_cents"]
                if not isinstance(order_id, str) or isinstance(amount, bool) or not isinstance(amount, int):
                    raise _ToolFailure("INVALID_ARGUMENTS", "order_id must be a string and amount_cents an integer")
                row = state["orders"].get(order_id)
                if row is None:
                    raise _ToolFailure("NOT_FOUND", "order not found")
                existing = state["refunds"].get(order_id)
                if existing and existing["amount_cents"] == amount:
                    return {
                        "ok": True,
                        "created": True,
                        "refund_id": existing["refund_id"],
                        "idempotent": True,
                    }
                if row["status"] != "delivered":
                    raise _ToolFailure("INVALID_STATE", "only delivered orders may be refunded")
                if amount != row["refundable_cents"]:
                    raise _ToolFailure("INVALID_AMOUNT", "refund must equal the refundable amount")
                refund_id = f"ref_{order_id[4:]}"
                if existing:
                    raise _ToolFailure("DUPLICATE", "order already refunded")
                state["refunds"][order_id] = {
                    "refund_id": refund_id,
                    "order_id": order_id,
                    "amount_cents": amount,
                }
                state["mutations"].append(["create_refund", order_id])
                value = {"created": True, "refund_id": refund_id}
            elif tool == "add_support_note":
                args = self._arguments(arguments, ("customer_id", "note"))
                customer_id, note = args["customer_id"], args["note"]
                if not isinstance(customer_id, str) or not isinstance(note, str):
                    raise _ToolFailure("INVALID_ARGUMENTS", "customer_id and note must be strings")
                if customer_id not in state["customers"]:
                    raise _ToolFailure("NOT_FOUND", "customer not found")
                if not 1 <= len(note) <= 1024:
                    raise _ToolFailure("INVALID_ARGUMENTS", "note must contain 1..1024 characters")
                if any(
                    row["customer_id"] == customer_id and row["note"] == note
                    for row in state["notes"]
                ):
                    return {
                        "ok": True,
                        "created": True,
                        "customer_id": customer_id,
                        "idempotent": True,
                    }
                state["notes"].append(
                    {
                        "note_id": len(state["notes"]) + 1,
                        "customer_id": customer_id,
                        "note": note,
                    }
                )
                state["mutations"].append(["add_note", customer_id])
                value = {"created": True, "customer_id": customer_id}
            else:
                raise _ToolFailure("UNKNOWN_TOOL", f"unknown tool: {tool}")
            return {"ok": True, **value}
        except _ToolFailure as error:
            state["invalid_actions"] = int(state.get("invalid_actions", 0)) + 1
            return {"ok": False, "error": {"code": error.code, "message": str(error)}}

    def step(
        self,
        index: int,
        state: dict[str, Any],
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        if isinstance(action, Mapping) and set(action) == {"final"} and isinstance(action["final"], str):
            state["final_response"] = action["final"]
            result = {"accepted": True}
            name, done, reason = "final", True, "finished"
        elif isinstance(action, Mapping) and set(action) == {"tool", "arguments"} and isinstance(action["tool"], str):
            result = self.call(state, action["tool"], action["arguments"])
            name, done = action["tool"], not result["ok"]
            reason = "invalid_action" if done else None
        else:
            state["invalid_actions"] = int(state.get("invalid_actions", 0)) + 1
            result = {"ok": False, "error": {"code": "INVALID_ACTION", "message": "expected tool call or final response"}}
            name, done, reason = "__invalid_action__", True, "invalid_action"
        return {
            "state": state,
            "events": [{"role": "tool", "name": name, "content": _canonical_json(result)}],
            "done": done,
            "termination_reason": reason,
        }

    def grade(
        self,
        index: int,
        state: Mapping[str, Any],
        actions: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        task = _build_task(index, self.split)
        private, expected = task["private"], task["private"]["expected"]
        family = task["metadata"]["family"]
        customer_id, order_id = private["target_customer_id"], private["target_order_id"]
        final = state.get("final_response") or next(
            (action["final"] for action in reversed(actions) if set(action) == {"final"} and isinstance(action["final"], str)),
            "",
        )
        notes = [row["note"] for row in state["notes"] if row["customer_id"] == customer_id]
        checks = [
            {"name": "required_support_note", "passed": expected["note"] in notes, "weight": 1.0, "detail": ""}
        ]
        if family == "address_update":
            passed = state["orders"].get(order_id, {}).get("shipping_address") == expected["address"]
            checks.append({"name": "shipping_address_updated", "passed": passed, "weight": 2.0, "detail": ""})
        elif family == "refund":
            refund = state["refunds"].get(order_id)
            passed = bool(refund and refund["refund_id"] == expected["refund_id"] and refund["amount_cents"] == expected["refund_cents"])
            checks.append({"name": "refund_created", "passed": passed, "weight": 2.0, "detail": ""})
        else:
            passed = state["orders"].get(order_id, {}).get("status") == expected["status"]
            checks.append({"name": "order_status_preserved", "passed": passed, "weight": 2.0, "detail": ""})
        expected_mutations = {
            "address_update": [["update_address", order_id], ["add_note", customer_id]],
            "refund": [["create_refund", order_id], ["add_note", customer_id]],
            "support_note": [["add_note", customer_id]],
        }[family]
        snapshot = _snapshot(state)
        checks.extend(
            [
                {"name": "final_response_identifiers", "passed": all(term.lower() in final.lower() for term in expected["final_terms"]), "weight": 1.0, "detail": ""},
                {"name": "finished_explicitly", "passed": bool(final), "weight": 0.5, "detail": ""},
                {"name": "mutations_exact", "passed": sorted(state["mutations"]) == sorted(expected_mutations), "weight": 2.0, "detail": ""},
                {"name": "no_invalid_tool_calls", "passed": state["invalid_actions"] == 0, "weight": 0.5, "detail": ""},
                {"name": "database_state_exact", "passed": snapshot == _expected_snapshot(task), "weight": 3.0, "detail": ""},
            ]
        )
        success = all(check["passed"] for check in checks)
        return {
            "reward": float(success),
            "success": success,
            "checks": checks,
            "state_digest": _sha256_json(snapshot),
            "environment_error": None,
        }

    def replay(
        self,
        index: int,
        actions: Sequence[Mapping[str, Any]],
        seed: int = 0,
    ) -> dict[str, Any]:
        reset = self.reset(index, seed)
        state, events, applied = reset["state"], [], []
        reason = "turn_limit"
        for action in actions[: self.max_turns]:
            applied.append(dict(action))
            result = self.step(index, state, action)
            events.extend(result["events"])
            if result["done"]:
                reason = result["termination_reason"]
                break
        return {
            "task": self.task(index),
            "state": state,
            "events": events,
            "actions": applied,
            "termination_reason": reason,
            "reward": self.grade(index, state, applied),
        }

    @staticmethod
    def close(state: dict[str, Any]) -> None:
        state["closed"] = True


class StatefulToolsState(vf.State):
    world: dict[str, Any] = Field(default_factory=dict)


class StatefulToolsToolset(vf.Toolset[vf.ToolsetConfig, StatefulToolsState]):
    TOOL_PREFIX = None

    def _run(self, name: str, **arguments: Any) -> str:
        return _canonical_json(StatefulToolsEnvironment().call(self.state.world, name, arguments))

    @vf.tool
    def search_customers(self, email: str) -> str:
        """Find customers whose email exactly matches the supplied email."""
        return self._run("search_customers", email=email)

    @vf.tool
    def get_customer(self, customer_id: str) -> str:
        """Get one customer by customer_id."""
        return self._run("get_customer", customer_id=customer_id)

    @vf.tool
    def list_orders(self, customer_id: str) -> str:
        """List a customer's orders, newest first."""
        return self._run("list_orders", customer_id=customer_id)

    @vf.tool
    def get_order(self, order_id: str) -> str:
        """Get one order and its refundable amount."""
        return self._run("get_order", order_id=order_id)

    @vf.tool
    def update_shipping_address(self, order_id: str, address: str) -> str:
        """Update the shipping address for one pending order."""
        return self._run("update_shipping_address", order_id=order_id, address=address)

    @vf.tool
    def create_refund(self, order_id: str, amount_cents: int) -> str:
        """Create an exact refund for one delivered order."""
        return self._run("create_refund", order_id=order_id, amount_cents=amount_cents)

    @vf.tool
    def add_support_note(self, customer_id: str, note: str) -> str:
        """Add an audit-visible note to a customer account."""
        return self._run("add_support_note", customer_id=customer_id, note=note)


class StatefulToolsData(vf.TaskData):
    task_id: str
    family: str
    generator_version: str
    difficulty: int
    split: Literal["train", "eval", "qualification"]


class StatefulToolsTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class StatefulToolsTask(vf.Task[StatefulToolsData, StatefulToolsState, StatefulToolsTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.task_id

    @classmethod
    def toolsets(cls, config: StatefulToolsTaskConfig) -> list[vf.Toolset]:
        return [StatefulToolsToolset(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        trace.state.world = StatefulToolsEnvironment(self.data.split).reset(
            self.data.idx or 0, 0
        )["state"]

    @staticmethod
    def _actions(trace: vf.Trace) -> list[dict[str, Any]] | None:
        """Recover only model-authored actions from the final trace branch."""
        branches = trace.branches
        if not branches:
            return None
        messages = [
            node.message
            for node in branches[-1].nodes
            if node.sampled and isinstance(node.message, vf.AssistantMessage)
        ]
        if not messages:
            return None

        actions: list[dict[str, Any]] = []
        call_ids: set[str] = set()
        allowed = {tool["name"] for tool in TOOLS}
        for position, message in enumerate(messages):
            calls = message.tool_calls or []
            if not calls and position != len(messages) - 1:
                return None
            for call in calls:
                if (
                    call.type != "function"
                    or not call.id
                    or call.id in call_ids
                    or call.name not in allowed
                ):
                    return None
                call_ids.add(call.id)
                try:
                    arguments = _strict_json_object(call.arguments)
                except (TypeError, ValueError, RecursionError):
                    return None
                action = {"tool": call.name, "arguments": arguments}
                if not _action_fits(action):
                    return None
                actions.append(action)

        last = messages[-1]
        if not last.tool_calls:
            if not isinstance(last.content, str):
                return None
            action = {"final": last.content}
            if not _action_fits(action):
                return None
            actions.append(action)
        return actions

    @vf.reward(weight=1.0)
    async def verified_outcome(self, trace: vf.Trace) -> float:
        actions = self._actions(trace)
        if actions is None:
            return 0.0
        result = StatefulToolsEnvironment(self.data.split).replay(
            self.data.idx or 0, actions
        )
        return float(result["reward"]["reward"])

    async def validate(self, runtime: vf.Runtime) -> bool:
        del runtime
        environment = StatefulToolsEnvironment(self.data.split)
        source = _build_task(self.data.idx or 0, self.data.split)
        good = environment.replay(
            self.data.idx or 0, source["private"]["reference_actions"]
        )
        bad = environment.replay(self.data.idx or 0, [{"final": "done"}])
        return good["reward"]["reward"] == 1.0 and bad["reward"]["reward"] == 0.0


class StatefulToolsConfig(vf.TasksetConfig):
    split: Literal["train", "eval", "qualification"] = "train"
    task: StatefulToolsTaskConfig = StatefulToolsTaskConfig()


class StatefulToolsTaskset(vf.Taskset[StatefulToolsTask, StatefulToolsConfig]):
    INFINITE = True

    def load(self) -> Iterator[StatefulToolsTask]:
        environment = StatefulToolsEnvironment(self.config.split)
        for index in itertools.count():
            task = environment.task(index)
            metadata = task["metadata"]
            yield StatefulToolsTask(
                StatefulToolsData(
                    idx=index,
                    prompt=task["prompt"],
                    network_allow=[],
                    task_id=task["id"],
                    family=metadata["family"],
                    generator_version=metadata["generator_version"],
                    difficulty=metadata["difficulty"],
                    split=self.config.split,
                ),
                self.config.task,
            )


if __name__ == "__main__":
    StatefulToolsToolset.run()
