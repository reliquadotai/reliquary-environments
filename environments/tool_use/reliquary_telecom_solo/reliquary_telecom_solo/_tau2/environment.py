"""The two toolkits joined into one world, and one namespace over both.

Vendored from τ²-bench: `TelecomEnvironment.sync_tools` from
`src/tau2/domains/telecom/environment.py` verbatim, and from
`src/tau2/environment/environment.py` the solo-mode dispatch of
`make_tool_call`, the `validate_solo_mode` name check, `get_response`'s
error handling and `to_json_str`.

What is not here is what solo mode removes or what this package replaces:
`set_state` and its trajectory replay (this package owns the episode loop and
so never re-derives state from recorded messages), the user/assistant requestor
split (one actor), `run_env_function_call`'s `getattr` over a toolkit — kept,
but reachable only from grading and scenario setup, never from a tool call —
and the dual-control policy files.

`sync_tools` is the join between the two databases, and it runs after every
tool call: the carrier's record of whether a line is active decides whether the
phone in the user's hand can see the network, and a bill the user has paid is
marked paid on the carrier's side. Upstream calls it from `get_response`, so
the same call in the same place here.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from reliquary_telecom_solo._tau2.data_model import LineStatus, TelecomDB
from reliquary_telecom_solo._tau2.tools import TelecomTools
from reliquary_telecom_solo._tau2.user_data_model import PaymentRequest, TelecomUserDB
from reliquary_telecom_solo._tau2.user_tools import TelecomUserTools


class TelecomSoloWorld:
    """One telecom scenario, mid-episode."""

    def __init__(self, db: TelecomDB, user_db: TelecomUserDB) -> None:
        self.tools = TelecomTools(db)
        self.user_tools = TelecomUserTools(user_db)
        self.validate_namespace()
        self.sync_tools()

    @property
    def db(self) -> TelecomDB:
        return self.tools.db

    @property
    def user_db(self) -> TelecomUserDB:
        return self.user_tools.db

    def validate_namespace(self) -> None:
        """Solo mode hands the agent both toolsets, so the names must not clash.

        Upstream checks the same thing in `Environment.validate_solo_mode`. It
        is checked at construction rather than trusted, because a name in two
        toolkits would make a tool call mean two things, and which one it meant
        would depend on the order this file happens to look them up in.
        """
        overlap = set(self.tools._tool_names()) & set(self.user_tools._tool_names())
        if overlap:
            raise ValueError(f"Tool names overlap: {sorted(overlap)}")

    def has_tool(self, name: str) -> bool:
        return self.tools.has_tool(name) or self.user_tools.has_tool(name)

    def toolkit_of(self, name: str):
        if self.user_tools.has_tool(name):
            return self.user_tools
        if self.tools.has_tool(name):
            return self.tools
        raise ValueError(f"Tool '{name}' not found.")

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Run one tool and let the world settle, upstream's dispatch order.

        The user toolkit is consulted first, matching
        `Environment.make_tool_call` in solo mode. With the namespaces checked
        disjoint the order cannot matter, which is the point of checking.
        """
        result = self.toolkit_of(name).use_tool(name, **arguments)
        self.sync_tools()
        return result

    def run_function(self, env_type: str, func_name: str, arguments: dict[str, Any]) -> Any:
        """Run any method of either toolkit, tool or not.

        This is how a scenario is set up and how it is marked. It is not
        reachable from a tool call: `call` goes through `use_tool`, which
        refuses a name that carries no `@is_tool`.
        """
        if env_type == "user":
            toolkit = self.user_tools
        elif env_type == "assistant":
            toolkit = self.tools
        else:
            raise ValueError(f"Invalid environment type: {env_type}")
        result = getattr(toolkit, func_name)(**arguments)
        self.sync_tools()
        return result

    def digest(self) -> str:
        """One hash over both databases, upstream's hash on each side."""
        return f"{self.db.get_hash()}:{self.user_db.get_hash()}"

    def sync_tools(self) -> None:
        """
        Sync the tools with the user's surroundings.
        If the line is roaming enabled, then the user is allowed to roam.
        """
        if self.user_tools.db.surroundings.phone_number is None:
            return
        phone_number = self.user_tools.db.surroundings.phone_number
        line = self.tools._get_line_by_phone(phone_number)
        if line is None:
            raise ValueError(
                f"Wrong scenario, line not found for phone number: {phone_number}"
            )
        # Check if the line is active
        if line.status == LineStatus.ACTIVE:
            self.user_tools.db.surroundings.line_active = True
        else:
            self.user_tools.db.surroundings.line_active = False

        # Check if the line is roaming enabled
        if line.roaming_enabled:
            self.user_tools.db.surroundings.roaming_allowed = True
        else:
            self.user_tools.db.surroundings.roaming_allowed = False

        # Check if the user has exceeded their data usage limit
        plan = self.tools._get_plan_by_id(line.plan_id)
        if plan is None:
            raise ValueError(
                f"Wrong scenario, invalid plan id ({line.plan_id}) for the phone number {phone_number}"
            )
        if line.data_used_gb >= plan.data_limit_gb + line.data_refueling_gb:
            self.user_tools.db.surroundings.mobile_data_usage_exceeded = True
        else:
            self.user_tools.db.surroundings.mobile_data_usage_exceeded = False

        # Check if the user has paid a bill
        current_payment_request = self.user_tools.db.surroundings.payment_request
        if current_payment_request is not None:
            if current_payment_request.paid:
                self.tools._set_bill_to_paid(current_payment_request.bill_id)
                self.user_tools.db.surroundings.payment_request = None

        # Check if the user has a payment request
        current_payment_request = self.user_tools.db.surroundings.payment_request
        if (
            current_payment_request is None
        ):  # If there already is a payment request, do nothing
            customer = self.tools.get_customer_by_phone(phone_number)
            bills = self.tools._get_bills_awaiting_payment(customer)
            if len(bills) != 0:
                bill = bills[0]
                self.user_tools.db.surroundings.payment_request = PaymentRequest(
                    bill_id=bill.bill_id, amount_due=bill.total_due
                )


def to_json_str(resp: Any) -> str:
    """Render a tool's return value the way upstream renders it.

    This is what the agent reads, so it is the transcript. Kept verbatim,
    including `default=str` and the `(int, float, bool)` branch that stringifies
    a scalar before the surrounding structure is serialised: the point is not
    that the shape is elegant, it is that a validator rebuilding the transcript
    gets the same bytes.
    """

    def _process(resp: Any) -> Any:
        if isinstance(resp, BaseModel):
            return resp.model_dump()
        elif isinstance(resp, str):
            return resp
        elif resp is None:
            return resp
        elif isinstance(resp, (int, float, bool)):
            return str(resp)
        elif isinstance(resp, list):
            return [_process(item) for item in resp]
        elif isinstance(resp, tuple):
            return tuple(_process(item) for item in resp)
        elif isinstance(resp, dict):
            return {k: _process(v) for k, v in resp.items()}
        elif isinstance(resp, (datetime, date)):
            return resp.isoformat()
        else:
            raise ValueError(f"Unsupported type: {type(resp)}")

    if not isinstance(resp, str):
        return json.dumps(_process(resp), default=str)
    return resp


__all__ = ["TelecomSoloWorld", "to_json_str"]
