"""τ²-bench telecom, solo, as a native Verifiers taskset.

A support ticket arrives and one actor works it: the same actor reads the
carrier's records and toggles the phone, because in solo mode τ² hands the
agent both toolsets and deletes the user simulator that used to hold the
second one. The reward is a set of predicates over the world the agent left
behind, so it is exact, and nothing in the loop is a model.

The toolset here is not a second implementation of the tools. Every call
replays through `TelecomSoloEnvironment`, which is what a validator runs, so
the two surfaces cannot drift apart: the state carried between calls is the
list of calls, and the world is what that list builds.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field

from reliquary_telecom_solo.corpus import tool_schemas
from reliquary_telecom_solo.environment import STOP_TOOL, TelecomSoloEnvironment

TOOL_NAMES = frozenset(
    schema["function"]["name"] for schema in tool_schemas()
) - {STOP_TOOL}


class TelecomSoloState(vf.State):
    """What travels between tool calls: which task, and what has been called.

    The world itself does not travel. It is two pydantic object graphs and the
    state channel carries JSON, and rebuilding it from the call log is not a
    workaround but the honest shape: a call log that rebuilds the same world
    every time is the property this environment is claiming, and letting the
    training surface depend on it means a bug there is a failing test here.
    """

    index: int = 0
    split: Literal["train", "eval", "qualification"] = "train"
    calls: list[dict[str, Any]] = Field(default_factory=list)
    ended: bool = False


class TelecomSoloToolset(vf.Toolset[vf.ToolsetConfig, TelecomSoloState]):
    TOOL_PREFIX = None

    def _run(self, name: str, **arguments: Any) -> str:
        """Replay every call so far and hand back the last observation.

        Termination is `TelecomSoloEnvironment.step`'s decision, not this
        method's, so that an episode which ended here ended in exactly the
        place the validator's replay will find it: a call accepted after `done`
        would be a call the agent read and the reward never saw.
        """
        environment = TelecomSoloEnvironment(self.state.split)
        if self.state.ended:
            return "Error: this episode has already ended"
        self.state.calls.append({"tool": name, "arguments": arguments})
        world = environment.reset(self.state.index)["state"]
        for call in self.state.calls:
            result = environment.step(self.state.index, world, call)
        # Only the last call can have ended the episode: an earlier one would
        # have been refused above.
        self.state.ended = bool(result["done"]) or len(self.state.calls) >= (
            environment.max_turns
        )
        return result["events"][-1]["content"]

    @vf.tool
    def done(self) -> str:
        """Call this function when you are done with the task."""
        return self._run(STOP_TOOL)

    @vf.tool
    def get_customer_by_phone(self, phone_number: str) -> str:
        """Finds a customer by their primary contact or line phone number."""
        return self._run("get_customer_by_phone", phone_number=phone_number)

    @vf.tool
    def get_customer_by_id(self, customer_id: str) -> str:
        """Retrieves a customer directly by their unique ID."""
        return self._run("get_customer_by_id", customer_id=customer_id)

    @vf.tool
    def get_customer_by_name(self, full_name: str, dob: str) -> str:
        """Searches for customers by name and DOB. May return multiple matches if names are similar,

        DOB helps disambiguate.
        """
        return self._run("get_customer_by_name", full_name=full_name, dob=dob)

    @vf.tool
    def get_details_by_id(self, id: str) -> str:
        """Retrieves the details for a given ID.

        The ID must be a valid ID for a Customer, Line, Device, Bill, or Plan.
        """
        return self._run("get_details_by_id", id=id)

    @vf.tool
    def suspend_line(self, customer_id: str, line_id: str, reason: str) -> str:
        """Suspends a specific line (max 6 months).

        Checks: Line status must be Active.
        Logic: Sets line status to Suspended, records suspension_start_date.
        """
        return self._run("suspend_line", customer_id=customer_id, line_id=line_id, reason=reason)

    @vf.tool
    def resume_line(self, customer_id: str, line_id: str) -> str:
        """Resumes a suspended line.

        Checks: Line status must be Suspended or Pending Activation.
        Logic: Sets line status to Active, clears suspension_start_date.
        """
        return self._run("resume_line", customer_id=customer_id, line_id=line_id)

    @vf.tool
    def get_bills_for_customer(self, customer_id: str, limit: int = 12) -> str:
        """Retrieves a list of the customer's bills, most recent first."""
        return self._run("get_bills_for_customer", customer_id=customer_id, limit=limit)

    @vf.tool
    def send_payment_request(self, customer_id: str, bill_id: str) -> str:
        """Sends a payment request to the customer for a specific bill.

        Checks:
            - Customer exists
            - Bill exists and belongs to the customer
            - No other bills are already awaiting payment for this customer
        Logic: Sets bill status to AWAITING_PAYMENT and notifies customer.
        Warning: This method does not check if the bill is already PAID.
        Always check the bill status before calling this method.
        """
        return self._run("send_payment_request", customer_id=customer_id, bill_id=bill_id)

    @vf.tool
    def get_data_usage(self, customer_id: str, line_id: str) -> str:
        """Retrieves current billing cycle data usage for a line, including data

        refueling amount, data limit, and cycle end date.
        """
        return self._run("get_data_usage", customer_id=customer_id, line_id=line_id)

    @vf.tool
    def enable_roaming(self, customer_id: str, line_id: str) -> str:
        """Enables international roaming on a line."""
        return self._run("enable_roaming", customer_id=customer_id, line_id=line_id)

    @vf.tool
    def disable_roaming(self, customer_id: str, line_id: str) -> str:
        """Disables international roaming on a line."""
        return self._run("disable_roaming", customer_id=customer_id, line_id=line_id)

    @vf.tool
    def transfer_to_human_agents(self, summary: str) -> str:
        """Transfer the user to a human agent, with a summary of the user's issue.

        Only transfer if
         -  the user explicitly asks for a human agent
         -  given the policy and the available tools, you cannot solve the user's issue.
        """
        return self._run("transfer_to_human_agents", summary=summary)

    @vf.tool
    def refuel_data(self, customer_id: str, line_id: str, gb_amount: float) -> str:
        """Refuels data for a specific line, adding to the customer's bill.

        Checks: Line status must be Active, Customer owns the line.
        Logic: Adds data to the line and charges customer based on the plan's refueling rate.
        """
        return self._run("refuel_data", customer_id=customer_id, line_id=line_id, gb_amount=gb_amount)

    @vf.tool
    def check_status_bar(self) -> str:
        """Shows what icons are currently visible in your phone's status bar (the area at the top of the screen). Displays network signal strength, mobile data status (enabled, disabled, data saver), Wi-Fi status, and battery level.
        """
        return self._run("check_status_bar")

    @vf.tool
    def check_network_status(self) -> str:
        """Checks your phone's connection status to cellular networks and Wi-Fi. Shows airplane mode status, signal strength, network type, whether mobile data is enabled, and whether data roaming is enabled.
        """
        return self._run("check_network_status")

    @vf.tool
    def check_network_mode_preference(self) -> str:
        """Shows the current network mode preference."""
        return self._run("check_network_mode_preference")

    @vf.tool
    def set_network_mode_preference(self, mode: str) -> str:
        """Changes the type of cellular network your phone prefers to connect to (e.g., 5G, LTE/4G, 3G). Higher-speed networks (LTE/5G) provide faster data but may use more battery.
        """
        return self._run("set_network_mode_preference", mode=mode)

    @vf.tool
    def run_speed_test(self) -> str:
        """Measures your current internet connection speed (download speed). Provides information about connection quality and what activities it can support.
        """
        return self._run("run_speed_test")

    @vf.tool
    def toggle_airplane_mode(self) -> str:
        """Toggles Airplane Mode ON or OFF. When ON, it disconnects all wireless communications including cellular, Wi-Fi, and Bluetooth.

        Returns the new state of airplane_mode.
        """
        return self._run("toggle_airplane_mode")

    @vf.tool
    def check_sim_status(self) -> str:
        """Checks if your SIM card is working correctly and displays its current status. Shows if the SIM is active, missing, or locked with a PIN or PUK code.
        """
        return self._run("check_sim_status")

    @vf.tool
    def reseat_sim_card(self) -> str:
        """Simulates removing and reinserting your SIM card. This can help resolve recognition issues.
        """
        return self._run("reseat_sim_card")

    @vf.tool
    def toggle_data(self) -> str:
        """Toggles your phone's mobile data connection ON or OFF. Controls whether your phone can use cellular data for internet access when Wi-Fi is unavailable.

        Returns the new data connection status.
        """
        return self._run("toggle_data")

    @vf.tool
    def toggle_roaming(self) -> str:
        """Toggles Data Roaming ON or OFF. When ON, your phone can use data networks in areas outside your carrier's coverage.

        Returns the new data roaming status.
        """
        return self._run("toggle_roaming")

    @vf.tool
    def check_data_restriction_status(self) -> str:
        """Checks if your phone has any data-limiting features active. Shows if Data Saver mode is on.
        """
        return self._run("check_data_restriction_status")

    @vf.tool
    def toggle_data_saver_mode(self) -> str:
        """Toggles Data Saver mode ON or OFF. When ON, it reduces data usage, which may affect data speed.

        Returns the new data saver mode status.
        """
        return self._run("toggle_data_saver_mode")

    @vf.tool
    def check_apn_settings(self) -> str:
        """Checks the technical APN settings your phone uses to connect to your carrier's mobile data network. Shows current APN name and MMSC URL for picture messaging.
        """
        return self._run("check_apn_settings")

    @vf.tool
    def set_apn_settings(self, apn_settings: dict) -> str:
        """Sets the APN settings for the phone."""
        return self._run("set_apn_settings", apn_settings=apn_settings)

    @vf.tool
    def reset_apn_settings(self) -> str:
        """Resets your APN settings to the default settings."""
        return self._run("reset_apn_settings")

    @vf.tool
    def check_wifi_status(self) -> str:
        """Checks your Wi-Fi connection status. Shows if Wi-Fi is turned on, which network you're connected to (if any), and the signal strength.
        """
        return self._run("check_wifi_status")

    @vf.tool
    def toggle_wifi(self) -> str:
        """Toggles your phone's Wi-Fi radio ON or OFF. Controls whether your phone can discover and connect to wireless networks for internet access.

        Returns the new Wi-Fi status.
        """
        return self._run("toggle_wifi")

    @vf.tool
    def check_wifi_calling_status(self) -> str:
        """Checks if Wi-Fi Calling is enabled on your device. This feature allows you to make and receive calls over a Wi-Fi network instead of using the cellular network.
        """
        return self._run("check_wifi_calling_status")

    @vf.tool
    def toggle_wifi_calling(self) -> str:
        """Toggles Wi-Fi Calling ON or OFF. This feature allows you to make and receive calls over Wi-Fi instead of the cellular network, which can help in areas with weak cellular signal.

        Returns the new Wi-Fi Calling status.
        """
        return self._run("toggle_wifi_calling")

    @vf.tool
    def check_vpn_status(self) -> str:
        """Checks if you're using a VPN (Virtual Private Network) connection. Shows if a VPN is active, connected, and displays any available connection details.
        """
        return self._run("check_vpn_status")

    @vf.tool
    def connect_vpn(self) -> str:
        """Connects to your VPN (Virtual Private Network)."""
        return self._run("connect_vpn")

    @vf.tool
    def disconnect_vpn(self) -> str:
        """Disconnects any active VPN (Virtual Private Network) connection. Stops routing your internet traffic through a VPN server, which might affect connection speed or access to content.
        """
        return self._run("disconnect_vpn")

    @vf.tool
    def check_installed_apps(self) -> str:
        """Returns the name of all installed apps on the phone."""
        return self._run("check_installed_apps")

    @vf.tool
    def check_app_status(self, app_name: str) -> str:
        """Checks detailed information about a specific app. Shows its permissions and background data usage settings.
        """
        return self._run("check_app_status", app_name=app_name)

    @vf.tool
    def check_app_permissions(self, app_name: str) -> str:
        """Checks what permissions a specific app currently has. Shows if the app has access to features like storage, camera, location, etc.
        """
        return self._run("check_app_permissions", app_name=app_name)

    @vf.tool
    def grant_app_permission(self, app_name: str, permission: str) -> str:
        """Gives a specific permission to an app (like access to storage, camera, or location). Required for some app functions to work properly.
        """
        return self._run("grant_app_permission", app_name=app_name, permission=permission)

    @vf.tool
    def can_send_mms(self) -> str:
        """Checks if the default messaging app can send MMS messages."""
        return self._run("can_send_mms")

    @vf.tool
    def reboot_device(self) -> str:
        """Restarts your phone completely. This can help resolve many temporary software glitches by refreshing all running services and connections.
        """
        return self._run("reboot_device")

    @vf.tool
    def check_payment_request(self) -> str:
        """Checks if the agent has sent you a payment request."""
        return self._run("check_payment_request")

    @vf.tool
    def make_payment(self) -> str:
        """Makes a payment for the bill that the agent has sent you."""
        return self._run("make_payment")

# FastMCP advertises a tool's docstring as written, indentation and all, while
# the replay surface advertises the frozen schema. Stripping the block indent
# here is what makes the two strings equal, which
# `test_both_surfaces_advertise_the_same_tools` checks.
for _name in TOOL_NAMES | {STOP_TOOL}:
    _method = getattr(TelecomSoloToolset, _name)
    _method.__doc__ = inspect.cleandoc(_method.__doc__)
del _name, _method


class TelecomSoloData(vf.TaskData):
    task_id: str
    family: str
    reward_basis: tuple[str, ...]
    assertions: int
    reference_actions: int
    split: Literal["train", "eval", "qualification"]


class TelecomSoloTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class TelecomSoloTask(
    vf.Task[TelecomSoloData, TelecomSoloState, TelecomSoloTaskConfig]
):
    @property
    def key(self) -> str:
        return self.data.task_id

    @classmethod
    def toolsets(cls, config: TelecomSoloTaskConfig) -> list[vf.Toolset]:
        return [TelecomSoloToolset(config.tools)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        trace.state.index = self.data.idx or 0
        trace.state.split = self.data.split

    @staticmethod
    def _actions(trace: vf.Trace) -> list[dict[str, Any]] | None:
        """Recover only model-authored tool calls from the final trace branch.

        A turn that said something instead of calling something is not an
        action: solo mode has nobody to say it to, and a trace that carries one
        is a trace this environment cannot price, so it scores nothing rather
        than being scored on a guess about what was meant.
        """
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
        allowed = TOOL_NAMES | {STOP_TOOL}
        for message in messages:
            calls = message.tool_calls or []
            if not calls:
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
                    arguments = json.loads(call.arguments)
                except (TypeError, ValueError, RecursionError):
                    return None
                if not isinstance(arguments, dict):
                    return None
                actions.append({"tool": call.name, "arguments": arguments})
        return actions

    @vf.reward(weight=1.0)
    async def ticket_resolved(self, trace: vf.Trace) -> float:
        actions = self._actions(trace)
        if actions is None:
            return 0.0
        result = TelecomSoloEnvironment(self.data.split).replay(
            self.data.idx or 0, actions
        )
        return float(result["reward"]["reward"])

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The reference solution scores the point and an empty episode does not.

        Both halves matter. The first says the task is solvable with the tools
        this package ships; the second says the reward is not already sitting
        on the floor, which for 32 of these tickets it very nearly is — their
        assertion holds from the first turn, and only the required
        `transfer_to_human_agents` call separates giving up correctly from
        doing nothing at all.
        """
        del runtime
        environment = TelecomSoloEnvironment(self.data.split)
        index = self.data.idx or 0
        good = environment.replay(index, environment.reference_actions(index))
        bad = environment.replay(index, [{"tool": STOP_TOOL, "arguments": {}}])
        return good["reward"]["reward"] == 1.0 and bad["reward"]["reward"] == 0.0


class TelecomSoloConfig(vf.TasksetConfig):
    split: Literal["train", "eval", "qualification"] = "train"
    task: TelecomSoloTaskConfig = TelecomSoloTaskConfig()


class TelecomSoloTaskset(vf.Taskset[TelecomSoloTask, TelecomSoloConfig]):
    def load(self) -> Iterator[TelecomSoloTask]:
        environment = TelecomSoloEnvironment(self.config.split)
        for index in range(len(environment)):
            task = environment.task(index)
            metadata = task["metadata"]
            yield TelecomSoloTask(
                TelecomSoloData(
                    idx=index,
                    prompt=task["prompt"],
                    network_allow=[],
                    # The readable key, not the replay id: the replay surface
                    # shortens it to a digest for a consumer that caps ids at
                    # 128 characters, and Verifiers has no such cap. Taking
                    # the key keeps this path exactly as it was.
                    task_id=metadata["key"],
                    family=metadata["family"],
                    reward_basis=tuple(metadata["reward_basis"]),
                    assertions=metadata["assertions"],
                    reference_actions=metadata["reference_actions"],
                    split=self.config.split,
                ),
                self.config.task,
            )


__all__ = ["TelecomSoloTaskset"]


if __name__ == "__main__":
    TelecomSoloToolset.run()
