"""Scripted episodes built from a task's own assertions, for tests and parity.

Upstream ships no reference solutions. These probes are the next best thing
for checking the port: an action list that reads the seeded services and
then writes what the positive assertions look for (an email to the named
recipient with the named phrases, a Slack message in the named channel, a
Salesforce field set to the named value, a sheet row holding the named
cell). Some tasks pass on them, most fail on a guard or a detail, and both
outcomes exercise the world and the rubric on real writes. They are not
solutions and say nothing about how hard a task is.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from reliquary_automationbench.corpus import Task

_SALESFORCE_OBJECTS = {
    "contacts": "Contact",
    "leads": "Lead",
    "accounts": "Account",
    "opportunities": "Opportunity",
    "cases": "Case",
}


def _texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, int, float))]
    return []


def _raw_email(to: str, subject: str, body: str) -> str:
    message = f"To: {to}\r\nSubject: {subject}\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n{body}"
    return base64.urlsafe_b64encode(message.encode("utf-8")).decode("ascii")


def _fetch(method: str, url: str, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments: dict[str, Any] = {"method": method, "url": url}
    if params is not None:
        arguments["params"] = json.dumps(params)
    if body is not None:
        arguments["body"] = json.dumps(body)
    return {"tool": "api_fetch", "arguments": arguments}


def probe_actions(task: Task) -> list[dict[str, Any]]:
    services = [key for key in task.initial_state if key != "meta"]
    actions: list[dict[str, Any]] = [
        {"tool": "api_search", "arguments": {"query": f"{services[0] if services else 'gmail'} list", "top_k": 3}},
        {"tool": "base64_encode", "arguments": {"text": task.name}},
    ]
    if "gmail" in services:
        actions.append(_fetch("GET", "https://gmail.googleapis.com/gmail/v1/users/me/messages", params={"maxResults": 5}))
    if "salesforce" in services:
        actions.append(_fetch("GET", "https://yourinstance.salesforce.com/services/data/v61.0/query", params={"q": "SELECT Id, Name FROM Contact LIMIT 3"}))
    if "slack" in services:
        actions.append(_fetch("GET", "https://slack.com/api/conversations.list"))
    for assertion in task.assertions:
        kind = str(assertion.get("type", ""))
        if "_not_" in kind or kind.endswith("_not_exists") or assertion.get("excluded") is True:
            continue
        if kind.startswith("gmail_message_sent") or kind == "gmail_email_body_contains":
            to = assertion.get("to") or assertion.get("to_contains") or "team@example.com"
            subject = assertion.get("subject") or assertion.get("subject_contains") or "Update"
            words = _texts(assertion.get("body_contains")) + _texts(assertion.get("text_contains"))
            actions.append(
                _fetch(
                    "POST",
                    "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                    body={"raw": _raw_email(str(to), str(subject), "\n".join(words) or "Done.")},
                )
            )
        elif kind.startswith("slack_message"):
            channel = assertion.get("channel_name") or assertion.get("channel") or assertion.get("channel_id") or "general"
            words = _texts(assertion.get("text_contains")) or _texts(assertion.get("text")) or ["Done."]
            actions.append(_fetch("POST", "https://slack.com/api/chat.postMessage", body={"channel": channel, "text": "\n".join(words)}))
        elif kind == "salesforce_field_equals":
            obj = _SALESFORCE_OBJECTS.get(str(assertion.get("collection")))
            if obj and assertion.get("record_id"):
                actions.append(
                    _fetch(
                        "PATCH",
                        f"https://yourinstance.salesforce.com/services/data/v61.0/sobjects/{obj}/{assertion['record_id']}",
                        body={str(assertion.get("field")): assertion.get("value")},
                    )
                )
        elif kind == "google_sheets_row_exists" and assertion.get("spreadsheet_id"):
            cell = assertion.get("cell_contains") or "x"
            actions.append(
                _fetch(
                    "POST",
                    f"https://sheets.googleapis.com/v4/spreadsheets/{assertion['spreadsheet_id']}/values/A1:append",
                    body={"values": [[str(cell)]]},
                    params={"valueInputOption": "RAW"},
                )
            )
    actions.append({"final": "Done."})
    return actions


__all__ = ["probe_actions"]


_PROBE_BODY = {
    "name": "Probe",
    "title": "Probe",
    "summary": "Probe",
    "subject": "Probe",
    "text": "Probe",
    "body": "Probe",
    "email": "probe@example.com",
    "start_time": "2026-03-01T10:00:00Z",
    "end_time": "2026-03-01T11:00:00Z",
}


def fuzz_actions(task: Task, count: int = 24) -> list[dict[str, Any]]:
    """Endpoint calls drawn from the task's own services, with placeholder ids and a
    generic body: most fail validation, many create records with default ids and
    timestamps, which is what a determinism check needs to exercise."""
    import random

    from reliquary_automationbench._upstream.tools.api.search import _compute_url, _load_schemas

    services = {key for key in task.initial_state if key != "meta"} | {"chatgpt"}
    aliases = {"openai": "chatgpt"}
    endpoints = []
    for api, schema in sorted(_load_schemas().items()):
        if aliases.get(api, api) not in services:
            continue
        for endpoint in schema.get("endpoints", []):
            endpoints.append((api, schema.get("baseUrl", ""), endpoint))
    if not endpoints:
        return [{"final": "Nothing to do."}]
    ids = [str(v) for v in _ids(task.initial_state)][:50] or ["1"]
    rng = random.Random(task.name)
    actions: list[dict[str, Any]] = []
    for _ in range(count):
        api, base, endpoint = rng.choice(endpoints)
        url = _compute_url(api, base, endpoint.get("path", ""))
        while "{" in url and "}" in url[url.index("{"):]:
            start = url.index("{")
            url = url[:start] + rng.choice(ids) + url[url.index("}", start) + 1:]
        method = endpoint.get("method", "GET")
        body = None if method in ("GET", "DELETE") else dict(_PROBE_BODY)
        actions.append(_fetch(method, url, body=body))
    actions.append({"final": "Done."})
    return actions


def _ids(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("id", "Id") and isinstance(item, (str, int)):
                yield item
            else:
                yield from _ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from _ids(item)
