# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Schema models for Google Calendar."""

from reliquary_automationbench._upstream.schema.google_calendar.base import (
    GoogleCalendarRecord,
    GoogleCalendarState,
    generate_google_calendar_id,
)
from reliquary_automationbench._upstream.schema.google_calendar.calendar import Calendar
from reliquary_automationbench._upstream.schema.google_calendar.event import CalendarEvent

# Resolve forward references once all imports are complete
GoogleCalendarState.model_rebuild()

__all__ = [
    "GoogleCalendarRecord",
    "GoogleCalendarState",
    "generate_google_calendar_id",
    "Calendar",
    "CalendarEvent",
]
