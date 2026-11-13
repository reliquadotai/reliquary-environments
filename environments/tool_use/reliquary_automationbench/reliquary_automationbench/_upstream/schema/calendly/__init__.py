# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Calendly schema model definitions used by AutomationBench."""

from reliquary_automationbench._upstream.schema.calendly.base import (
    CalendlyRecord,
    CalendlyState,
    generate_calendly_uri,
    generate_calendly_uuid,
)
from reliquary_automationbench._upstream.schema.calendly.event import ScheduledEvent
from reliquary_automationbench._upstream.schema.calendly.event_type import EventType
from reliquary_automationbench._upstream.schema.calendly.invitee import Invitee, QuestionAnswer
from reliquary_automationbench._upstream.schema.calendly.user import User

# Resolve forward references declared in CalendlyState after all imports are complete
CalendlyState.model_rebuild()

__all__ = [
    # Base
    "CalendlyRecord",
    "CalendlyState",
    "generate_calendly_uuid",
    "generate_calendly_uri",
    # Models
    "User",
    "EventType",
    "ScheduledEvent",
    "Invitee",
    "QuestionAnswer",
]
