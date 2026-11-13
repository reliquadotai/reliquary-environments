# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Zoom schema model definitions used by AutomationBench."""

from reliquary_automationbench._upstream.schema.zoom.base import (
    ZoomRecord,
    ZoomState,
    generate_zoom_meeting_id,
    generate_zoom_uuid,
)
from reliquary_automationbench._upstream.schema.zoom.meeting import Meeting
from reliquary_automationbench._upstream.schema.zoom.participant import Participant
from reliquary_automationbench._upstream.schema.zoom.recording import Recording
from reliquary_automationbench._upstream.schema.zoom.registrant import Registrant
from reliquary_automationbench._upstream.schema.zoom.webinar import Webinar

# Resolve forward references declared in ZoomState after all imports are complete
ZoomState.model_rebuild()

__all__ = [
    "ZoomRecord",
    "ZoomState",
    "generate_zoom_meeting_id",
    "generate_zoom_uuid",
    "Meeting",
    "Recording",
    "Registrant",
    "Participant",
    "Webinar",
]
